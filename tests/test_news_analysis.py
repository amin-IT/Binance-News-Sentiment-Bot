import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import news_analysis as bot  # noqa: E402

NOW = datetime.now(timezone.utc)


def rss(*items):
    body = ''.join(
        f'<item><title>{title}</title><pubDate>{date.strftime("%a, %d %b %Y %H:%M:%S GMT")}'
        '</pubDate></item>'
        for title, date in items
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>{body}</channel></rss>'


def headline(title):
    return bot.Headline('feed', title, NOW)


# ---------------------------------------------------------------- news

def test_parse_feed_keeps_all_recent_items_only():
    body = rss(('Fresh one', NOW - timedelta(hours=1)),
               ('Fresh two', NOW - timedelta(hours=5)),
               ('Too old', NOW - timedelta(hours=30)))
    result = bot.parse_feed(body, 'feed', NOW - timedelta(hours=24))
    assert [h.title for h in result] == ['Fresh one', 'Fresh two']


def test_parse_feed_handles_atom():
    body = ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>x</title>'
            f'<entry><title>Atom headline</title><updated>{NOW.isoformat()}</updated></entry></feed>')
    result = bot.parse_feed(body, 'feed', NOW - timedelta(hours=1))
    assert [h.title for h in result] == ['Atom headline']


def test_parse_feed_ignores_garbage():
    assert bot.parse_feed(b'not xml at all', 'feed', NOW) == []


def test_load_feeds_skips_header_bom_and_duplicates(tmp_path):
    path = tmp_path / 'feeds.csv'
    path.write_text('﻿Feeds\nhttps://a.com/feed\n\nhttps://b.com/rss\nhttps://a.com/feed\n',
                    encoding='utf-8')
    assert bot.load_feeds(path) == ['https://a.com/feed', 'https://b.com/rss']


def test_repository_feeds_file_loads():
    feeds = bot.load_feeds(Path(__file__).resolve().parents[1] / bot.FEEDS_FILE)
    assert len(feeds) >= 15
    assert all(f.startswith('http') for f in feeds)


def test_dedupe_headlines_ignores_case_and_punctuation():
    result = bot.dedupe_headlines([headline('Bitcoin hits $100k!'),
                                   headline('bitcoin hits $100k'),
                                   headline('Ethereum upgrade')])
    assert [h.title for h in result] == ['Bitcoin hits $100k!', 'Ethereum upgrade']


# ---------------------------------------------------------------- sentiment

def test_categorise_is_case_insensitive_and_whole_word():
    keywords = {'BTC': ['BTC', 'Bitcoin'], 'ETH': ['ETH'], 'SOL': ['SOL', 'Solana']}
    headlines = [headline('BITCOIN rallies'),
                 headline('Ethereum news'),          # 'ETH' must not match inside 'Ethereum'
                 headline('$ETH and btc move up'),
                 headline('Consolidation continues')]  # 'sol' must not match inside a word
    result = bot.categorise_headlines(headlines, keywords)
    assert [h.title for h in result['BTC']] == ['BITCOIN rallies', '$ETH and btc move up']
    assert [h.title for h in result['ETH']] == ['$ETH and btc move up']
    assert result['SOL'] == []


def test_analyse_sentiment_averages_and_skips_empty():
    categorised = {'BTC': [headline('Bitcoin is great and wonderful'),
                           headline('Bitcoin crash is a terrible disaster')],
                   'ETH': []}
    result = bot.analyse_sentiment(categorised, SentimentIntensityAnalyzer())
    assert set(result) == {'BTC'}
    assert result['BTC'].articles == 2
    assert -1 <= result['BTC'].score <= 1


# ---------------------------------------------------------------- exchange maths

SYMBOL_INFO = {
    'symbol': 'BTCUSDT', 'status': 'TRADING',
    'filters': [
        {'filterType': 'PRICE_FILTER', 'tickSize': '0.01'},
        {'filterType': 'LOT_SIZE', 'minQty': '0.00001000', 'stepSize': '0.00001000'},
        {'filterType': 'NOTIONAL', 'minNotional': '5.00000000'},
    ],
}


def test_parse_symbol_rules_finds_filters_by_type():
    rules = bot.parse_symbol_rules(SYMBOL_INFO)
    assert rules == bot.SymbolRules(Decimal('0.00001'), Decimal('0.00001'), Decimal('5'))


@pytest.mark.parametrize('qty, step, expected', [
    ('0.123456789', '0.00001', '0.12345'),
    ('12.9', '1', '12'),
    ('5.55', '0.1', '5.5'),
    ('3.3', '0', '3.3'),
])
def test_round_step_rounds_down(qty, step, expected):
    result = bot.round_step(Decimal(qty), Decimal(step))
    assert bot.format_decimal(result) == expected


def test_format_decimal_has_no_exponent():
    assert bot.format_decimal(Decimal('100')) == '100'
    assert bot.format_decimal(Decimal('0.00001000')) == '0.00001'


def test_net_filled_quantity_subtracts_coin_commission():
    order = {'executedQty': '0.00150000',
             'fills': [{'commission': '0.00000150', 'commissionAsset': 'BTC'},
                       {'commission': '0.001', 'commissionAsset': 'BNB'}]}
    assert bot.net_filled_quantity(order, 'BTC') == Decimal('0.0014985')


# ---------------------------------------------------------------- trading flow

class FakeClient:
    def __init__(self, free='1'):
        self.orders = []
        self.free = free

    def get_symbol_info(self, symbol):
        return dict(SYMBOL_INFO, symbol=symbol) if symbol != 'NOPEUSDT' else None

    def get_symbol_ticker(self, symbol):
        return {'symbol': symbol, 'price': '50000'}

    def get_asset_balance(self, asset):
        return {'asset': asset, 'free': self.free}

    def order_market_buy(self, symbol, quoteOrderQty):
        self.orders.append(('BUY', symbol, quoteOrderQty))
        return {'orderId': 1, 'executedQty': '0.002', 'cummulativeQuoteQty': '100',
                'fills': [{'commission': '0.000002', 'commissionAsset': 'BTC'}]}

    def order_market_sell(self, symbol, quantity):
        self.orders.append(('SELL', symbol, quantity))
        return {'orderId': 2, 'executedQty': quantity, 'cummulativeQuoteQty': '99'}


def make_trader(tmp_path, client=None, dry_run=False):
    holdings = bot.Holdings(tmp_path / 'state.json', ['BTC'])
    return bot.Trader(client or FakeClient(), holdings, 'USDT', Decimal('100'), dry_run)


def sentiment(score, articles=5):
    return {'BTC': bot.CoinSentiment('BTC', score, articles)}


def test_tradable_coins_drops_unknown_symbols(tmp_path):
    trader = make_trader(tmp_path)
    assert trader.tradable_coins(['BTC', 'NOPE']) == ['BTC']


def test_buys_on_positive_sentiment_with_quote_quantity(tmp_path):
    trader = make_trader(tmp_path)
    trader.act(sentiment(0.5), ['BTC'])
    assert trader.client.orders == [('BUY', 'BTCUSDT', '100')]
    assert trader.holdings['BTC'] == Decimal('0.001998')


def test_does_not_buy_twice_or_with_too_few_articles(tmp_path):
    trader = make_trader(tmp_path)
    trader.act(sentiment(0.5, articles=1), ['BTC'])
    assert trader.client.orders == []
    trader.act(sentiment(0.5), ['BTC'])
    trader.act(sentiment(0.5), ['BTC'])
    assert len(trader.client.orders) == 1


def test_sells_held_coins_limited_by_free_balance(tmp_path):
    client = FakeClient(free='0.0015')
    trader = make_trader(tmp_path, client)
    trader.holdings['BTC'] = Decimal('0.001998')
    trader.act(sentiment(-0.5), ['BTC'])
    assert client.orders == [('SELL', 'BTCUSDT', '0.0015')]
    # what is left is still above the minimum quantity so it is remembered
    assert trader.holdings['BTC'] == Decimal('0.000498')


def test_sell_rounds_to_step_and_clears_dust(tmp_path):
    client = FakeClient(free='1')
    trader = make_trader(tmp_path, client)
    trader.holdings['BTC'] = Decimal('0.001998')
    trader.act(sentiment(-0.5), ['BTC'])
    assert client.orders == [('SELL', 'BTCUSDT', '0.00199')]
    assert trader.holdings['BTC'] == 0


def test_sell_below_min_notional_forgets_position(tmp_path):
    client = FakeClient(free='0.00001')  # 0.5 USDT worth, below the 5 USDT minimum
    trader = make_trader(tmp_path, client)
    trader.holdings['BTC'] = Decimal('0.002')
    trader.act(sentiment(-0.5), ['BTC'])
    assert client.orders == []
    assert trader.holdings['BTC'] == 0


def test_failed_order_is_logged_not_raised(tmp_path):
    class Broken(FakeClient):
        def order_market_buy(self, symbol, quoteOrderQty):
            raise RuntimeError('insufficient balance')

    trader = make_trader(tmp_path, Broken())
    trader.act(sentiment(0.5), ['BTC'])
    assert trader.holdings['BTC'] == 0


def test_dry_run_places_no_orders_but_tracks_holdings(tmp_path):
    trader = make_trader(tmp_path, dry_run=True)
    trader.act(sentiment(0.5), ['BTC'])
    assert trader.client.orders == []
    assert trader.holdings['BTC'] == Decimal('0.002')
    trader.act(sentiment(-0.5), ['BTC'])
    assert trader.client.orders == []
    assert trader.holdings['BTC'] == 0


# ---------------------------------------------------------------- state file

def test_holdings_round_trip_and_old_format(tmp_path):
    path = tmp_path / 'coins_in_hand.json'
    # format written by the previous version of the bot
    path.write_text(json.dumps({'BTC': 0.0021, 'XRP': 0}))
    holdings = bot.Holdings(path, ['BTC', 'ETH'])
    assert holdings['BTC'] == Decimal('0.0021')
    assert holdings['ETH'] == 0
    assert holdings.held() == {'BTC': Decimal('0.0021')}

    holdings['ETH'] = Decimal('0.5')
    holdings.save()
    assert json.loads(path.read_text()) == {'BTC': 0.0021, 'XRP': 0.0, 'ETH': 0.5}


def test_holdings_survive_corrupt_file(tmp_path):
    path = tmp_path / 'coins_in_hand.json'
    path.write_text('{not json')
    holdings = bot.Holdings(path, ['BTC'])
    assert holdings['BTC'] == 0


def test_state_file_names():
    assert str(bot.state_file(True, False)) == 'testnet_coins_in_hand.json'
    assert str(bot.state_file(False, False)) == 'coins_in_hand.json'
    assert str(bot.state_file(False, True)) == 'dryrun_coins_in_hand.json'
