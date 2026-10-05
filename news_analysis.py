"""
Binance News Sentiment Bot.

Every REPEAT_EVERY minutes the bot:
  1. pulls recent headlines from every RSS/Atom feed listed in FEEDS_FILE,
  2. matches them to the coins defined in KEYWORDS,
  3. scores each headline with the VADER sentiment model,
  4. buys a coin when its average sentiment is positive enough and it is not
     already held, and sells it when the sentiment turns negative.

Run `python news_analysis.py --help` for command line options.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from statistics import mean

import aiohttp
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


############################################
#     USER INPUT VARIABLES LIVE BELOW      #
# You may edit those to configure your bot #
############################################

# Use the Binance Spot testnet (True) or your real account (False).
# Can also be set with the BOT_TESTNET environment variable or --live flag.
TESTNET = _env_bool('BOT_TESTNET', True)

# Dry run: analyse news and log the trades the bot WOULD make, without placing
# any orders. Holdings are simulated in a separate file.
# Can also be set with the BOT_DRY_RUN environment variable or --dry-run flag.
DRY_RUN = _env_bool('BOT_DRY_RUN', False)

# Coins to look for in headlines.
# The key MUST be the symbol used for that coin on Binance.
# The list holds the keywords for that coin. Matching is case-insensitive and
# whole-word only, so 'ETH' will not match 'Ethereum' (list both if you want both).
KEYWORDS = {
    'BTC': ['BTC', 'Bitcoin'],
    'ETH': ['ETH', 'Ethereum', 'Ether'],
    'XRP': ['XRP', 'Ripple'],
    'BNB': ['BNB', 'Binance Coin'],
    'SOL': ['SOL', 'Solana'],
    'LTC': ['LTC', 'Litecoin'],
    'XLM': ['XLM', 'Stellar Lumens'],
    # 'BCH': ['BCH', 'Bitcoin Cash'],
    # 'DOGE': ['DOGE', 'Dogecoin'],
}

# Amount to spend on each buy, in the PAIRING currency.
# 100 with USDT pairing buys 100 USDT worth of the coin.
QUANTITY = Decimal('100')

# What to pair each coin with. Avoid pairing with one of the coins in KEYWORDS.
PAIRING = 'USDT'

# Average VADER compound score (between -1 and 1) above which the bot buys...
SENTIMENT_THRESHOLD = 0.0
# ...and below which it sells a coin it holds.
NEGATIVE_SENTIMENT_THRESHOLD = 0.0

# Minimum number of matching headlines needed before the bot acts on a coin.
# 1 is rarely representative of the overall sentiment.
MINIMUM_ARTICLES = 3

# How often to run the analysis and trade check, in minutes.
REPEAT_EVERY = 60

# Only headlines published in the last HOURS_PAST hours are analysed.
HOURS_PAST = 24

# CSV file with one RSS/Atom feed URL per row (first row is a header).
FEEDS_FILE = 'Crypto feeds.csv'

# Seconds to wait for a single feed before giving up on it.
FEED_TIMEOUT = 20

# Maximum number of feeds downloaded at the same time.
MAX_CONCURRENT_FEEDS = 20


############################################
#        END OF USER INPUT VARIABLES       #
#             Edit with care               #
############################################

log = logging.getLogger('news-bot')

USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'
)


# --------------------------------------------------------------------------- #
# News                                                                        #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Headline:
    source: str
    title: str
    published: datetime


def load_feeds(path: str | Path) -> list[str]:
    """Read feed URLs from the first column of a CSV file, skipping the header."""
    feeds = []
    # utf-8-sig strips the byte order mark some editors add to CSV files
    with open(path, newline='', encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            if row and row[0].strip().startswith('http'):
                feeds.append(row[0].strip())
    # drop duplicates but keep the original order
    return list(dict.fromkeys(feeds))


def parse_feed(body: bytes | str, source: str, cutoff: datetime) -> list[Headline]:
    """Return every entry of an RSS/Atom document published after `cutoff`."""
    parsed = feedparser.parse(body)
    headlines = []
    for entry in parsed.entries:
        title = ' '.join((entry.get('title') or '').split())
        published = entry.get('published_parsed') or entry.get('updated_parsed')
        if not title or published is None:
            continue
        # feedparser always normalises dates to UTC
        published_at = datetime(*published[:6], tzinfo=timezone.utc)
        if published_at >= cutoff:
            headlines.append(Headline(source, title, published_at))
    return headlines


async def _fetch_feed(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                      url: str, cutoff: datetime) -> list[Headline]:
    async with semaphore:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                body = await response.read()
        except Exception as e:  # network errors, timeouts, HTTP errors...
            log.debug('Could not fetch %s: %r', url, e)
            return []
    try:
        return parse_feed(body, url, cutoff)
    except Exception as e:
        log.debug('Could not parse %s: %r', url, e)
        return []


async def fetch_headlines(feeds: list[str], hours_past: float) -> list[Headline]:
    """Download all feeds concurrently and return de-duplicated recent headlines."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_past)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FEEDS)
    timeout = aiohttp.ClientTimeout(total=FEED_TIMEOUT)
    headers = {'User-Agent': USER_AGENT}

    start = time.monotonic()
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        results = await asyncio.gather(
            *(_fetch_feed(session, semaphore, url, cutoff) for url in feeds)
        )

    working = sum(1 for r in results if r)
    headlines = dedupe_headlines(h for result in results for h in result)
    log.info('Fetched %d recent headlines from %d/%d feeds in %.1fs',
             len(headlines), working, len(feeds), time.monotonic() - start)
    return headlines


def dedupe_headlines(headlines) -> list[Headline]:
    """Syndicated stories appear on several sites; count each title only once."""
    seen = set()
    unique = []
    for headline in headlines:
        key = re.sub(r'\W+', ' ', headline.title).strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(headline)
    return unique


# --------------------------------------------------------------------------- #
# Sentiment                                                                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CoinSentiment:
    coin: str
    score: float  # average VADER compound score, -1..1
    articles: int


def build_keyword_patterns(keywords: dict[str, list[str]]) -> dict[str, re.Pattern]:
    """One case-insensitive, whole-word regex per coin."""
    patterns = {}
    for coin, words in keywords.items():
        alternatives = '|'.join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        patterns[coin] = re.compile(rf'(?<![A-Za-z0-9])(?:{alternatives})(?![A-Za-z0-9])',
                                    re.IGNORECASE)
    return patterns


def categorise_headlines(headlines: list[Headline],
                         keywords: dict[str, list[str]]) -> dict[str, list[Headline]]:
    """Group headlines by the coins they mention. A headline can match several coins."""
    patterns = build_keyword_patterns(keywords)
    categorised: dict[str, list[Headline]] = {coin: [] for coin in keywords}
    for headline in headlines:
        for coin, pattern in patterns.items():
            if pattern.search(headline.title):
                categorised[coin].append(headline)
    return categorised


def analyse_sentiment(categorised: dict[str, list[Headline]],
                      analyzer: SentimentIntensityAnalyzer) -> dict[str, CoinSentiment]:
    """Average compound sentiment for every coin that has at least one headline."""
    sentiment = {}
    for coin, coin_headlines in categorised.items():
        if not coin_headlines:
            continue
        scores = [analyzer.polarity_scores(h.title)['compound'] for h in coin_headlines]
        sentiment[coin] = CoinSentiment(coin, mean(scores), len(scores))
    return sentiment


# --------------------------------------------------------------------------- #
# Trading                                                                     #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SymbolRules:
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal


def parse_symbol_rules(info: dict) -> SymbolRules:
    """Extract the quantity rules from a Binance get_symbol_info() response."""
    filters = {f['filterType']: f for f in info.get('filters', [])}
    lot = filters.get('LOT_SIZE', {})
    notional = filters.get('NOTIONAL') or filters.get('MIN_NOTIONAL') or {}
    return SymbolRules(
        step_size=Decimal(lot.get('stepSize', '0')),
        min_qty=Decimal(lot.get('minQty', '0')),
        min_notional=Decimal(notional.get('minNotional', '0')),
    )


def round_step(quantity: Decimal, step: Decimal) -> Decimal:
    """Round a quantity DOWN to the symbol's step size."""
    if step <= 0:
        return quantity
    return (quantity / step).to_integral_value(rounding=ROUND_DOWN) * step


def format_decimal(value: Decimal) -> str:
    """Plain string without exponent or trailing zeros, as Binance expects."""
    return format(value.normalize(), 'f')


def net_filled_quantity(order: dict, base_asset: str) -> Decimal:
    """Quantity actually received from an order, minus commissions paid in the coin."""
    quantity = Decimal(order.get('executedQty', '0'))
    for fill in order.get('fills', []):
        if fill.get('commissionAsset') == base_asset:
            quantity -= Decimal(fill.get('commission', '0'))
    return max(quantity, Decimal('0'))


class Holdings:
    """Coins bought by the bot, persisted to a JSON file between runs."""

    def __init__(self, path: Path, coins):
        self.path = path
        self.amounts: dict[str, Decimal] = {}
        if path.is_file():
            try:
                with open(path) as file:
                    saved = json.load(file)
                self.amounts = {c: Decimal(str(a)) for c, a in saved.items()}
            except (OSError, ValueError, InvalidOperation) as e:
                log.error('Could not read %s (%s), starting with empty holdings', path, e)
        for coin in coins:
            self.amounts.setdefault(coin, Decimal('0'))

    def __getitem__(self, coin: str) -> Decimal:
        return self.amounts.get(coin, Decimal('0'))

    def __setitem__(self, coin: str, amount: Decimal):
        self.amounts[coin] = max(amount, Decimal('0'))

    def held(self) -> dict[str, Decimal]:
        return {c: a for c, a in self.amounts.items() if a > 0}

    def save(self):
        # write to a temporary file first so a crash never leaves a corrupt file
        tmp = self.path.with_suffix(self.path.suffix + '.tmp')
        with open(tmp, 'w') as file:
            json.dump({c: float(a) for c, a in self.amounts.items()}, file, indent=4)
        os.replace(tmp, self.path)


class Trader:
    def __init__(self, client, holdings: Holdings, pairing: str, quantity: Decimal,
                 dry_run: bool):
        self.client = client
        self.holdings = holdings
        self.pairing = pairing
        self.quantity = quantity
        self.dry_run = dry_run
        self._rules: dict[str, SymbolRules] = {}

    def symbol(self, coin: str) -> str:
        return coin + self.pairing

    def rules(self, coin: str) -> SymbolRules | None:
        symbol = self.symbol(coin)
        if symbol not in self._rules:
            info = self.client.get_symbol_info(symbol)
            if not info or info.get('status') != 'TRADING':
                return None
            self._rules[symbol] = parse_symbol_rules(info)
        return self._rules[symbol]

    def tradable_coins(self, coins) -> list[str]:
        tradable = []
        for coin in coins:
            try:
                if self.rules(coin):
                    tradable.append(coin)
                    continue
            except Exception as e:
                log.warning('Could not load %s info: %s', self.symbol(coin), e)
                continue
            log.warning('%s is not available for trading, ignoring %s',
                        self.symbol(coin), coin)
        return tradable

    def price(self, coin: str) -> Decimal:
        return Decimal(self.client.get_symbol_ticker(symbol=self.symbol(coin))['price'])

    def act(self, sentiment: dict[str, CoinSentiment], coins) -> None:
        """Make at most one buy or sell decision per coin."""
        for coin in coins:
            s = sentiment.get(coin)
            if s is None or s.articles < MINIMUM_ARTICLES:
                count = s.articles if s else 0
                log.info('%-5s %2d headline(s), not enough to act', coin, count)
                continue

            held = self.holdings[coin]
            if s.score > SENTIMENT_THRESHOLD and held == 0:
                log.info('%-5s %2d headline(s), score %+.3f -> BUY', coin, s.articles, s.score)
                self._try(self.buy, coin)
            elif s.score < NEGATIVE_SENTIMENT_THRESHOLD and held > 0:
                log.info('%-5s %2d headline(s), score %+.3f -> SELL', coin, s.articles, s.score)
                self._try(self.sell, coin)
            else:
                state = 'held' if held > 0 else 'not held'
                log.info('%-5s %2d headline(s), score %+.3f, %s -> no action',
                         coin, s.articles, s.score, state)

    def _try(self, action, coin):
        try:
            action(coin)
        except Exception as e:  # BinanceAPIException, network errors...
            log.error('Order for %s failed: %s', self.symbol(coin), e)

    def buy(self, coin: str) -> None:
        symbol = self.symbol(coin)
        rules = self.rules(coin)
        if self.quantity < rules.min_notional:
            log.warning('QUANTITY %s %s is below the %s minimum order of %s',
                        self.quantity, self.pairing, symbol, rules.min_notional)
            return

        if self.dry_run:
            price = self.price(coin)
            amount = round_step(self.quantity / price, rules.step_size)
            self.holdings[coin] = amount
            log.info('[DRY RUN] would buy %s %s for %s %s at %s',
                     format_decimal(amount), coin, self.quantity, self.pairing, price)
            return

        # quoteOrderQty lets Binance work out the coin amount for the money we spend
        order = self.client.order_market_buy(symbol=symbol,
                                             quoteOrderQty=format_decimal(self.quantity))
        amount = net_filled_quantity(order, coin)
        self.holdings[coin] = self.holdings[coin] + amount
        log.info('Order %s: bought %s %s for %s %s',
                 order.get('orderId'), format_decimal(amount), coin,
                 order.get('cummulativeQuoteQty'), self.pairing)

    def sell(self, coin: str) -> None:
        symbol = self.symbol(coin)
        rules = self.rules(coin)
        held = self.holdings[coin]

        if self.dry_run:
            price = self.price(coin)
            log.info('[DRY RUN] would sell %s %s at %s (~%.2f %s)',
                     format_decimal(held), coin, price, held * price, self.pairing)
            self.holdings[coin] = Decimal('0')
            return

        # never try to sell more than we actually have on the exchange
        free = Decimal(self.client.get_asset_balance(asset=coin)['free'])
        amount = round_step(min(held, free), rules.step_size)
        price = self.price(coin)
        if amount <= 0 or amount < rules.min_qty or amount * price < rules.min_notional:
            log.warning('Cannot sell %s %s (free balance %s): below the %s minimum order. '
                        'Forgetting this position.', format_decimal(held), coin,
                        format_decimal(free), symbol)
            self.holdings[coin] = Decimal('0')
            return

        order = self.client.order_market_sell(symbol=symbol, quantity=format_decimal(amount))
        sold = Decimal(order.get('executedQty', '0'))
        remaining = held - sold
        # whatever is left after rounding is dust that can't be sold on its own
        if remaining < rules.min_qty or remaining < rules.step_size:
            remaining = Decimal('0')
        self.holdings[coin] = remaining
        log.info('Order %s: sold %s %s for %s %s',
                 order.get('orderId'), format_decimal(sold), coin,
                 order.get('cummulativeQuoteQty'), self.pairing)


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #

def create_client(testnet: bool, dry_run: bool):
    from binance.client import Client

    if testnet:
        key = os.getenv('BINANCE_TESTNET_API_KEY') or os.getenv('binance_api_stalkbot_testnet')
        secret = (os.getenv('BINANCE_TESTNET_API_SECRET')
                  or os.getenv('binance_secret_stalkbot_testnet'))
    else:
        key = os.getenv('BINANCE_API_KEY') or os.getenv('binance_api_stalkbot_live')
        secret = os.getenv('BINANCE_API_SECRET') or os.getenv('binance_secret_stalkbot_live')

    if not (key and secret) and not dry_run:
        prefix = 'BINANCE_TESTNET_' if testnet else 'BINANCE_'
        sys.exit(f'Missing API credentials: set {prefix}API_KEY and {prefix}API_SECRET '
                 'environment variables, or run with --dry-run.')

    return Client(key, secret, testnet=testnet)


def state_file(testnet: bool, dry_run: bool) -> Path:
    name = 'coins_in_hand.json'
    if testnet:
        name = 'testnet_' + name
    if dry_run:
        name = 'dryrun_' + name
    return Path(name)


def run_once(trader: Trader, feeds: list[str], coins: list[str],
             analyzer: SentimentIntensityAnalyzer) -> None:
    headlines = asyncio.run(fetch_headlines(feeds, HOURS_PAST))
    categorised = categorise_headlines(headlines, {c: KEYWORDS[c] for c in coins})
    sentiment = analyse_sentiment(categorised, analyzer)

    for coin, coin_headlines in categorised.items():
        for h in coin_headlines:
            log.debug('%-5s %+.3f  %s', coin, analyzer.polarity_scores(h.title)['compound'],
                      h.title)

    trader.act(sentiment, coins)
    trader.holdings.save()

    held = trader.holdings.held()
    if held:
        log.info('Holdings: %s', ', '.join(f'{c} {format_decimal(a)}' for c, a in held.items()))
    else:
        log.info('Holdings: none')


def main(argv=None) -> None:
    global TESTNET, DRY_RUN

    parser = argparse.ArgumentParser(description='Trade on Binance based on crypto news sentiment.')
    parser.add_argument('--once', action='store_true', help='run a single check and exit')
    parser.add_argument('--dry-run', action='store_true', help='log trades without placing orders')
    parser.add_argument('--live', action='store_true', help='trade on the real Binance account')
    parser.add_argument('-v', '--verbose', action='store_true', help='show every headline and feed error')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    # keep third party libraries quiet even in verbose mode
    for noisy in ('urllib3', 'asyncio', 'chardet', 'charset_normalizer'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.live:
        TESTNET = False
    if args.dry_run:
        DRY_RUN = True

    mode = 'TESTNET' if TESTNET else 'LIVE'
    log.info('Starting in %s mode%s', mode, ' (dry run, no orders will be placed)' if DRY_RUN else '')

    client = create_client(TESTNET, DRY_RUN)
    holdings = Holdings(state_file(TESTNET, DRY_RUN), KEYWORDS)
    trader = Trader(client, holdings, PAIRING, QUANTITY, DRY_RUN)

    coins = trader.tradable_coins(KEYWORDS)
    if not coins:
        sys.exit('None of the configured coins can be traded, check KEYWORDS and PAIRING.')

    feeds = load_feeds(FEEDS_FILE)
    analyzer = SentimentIntensityAnalyzer()
    log.info('Watching %s against %s using %d feeds', ', '.join(coins), PAIRING, len(feeds))

    iteration = 0
    try:
        while True:
            iteration += 1
            log.info('--- Iteration %d ---', iteration)
            try:
                run_once(trader, feeds, coins, analyzer)
            except Exception:
                # one bad iteration (e.g. a network outage) should not kill the bot
                log.exception('Iteration %d failed', iteration)
            if args.once:
                break
            log.info('Next check in %d minutes (Ctrl-C to stop)', REPEAT_EVERY)
            time.sleep(60 * REPEAT_EVERY)
    except KeyboardInterrupt:
        log.info('Stopping')
    finally:
        holdings.save()


if __name__ == '__main__':
    main()
