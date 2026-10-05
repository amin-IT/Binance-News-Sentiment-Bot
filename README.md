# Binance News Sentiment Bot

A Binance trading bot that buys and sells cryptocurrency based on the sentiment of
recent news headlines from the major crypto news outlets
(CoinDesk, Cointelegraph, Decrypt, The Block, Blockworks, Bitcoin Magazine and more).

> **Warning:** this is an experimental bot. News sentiment is a weak trading signal.
> Start on the testnet or with `--dry-run`, and only trade money you can afford to lose.

## How it works

Every `REPEAT_EVERY` minutes (60 by default) the bot:

1. Downloads every RSS/Atom feed in `Crypto feeds.csv` concurrently and keeps the
   headlines published in the last `HOURS_PAST` hours (24 by default). Duplicate
   headlines syndicated across several sites are counted once.
2. Matches headlines to coins using the keywords in `KEYWORDS`
   (case-insensitive, whole words only).
3. Scores every headline with the [VADER](https://github.com/cjhutto/vaderSentiment)
   sentiment model and averages the compound score (-1 to 1) for each coin.
4. For each coin with at least `MINIMUM_ARTICLES` headlines:
   - **buys** `QUANTITY` worth of `PAIRING` (100 USDT by default) if the score is
     above `SENTIMENT_THRESHOLD` and the bot doesn't already hold it;
   - **sells** what the bot bought if the score is below `NEGATIVE_SENTIMENT_THRESHOLD`.
5. Saves what it holds to `coins_in_hand.json` (`testnet_coins_in_hand.json` on the
   testnet), so it remembers its positions after a restart.

## Setup

Requires Python 3.9 or newer.

```bash
git clone https://github.com/amin-IT/Binance-News-Sentiment-Bot.git
cd Binance-News-Sentiment-Bot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### API keys

Keys are read from environment variables, never from the code.

| Mode    | Variables                                              | Where to get keys |
|---------|--------------------------------------------------------|-------------------|
| Testnet | `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET` | https://testnet.binance.vision |
| Live    | `BINANCE_API_KEY`, `BINANCE_API_SECRET`                 | Binance → API Management (enable *Spot Trading* only, never withdrawals) |

```bash
export BINANCE_TESTNET_API_KEY=your_key
export BINANCE_TESTNET_API_SECRET=your_secret
```

The variable names from the original version (`binance_api_stalkbot_testnet`, etc.)
still work.

## Usage

```bash
python news_analysis.py              # run on the testnet, checking every hour
python news_analysis.py --dry-run    # analyse and log trades without placing orders (no keys needed)
python news_analysis.py --once       # run a single check and exit (useful with cron)
python news_analysis.py --live       # trade on your real account
python news_analysis.py -v           # also show every matched headline and feed error
```

Stop the bot with `Ctrl-C`; it saves its holdings before exiting.

`TESTNET` and `DRY_RUN` can also be set with the `BOT_TESTNET` and `BOT_DRY_RUN`
environment variables (`true`/`false`).

## Configuration

Edit the block marked `USER INPUT VARIABLES` at the top of `news_analysis.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `KEYWORDS` | BTC, ETH, XRP, BNB, SOL, LTC, XLM | Coins to trade and the words that identify them in headlines. Keys must be Binance symbols. |
| `QUANTITY` | `100` | Amount of `PAIRING` to spend per buy. |
| `PAIRING` | `USDT` | Quote currency. Don't use one of the coins in `KEYWORDS`. |
| `SENTIMENT_THRESHOLD` | `0.0` | Buy when the average score is above this. |
| `NEGATIVE_SENTIMENT_THRESHOLD` | `0.0` | Sell when the average score is below this. |
| `MINIMUM_ARTICLES` | `3` | Headlines needed before acting on a coin. |
| `REPEAT_EVERY` | `60` | Minutes between checks. |
| `HOURS_PAST` | `24` | Maximum headline age in hours. |

To follow other news sites, add their RSS feed URLs to `Crypto feeds.csv`
(one per line). The list favours established newsrooms over blogs and exchange
marketing pages, since low-quality sites add noise to the sentiment. Feeds that are down or invalid are skipped automatically; run with
`-v` to see which ones fail.

## Running the tests

The tests use a fake Binance client and need no network access or API keys.

```bash
pip install -r requirements-dev.txt
pytest
```

## Credits

Originally written by [CyberPunkMetalHead](https://github.com/CyberPunkMetalHead).
Original step-by-step guide:
https://www.cryptomaton.org/2021/04/17/how-to-code-a-binance-crypto-trading-bot-that-trades-based-on-daily-news-sentiment/

Licensed under the MIT License, see [LICENSE](LICENSE).
