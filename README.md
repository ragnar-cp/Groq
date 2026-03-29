# XAUUSD AI Trading Bot

A professional-grade Gold trading bot powered by a 3-layer hybrid AI engine (Rule-based + XGBoost ML + Claude LLM), real-time MT5 data, and a full Telegram interface.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+ on **Windows** (MT5 library is Windows-only)
- MetaTrader 5 installed with a broker account (demo is fine)
- Anthropic API key → https://console.anthropic.com
- NewsAPI key → https://newsapi.org (free tier)
- Telegram bot token → message @BotFather on Telegram

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure
Edit `config.py` and fill in:
```python
MT5_LOGIN    = 123456789
MT5_PASSWORD = "your_password"
MT5_SERVER   = "YourBroker-Demo"

ANTHROPIC_API_KEY     = "sk-ant-..."
NEWS_API_KEY          = "your_newsapi_key"
TELEGRAM_BOT_TOKEN    = "123456:ABC..."
TELEGRAM_ALLOWED_UIDS = [your_telegram_id]
```

> To find your Telegram user ID: message @userinfobot on Telegram.

### 4. Train the ML model
```bash
python main.py --train
```
This fetches 3 years of XAUUSD M15 data from MT5, labels it, and trains an XGBoost model. Takes ~5 minutes.

### 5. Run in replay mode (recommended first)
```bash
python main.py --replay
```
Feeds historical candles through the full pipeline — no real money at risk. Watch your Telegram for signals.

### 6. Run live
```bash
python main.py --live
```

---

## Modes

| Mode | Command | Description |
|---|---|---|
| Replay | `python main.py --replay` | Historical candles fed as live — full debug |
| Live | `python main.py --live` | Real MT5 market data |
| Train | `python main.py --train` | Train ML model only, then exit |

Mode can also be set in `config.py`:
```python
MODE = "replay"  # or "live"
```

---

## Telegram Commands

| Command | Description |
|---|---|
| `/status` | Price, bias state, ML score, open trade |
| `/analysis` | Full TA breakdown (RSI, MACD, ATR, ADX, S&R) |
| `/news` | Latest headlines + sentiment |
| `/trades` | Today's trade results |
| `/memory` | Win rate stats by setup & session |
| `/autoon` | Enable auto-execution (live mode only) |
| `/autooff` | Disable auto-execution |
| `/pause` | Pause bot scanning |
| `/resume` | Resume bot scanning |
| `/risk` | View current risk settings & modifiers |
| `/health` | Thread status + MT5/API connection check |
| `/help` | Show all commands |

You can also just **talk to the bot in plain English**:
- "What's gold doing right now?"
- "Why no signals in the last hour?"
- "What's your plan for today?"
- "What does the news say?"

---

## Architecture

```
4 Threads running simultaneously:

Thread 1: Tick Stream       — millisecond price capture (live) / passive (replay)
Thread 2: Candle Watcher    — fires analysis on M15/H1/H4 close + replay engine
Thread 3: News Watcher      — polls news + ForexFactory every 3 min
Thread 4: Telegram Bot      — listens for commands and free-text chat
```

### Signal Flow (5 Gates)
```
M15 Candle Closes
├── Gate 1: Bot paused or daily loss limit hit? → Exit
├── Gate 2: Trade already open? → Exit (one trade at a time)
├── Gate 3: Bias CONSOLIDATING? → Exit
├── Gate 4: News lockout? → Exit
└── Gate 5: Run full 3-layer analysis
      ├── Rule-based TA score (30%)
      ├── ML model score (35%)
      └── LLM (Claude) score (35%)
      → Fuse → Confidence score
      → >= 60%: Telegram alert
      → >= 75% + /autoon: Auto-execute
```

### Bias Cascade (H4 → H1 → M15)
| H4 | H1 | State | Action |
|---|---|---|---|
| Bullish | Bullish | 🟢 FULL BULL | BUY signals only |
| Bearish | Bearish | 🔴 FULL BEAR | SELL signals only |
| Mismatch | Mismatch | 🟡 CONSOLIDATING | No trades |

---

## File Structure

```
xauusd_bot/
├── main.py                    # Entry point
├── config.py                  # All settings
├── shared_state.py            # Thread-safe live data hub
├── requirements.txt
│
├── data/
│   ├── mt5_feed.py            # MT5 candle + account data
│   ├── tick_stream.py         # Real-time tick monitoring
│   ├── candle_watcher.py      # Candle close detector + Replay engine
│   ├── historical_fetcher.py  # MT5 history + auto-labeling for ML
│   ├── news_feed.py           # NewsAPI + RSS headlines
│   └── economic_calendar.py  # ForexFactory scraper
│
├── analysis/
│   ├── trend.py               # EMA + market structure
│   ├── candle_patterns.py     # Engulfing, hammer, pin bar…
│   ├── chart_patterns.py      # H&S, double top/bottom, triangles
│   ├── indicators.py          # RSI, MACD, ATR, ADX, BB, S&R
│   └── sentiment.py           # News → Claude → sentiment score
│
├── ml/
│   ├── train.py               # XGBoost trainer (full feature set)
│   ├── predict.py             # Live ML inference
│   └── model.pkl              # Saved model (created after training)
│
├── engine/
│   ├── signal_fusion.py       # 3-layer combiner → final signal
│   ├── risk_manager.py        # Dynamic position sizing
│   └── trade_memory.py        # Persistent trade learning
│
├── execution/
│   ├── mt5_executor.py        # Trade placement + partial TP + trail
│   └── alert.py               # All Telegram message formatters
│
├── telegram_bot/
│   ├── bot.py                 # Main Telegram listener
│   ├── commands.py            # Slash command handlers
│   └── chat_handler.py        # Free text → Claude → reply
│
└── logs/
    ├── trades.log             # All bot activity
    └── trade_memory.json      # Persistent trade history
```

---

## Risk Warnings

- This bot is for **educational purposes**. Past performance does not guarantee future results.
- Always test in **replay mode** and then **demo mode** before using real money.
- Never risk money you cannot afford to lose.
- The AI components (Claude LLM) require API calls which have associated costs.
- MT5 auto-execution is **disabled by default** — you must explicitly send `/autoon`.
