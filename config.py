# ============================================================
# config.py — Central configuration for XAUUSD Trading Bot
# ============================================================
# Fill in your API keys and adjust settings before running.

# ── Mode ─────────────────────────────────────────────────────
# "live"   = real market data from MT5
# "replay" = feed historical candles as if live (for testing)
MODE = "replay"

REPLAY_SPEED_SECONDS = 1      # Delay between each replayed candle (lower = faster)
REPLAY_START_DATE    = "2026-01-01"   # Start date for replay
REPLAY_END_DATE      = "2026-03-01"   # End date for replay

# ── MT5 Credentials ──────────────────────────────────────────
MT5_LOGIN    = 24468847              # Your MT5 account number
MT5_PASSWORD = "&x$c1%G4"
MT5_SERVER   = "VantageInternational-Demo"

# ── API Keys ─────────────────────────────────────────────────
GROQ_API_KEY  = "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"    # https://console.groq.com/keys
GROQ_MODEL    = "llama-3.3-70b-versatile"  # fast + free. alternatives: "mixtral-8x7b-32768", "gemma2-9b-it"
NEWS_API_KEY      = "3ab899b86f6f4509bd0e9db2107f1c98"

# ── Telegram ─────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN    = "8777450125:AAE2mccZn4IO8d3ePYIkGliPRVkm6nLsxdM"
TELEGRAM_ALLOWED_UIDS = [7858967749]   # Your Telegram user ID (whitelist)

# ── Symbol & Timeframes ──────────────────────────────────────
SYMBOL      = "XAUUSD"
TF_ENTRY    = "M15"
TF_BIAS_1   = "H1"
TF_BIAS_2   = "H4"

# ── Risk Management ──────────────────────────────────────────
BASE_RISK_PERCENT   = 1.5     # % of account risked per trade
MAX_DAILY_LOSS_PCT  = 3.0     # % daily drawdown limit — pause for the day if hit
ATR_SL_MULTIPLIER   = 1.5     # SL = ATR × this multiplier
ATR_TRAIL_MULTIPLIER= 1.5     # Trailing stop = ATR × this multiplier
TP1_RR              = 1.5     # TP1 at 1.5× risk
TP2_RR              = 3.0     # TP2 at 3.0× risk
TP1_CLOSE_PCT       = 0.5     # Close 50% at TP1

# Dynamic sizing confidence modifiers
CONFIDENCE_HIGH_THRESHOLD  = 80   # > 80% → scale up
CONFIDENCE_LOW_THRESHOLD   = 70   # < 70% → scale down
SIZE_MODIFIER_HIGH         = 1.3
SIZE_MODIFIER_NORMAL       = 1.0
SIZE_MODIFIER_LOW          = 0.7
SIZE_MODIFIER_LOSS_STREAK  = 0.6  # After 2 consecutive losses
SIZE_MODIFIER_HIGH_VOLATILITY = 0.8

# ── Signal Thresholds ────────────────────────────────────────
ALERT_CONFIDENCE_MIN   = 60   # Minimum confidence % to send alert
AUTOEXEC_CONFIDENCE_MIN= 75   # Minimum confidence % for auto-execution

# Signal fusion weights
WEIGHT_RULE_BASED = 0.30
WEIGHT_ML         = 0.35
WEIGHT_LLM        = 0.35

# ── News Lockout ─────────────────────────────────────────────
NEWS_LOCKOUT_MINUTES_BEFORE = 30
NEWS_LOCKOUT_MINUTES_AFTER  = 15

# ── Trend Detection (EMA periods) ────────────────────────────
EMA_FAST   = 20
EMA_MID    = 50
EMA_SLOW   = 200

# ── ML Model ─────────────────────────────────────────────────
ML_LABEL_PIPS        = 15     # Min pip move to label as BUY/SELL
ML_LABEL_LOOKFORWARD = 16     # M15 candles to look forward (~4H)
ML_HISTORY_YEARS     = 1      # Years of history to fetch for training
ML_RETRAIN_DAYS      = 7      # Retrain every N days

# ── Logging ──────────────────────────────────────────────────
LOG_FILE          = "logs/trades.log"
TRADE_MEMORY_FILE = "logs/trade_memory.json"

# ── Health Check ─────────────────────────────────────────────
HEALTH_TICK_STALE_SECONDS = 30   # Alert if no tick received for this long
