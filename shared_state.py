# ============================================================
# shared_state.py — Thread-safe live market data hub
# ============================================================
import threading
from datetime import datetime

_lock = threading.Lock()

_state = {
    # ── Real-time price ──────────────────────────────────────
    "live_price":           0.0,
    "bid":                  0.0,
    "ask":                  0.0,
    "spread":               0.0,
    "last_tick_time":       None,       # datetime object

    # ── Bias system ──────────────────────────────────────────
    "h4_trend":             "UNKNOWN",  # BULLISH / BEARISH / NEUTRAL
    "h1_trend":             "UNKNOWN",
    "bias_state":           "UNKNOWN",  # FULL_BULL / FULL_BEAR / CONSOLIDATING
    "allowed_side":         "NONE",     # BUY / SELL / NONE
    "bias_changed_at":      None,

    # ── Technical analysis ───────────────────────────────────
    "rsi_h1":               0.0,
    "macd_h1":              "UNKNOWN",
    "atr_h1":               0.0,
    "adx_h1":               0.0,
    "key_support":          0.0,
    "key_resistance":       0.0,
    "ema20_h1":             0.0,
    "ema50_h1":             0.0,
    "ema200_h1":            0.0,
    "daily_high":           0.0,
    "daily_low":            0.0,

    # ── ML + LLM scores ──────────────────────────────────────
    "ml_score":             0.0,        # 0.0 – 1.0
    "ml_direction":         "NONE",
    "sentiment_score":      "NEUTRAL",
    "last_signal":          None,       # dict with full signal details
    "last_signal_time":     None,

    # ── Open trade tracking ──────────────────────────────────
    "open_trade":           None,       # Only 1 trade at a time
    "daily_pnl":            0.0,
    "daily_pnl_pct":        0.0,
    "daily_paused":         False,      # True if daily SL limit hit
    "consecutive_losses":   0,

    # ── News ─────────────────────────────────────────────────
    "latest_headlines":     [],
    "next_news_event":      None,       # {"name": "CPI", "time": datetime}
    "in_news_lockout":      False,
    "sentiment_summary":    "No data",

    # ── Bot controls ─────────────────────────────────────────
    "auto_exec":            False,
    "bot_paused":           False,
    "mode":                 "live",     # "live" or "replay"
    "replay_progress":      "",         # e.g. "2024-01-15 14:30"

    # ── Thread health ────────────────────────────────────────
    "thread_status": {
        "tick_stream":      "stopped",
        "candle_watcher":   "stopped",
        "news_watcher":     "stopped",
        "telegram_bot":     "stopped",
    },
    "last_tick_received":   None,
    "mt5_connected":        False,
    "anthropic_ok":         True,
}


def get(key):
    with _lock:
        return _state.get(key)


def set(key, value):
    with _lock:
        _state[key] = value


def get_all():
    """Return a shallow copy of the full state (for Claude context injection)."""
    with _lock:
        return dict(_state)


def set_thread_status(thread_name: str, status: str):
    """Update individual thread health status."""
    with _lock:
        _state["thread_status"][thread_name] = status


def update_tick(bid: float, ask: float):
    """Fast path for tick updates — minimal locking overhead."""
    with _lock:
        _state["bid"]             = bid
        _state["ask"]             = ask
        _state["live_price"]      = (bid + ask) / 2
        _state["spread"]          = round(ask - bid, 2)
        _state["last_tick_time"]  = datetime.now()
        _state["last_tick_received"] = datetime.now()


def is_trade_open() -> bool:
    with _lock:
        return _state["open_trade"] is not None


def set_open_trade(trade: dict):
    with _lock:
        _state["open_trade"] = trade


def clear_open_trade():
    with _lock:
        _state["open_trade"] = None
