# ============================================================
# data/mt5_feed.py — MT5 connection + candle fetcher
# ============================================================
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

import shared_state
import config

logger = logging.getLogger(__name__)

# MetaTrader5 is Windows-only. Guard import so the rest of the
# codebase (including replay mode) can import this module on any OS.
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 library not found. Running in replay/stub mode.")

# Map human-readable TF strings → MT5 constants
_TF_MAP = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "M30": 30,
    "H1":  16385,
    "H4":  16388,
    "D1":  16408,
}


def connect() -> bool:
    """Initialise and log in to MT5. Returns True on success."""
    if not MT5_AVAILABLE:
        logger.warning("MT5 not available — skipping connection.")
        shared_state.set("mt5_connected", False)
        return False

    if not mt5.initialize():
        logger.error("MT5 initialize() failed: %s", mt5.last_error())
        shared_state.set("mt5_connected", False)
        return False

    authorised = mt5.login(
        config.MT5_LOGIN,
        password=config.MT5_PASSWORD,
        server=config.MT5_SERVER,
    )
    if not authorised:
        logger.error("MT5 login failed: %s", mt5.last_error())
        shared_state.set("mt5_connected", False)
        return False

    logger.info("MT5 connected — account #%s", config.MT5_LOGIN)
    shared_state.set("mt5_connected", True)
    return True


def disconnect():
    if MT5_AVAILABLE and mt5:
        mt5.shutdown()
    shared_state.set("mt5_connected", False)
    logger.info("MT5 disconnected.")


def get_candles(timeframe: str, count: int = 200) -> Optional[pd.DataFrame]:
    """
    Fetch the last `count` closed candles for SYMBOL on `timeframe`.
    Returns a DataFrame with columns: time, open, high, low, close, volume.
    Returns None if MT5 is unavailable or an error occurs.
    """
    if not MT5_AVAILABLE or not shared_state.get("mt5_connected"):
        return None

    tf_const = _TF_MAP.get(timeframe)
    if tf_const is None:
        logger.error("Unknown timeframe: %s", timeframe)
        return None

    rates = mt5.copy_rates_from_pos(config.SYMBOL, tf_const, 0, count + 1)
    if rates is None or len(rates) == 0:
        logger.warning("No candles returned for %s %s", config.SYMBOL, timeframe)
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]]

    # Drop the last (still-forming) candle — only use closed candles
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def get_candles_range(
    timeframe: str,
    start: datetime,
    end: datetime,
) -> Optional[pd.DataFrame]:
    """
    Fetch candles between start and end dates.
    Used by historical_fetcher.py for ML training data.
    """
    if not MT5_AVAILABLE or not shared_state.get("mt5_connected"):
        return None

    tf_const = _TF_MAP.get(timeframe)
    if tf_const is None:
        return None

    # Ensure symbol is selected/enabled before fetching
    mt5.symbol_select(config.SYMBOL, True)

    # MT5 requires timezone-aware UTC datetimes — use stdlib timezone (no pytz needed)
    from datetime import timezone as _tz
    if start.tzinfo is None:
        start = start.replace(tzinfo=_tz.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=_tz.utc)

    rates = mt5.copy_rates_range(config.SYMBOL, tf_const, start, end)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        logger.warning("copy_rates_range returned None for %s %s. MT5 error: %s",
                       config.SYMBOL, timeframe, err)
        # Fallback: fetch by count instead of date range
        logger.info("Trying fallback: copy_rates_from_pos for %s %s", config.SYMBOL, timeframe)
        candles_needed = {
            "M15": 365 * 3 * 24 * 4,   # 3 years of M15
            "H1":  365 * 3 * 24,
            "H4":  365 * 3 * 6,
        }.get(timeframe, 10000)
        rates = mt5.copy_rates_from_pos(config.SYMBOL, tf_const, 0, min(candles_needed, 99000))
        if rates is None or len(rates) == 0:
            logger.error("Fallback also failed. MT5 error: %s", mt5.last_error())
            return None
        logger.info("Fallback succeeded — got %d candles", len(rates))

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]]
    return df


def get_account_info() -> Optional[dict]:
    """Return basic account info dict or None."""
    if not MT5_AVAILABLE or not shared_state.get("mt5_connected"):
        return None
    info = mt5.account_info()
    if info is None:
        return None
    return {
        "balance":  info.balance,
        "equity":   info.equity,
        "margin":   info.margin,
        "currency": info.currency,
    }
