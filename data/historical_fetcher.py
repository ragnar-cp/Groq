# ============================================================
# data/historical_fetcher.py — MT5 history fetch + auto-labeling
# ============================================================
import logging
import pytz
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


def fetch_for_training() -> Optional[pd.DataFrame]:
    """
    Fetch N years of M15 XAUUSD history from MT5 and auto-label each candle.
    Labels:
        1  = BUY  (price rose > ML_LABEL_PIPS in next ML_LABEL_LOOKFORWARD candles)
        0  = SELL (price fell > ML_LABEL_PIPS)
        -1 = NEUTRAL (skip during training)
    Returns a labelled DataFrame ready for ml/train.py.
    """
    # ── Connect MT5 if not already connected ──────────────────
    from data.mt5_feed import connect, get_candles_range
    import shared_state
    if not shared_state.get("mt5_connected"):
        logger.info("MT5 not connected — connecting now for training…")
        if not connect():
            logger.error(
                "MT5 connection failed.\n"
                "Make sure:\n"
                "  1. MetaTrader 5 desktop app is OPEN and logged in\n"
                "  2. MT5_LOGIN, MT5_PASSWORD, MT5_SERVER are correct in config.py\n"
                "  3. Your broker server name matches exactly"
            )
            return None
        logger.info("MT5 connected successfully.")

    # ── Diagnose symbol availability ─────────────────────────
    try:
        import MetaTrader5 as mt5
        # Find any gold-related symbol available on this broker
        all_symbols = mt5.symbols_get()
        gold_symbols = [s.name for s in all_symbols if "XAU" in s.name.upper() or "GOLD" in s.name.upper()]
        logger.info("Gold-related symbols on your broker: %s", gold_symbols)

        if not gold_symbols:
            logger.error("No gold symbols found on your broker account.")
            return None

        # Use the first matching symbol if config symbol not available
        available_names = [s.name for s in all_symbols]
        if config.SYMBOL not in available_names:
            best = gold_symbols[0]
            logger.warning(
                "Symbol '%s' not found. Using '%s' instead. "
                "Update SYMBOL in config.py to fix permanently.",
                config.SYMBOL, best
            )
            config.SYMBOL = best
        else:
            logger.info("Symbol '%s' confirmed available.", config.SYMBOL)

        # Enable the symbol for data access
        mt5.symbol_select(config.SYMBOL, True)

    except Exception as e:
        logger.warning("Symbol check failed: %s", e)

    utc      = pytz.utc
    end_dt   = datetime.now(utc)
    start_dt = end_dt - timedelta(days=365 * config.ML_HISTORY_YEARS)

    logger.info("Fetching %d years of M15 history (%s → %s)…",
                config.ML_HISTORY_YEARS, start_dt.date(), end_dt.date())

    df = get_candles_range("M15", start_dt, end_dt)
    if df is None or df.empty:
        logger.error("No historical data returned from MT5.")
        return None

    logger.info("Fetched %d M15 candles. Labelling…", len(df))
    df = _add_labels(df)

    # Drop neutral rows
    labelled = df[df["label"] != -1].copy()
    logger.info("Labelled dataset: %d rows (%d BUY, %d SELL)",
                len(labelled),
                (labelled["label"] == 1).sum(),
                (labelled["label"] == 0).sum())
    return labelled


def fetch_range_for_replay(
    timeframe: str,
    start: datetime,
    end: datetime,
) -> Optional[pd.DataFrame]:
    """Used by candle_watcher replay mode — raw candles, no labelling."""
    from data.mt5_feed import get_candles_range
    utc = pytz.utc
    if start.tzinfo is None:
        start = utc.localize(start)
    if end.tzinfo is None:
        end = utc.localize(end)
    return get_candles_range(timeframe, start, end)


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each candle at index i, look forward ML_LABEL_LOOKFORWARD candles.
    If max future high - close > threshold → BUY (1)
    If close - min future low  > threshold → SELL (0)
    Otherwise → NEUTRAL (-1)
    """
    n     = config.ML_LABEL_LOOKFORWARD
    pips  = config.ML_LABEL_PIPS * 0.1  # Convert pips to price units (XAUUSD)

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    labels = np.full(len(df), -1, dtype=int)

    for i in range(len(df) - n):
        close   = closes[i]
        future_highs = highs[i + 1 : i + 1 + n]
        future_lows  = lows[i + 1 : i + 1 + n]

        up_move   = np.max(future_highs) - close
        down_move = close - np.min(future_lows)

        if up_move > pips and up_move > down_move:
            labels[i] = 1
        elif down_move > pips and down_move > up_move:
            labels[i] = 0
        # else remains -1 (neutral)

    df = df.copy()
    df["label"] = labels
    return df
