# ============================================================
# analysis/trend.py — EMA trend + market structure detection
# ============================================================
import logging
import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


def calculate_trend(df: pd.DataFrame) -> str:
    """
    Determine trend for a given candle DataFrame.
    Returns: "BULLISH" | "BEARISH" | "NEUTRAL"

    Rules (ALL must be true for a clear direction):
      BULLISH: close > EMA200, EMA20 > EMA50, Higher Highs + Higher Lows,
               last close above EMA20
      BEARISH: close < EMA200, EMA20 < EMA50, Lower Highs + Lower Lows,
               last close below EMA20
    """
    if df is None or len(df) < config.EMA_SLOW + 5:
        return "NEUTRAL"

    df = df.copy()
    df["ema20"]  = df["close"].ewm(span=config.EMA_FAST,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=config.EMA_MID,   adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=config.EMA_SLOW,  adjust=False).mean()

    last = df.iloc[-1]
    close   = last["close"]
    ema20   = last["ema20"]
    ema50   = last["ema50"]
    ema200  = last["ema200"]

    # Market structure
    structure = _detect_structure(df)

    bullish_conditions = [
        close > ema200,
        ema20 > ema50,
        structure == "BULLISH",
        close > ema20,
    ]
    bearish_conditions = [
        close < ema200,
        ema20 < ema50,
        structure == "BEARISH",
        close < ema20,
    ]

    bull_score = sum(bullish_conditions)
    bear_score = sum(bearish_conditions)

    if bull_score == 4:
        return "BULLISH"
    elif bear_score == 4:
        return "BEARISH"
    elif bull_score >= 3:
        return "BULLISH"   # Strong lean
    elif bear_score >= 3:
        return "BEARISH"
    else:
        return "NEUTRAL"


def _detect_structure(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Detect Higher Highs/Higher Lows (bullish) or
    Lower Highs/Lower Lows (bearish) using recent swing points.
    """
    recent = df.tail(lookback)
    highs  = recent["high"].values
    lows   = recent["low"].values

    # Find local swing highs and lows (simple pivot detection)
    swing_highs = _find_swings(highs, mode="high")
    swing_lows  = _find_swings(lows,  mode="low")

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "NEUTRAL"

    # Check last two swings
    hh = swing_highs[-1] > swing_highs[-2]   # Higher High
    hl = swing_lows[-1]  > swing_lows[-2]    # Higher Low
    lh = swing_highs[-1] < swing_highs[-2]   # Lower High
    ll = swing_lows[-1]  < swing_lows[-2]    # Lower Low

    if hh and hl:
        return "BULLISH"
    elif lh and ll:
        return "BEARISH"
    return "NEUTRAL"


def _find_swings(values: np.ndarray, mode: str, window: int = 3) -> list:
    """Return values at local peaks (mode=high) or troughs (mode=low)."""
    swings = []
    for i in range(window, len(values) - window):
        segment = values[i - window : i + window + 1]
        if mode == "high" and values[i] == max(segment):
            swings.append(values[i])
        elif mode == "low" and values[i] == min(segment):
            swings.append(values[i])
    return swings
