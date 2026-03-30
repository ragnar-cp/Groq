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

    Scoring system — 4 conditions, need 3/4 for a clear direction:
      1. Price vs EMA200
      2. EMA20 vs EMA50
      3. Recent market structure (last 30 candles only)
      4. Last candle close vs EMA20
    """
    if df is None or len(df) < 30:
        return "NEUTRAL"

    df = df.copy()
    df["ema20"]  = df["close"].ewm(span=config.EMA_FAST,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=config.EMA_MID,   adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=config.EMA_SLOW,  adjust=False).mean()

    last    = df.iloc[-1]
    close   = float(last["close"])
    ema20   = float(last["ema20"])
    ema50   = float(last["ema50"])
    ema200  = float(last["ema200"])

    # Condition 1: Price vs EMA200
    c1_bull = close > ema200
    c1_bear = close < ema200

    # Condition 2: EMA20 vs EMA50
    c2_bull = ema20 > ema50
    c2_bear = ema20 < ema50

    # Condition 3: Recent market structure (last 30 candles only)
    structure = _detect_structure(df, lookback=30)
    c3_bull = structure == "BULLISH"
    c3_bear = structure == "BEARISH"

    # Condition 4: Last close vs EMA20
    c4_bull = close > ema20
    c4_bear = close < ema20

    bull_score = sum([c1_bull, c2_bull, c3_bull, c4_bull])
    bear_score = sum([c1_bear, c2_bear, c3_bear, c4_bear])

    logger.debug(
        "Trend scores — BULL: %d/4 (EMA200:%s EMA2050:%s Struct:%s Close:%s) "
        "BEAR: %d/4",
        bull_score, c1_bull, c2_bull, c3_bull, c4_bull,
        bear_score
    )

    # Need 3 out of 4 conditions (was 4/4 — too strict)
    if bull_score >= 3:
        return "BULLISH"
    elif bear_score >= 3:
        return "BEARISH"
    else:
        return "NEUTRAL"


def _detect_structure(df: pd.DataFrame, lookback: int = 30) -> str:
    """
    Detect Higher Highs/Higher Lows (bullish) or
    Lower Highs/Lower Lows (bearish) using recent swing points only.
    Uses last `lookback` candles for responsiveness.
    """
    # Only look at recent candles
    recent = df.tail(lookback)
    highs  = recent["high"].values
    lows   = recent["low"].values

    swing_highs = _find_swings(highs, mode="high", window=2)
    swing_lows  = _find_swings(lows,  mode="low",  window=2)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        # Not enough swings — fall back to simple slope check
        return _slope_structure(df.tail(lookback))

    hh = swing_highs[-1] > swing_highs[-2]   # Higher High
    hl = swing_lows[-1]  > swing_lows[-2]    # Higher Low
    lh = swing_highs[-1] < swing_highs[-2]   # Lower High
    ll = swing_lows[-1]  < swing_lows[-2]    # Lower Low

    if hh and hl:
        return "BULLISH"
    elif lh and ll:
        return "BEARISH"
    return "NEUTRAL"


def _slope_structure(df: pd.DataFrame) -> str:
    """
    Fallback: use simple linear regression slope of closes.
    Positive slope = bullish structure, negative = bearish.
    """
    closes = df["close"].values
    n      = len(closes)
    if n < 5:
        return "NEUTRAL"

    x     = np.arange(n)
    slope = np.polyfit(x, closes, 1)[0]

    # Normalise slope as % of price
    slope_pct = slope / closes.mean() * 100

    if slope_pct > 0.01:    # Rising at least 0.01% per candle
        return "BULLISH"
    elif slope_pct < -0.01:
        return "BEARISH"
    return "NEUTRAL"


def _find_swings(values: np.ndarray, mode: str, window: int = 2) -> list:
    """Return values at local peaks (mode=high) or troughs (mode=low).
    Window=2 means look 2 candles either side — more responsive than 3."""
    swings = []
    for i in range(window, len(values) - window):
        segment = values[i - window : i + window + 1]
        if mode == "high" and values[i] == max(segment):
            swings.append(values[i])
        elif mode == "low" and values[i] == min(segment):
            swings.append(values[i])
    return swings
