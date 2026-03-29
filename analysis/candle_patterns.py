# ============================================================
# analysis/candle_patterns.py — Candle pattern detector
# ============================================================
import logging
from typing import Optional, Dict
import pandas as pd

logger = logging.getLogger(__name__)

DOJI_BODY_RATIO    = 0.1    # Body/range < 10% → Doji
PIN_BAR_TAIL_RATIO = 0.6    # Tail/range > 60% → Pin Bar
ENGULF_BODY_MULT   = 1.2    # Engulfing body must be 1.2× previous


def detect_candle_patterns(df: pd.DataFrame) -> Optional[Dict]:
    """
    Detect the most significant candle pattern on the last 2 candles.
    Returns dict with keys: pattern (str), direction (BUY/SELL/NEUTRAL), strength (0-100)
    Returns None if no significant pattern found.
    """
    if df is None or len(df) < 3:
        return None

    c1 = df.iloc[-1]   # Most recent closed candle
    c2 = df.iloc[-2]   # Previous candle

    checks = [
        _bullish_engulfing(c1, c2),
        _bearish_engulfing(c1, c2),
        _hammer(c1),
        _shooting_star(c1),
        _pin_bar_bull(c1),
        _pin_bar_bear(c1),
        _doji(c1),
        _morning_star(df),
        _evening_star(df),
    ]

    # Return highest-strength pattern found
    valid = [p for p in checks if p is not None]
    if not valid:
        return None
    return max(valid, key=lambda p: p["strength"])


def _body(c) -> float:
    return abs(c["close"] - c["open"])

def _range(c) -> float:
    return c["high"] - c["low"]

def _upper_wick(c) -> float:
    return c["high"] - max(c["open"], c["close"])

def _lower_wick(c) -> float:
    return min(c["open"], c["close"]) - c["low"]

def _bullish(c) -> bool:
    return c["close"] > c["open"]

def _bearish(c) -> bool:
    return c["close"] < c["open"]


def _bullish_engulfing(c1, c2) -> Optional[Dict]:
    if not (_bullish(c1) and _bearish(c2)):
        return None
    if _body(c1) >= _body(c2) * ENGULF_BODY_MULT:
        if c1["open"] <= c2["close"] and c1["close"] >= c2["open"]:
            return {"pattern": "Bullish Engulfing", "direction": "BUY", "strength": 85}
    return None


def _bearish_engulfing(c1, c2) -> Optional[Dict]:
    if not (_bearish(c1) and _bullish(c2)):
        return None
    if _body(c1) >= _body(c2) * ENGULF_BODY_MULT:
        if c1["open"] >= c2["close"] and c1["close"] <= c2["open"]:
            return {"pattern": "Bearish Engulfing", "direction": "SELL", "strength": 85}
    return None


def _hammer(c) -> Optional[Dict]:
    r = _range(c)
    if r == 0:
        return None
    body = _body(c)
    lower = _lower_wick(c)
    upper = _upper_wick(c)
    if body / r < 0.35 and lower / r > 0.55 and upper / r < 0.1:
        return {"pattern": "Hammer", "direction": "BUY", "strength": 75}
    return None


def _shooting_star(c) -> Optional[Dict]:
    r = _range(c)
    if r == 0:
        return None
    body  = _body(c)
    lower = _lower_wick(c)
    upper = _upper_wick(c)
    if body / r < 0.35 and upper / r > 0.55 and lower / r < 0.1:
        return {"pattern": "Shooting Star", "direction": "SELL", "strength": 75}
    return None


def _pin_bar_bull(c) -> Optional[Dict]:
    r = _range(c)
    if r == 0:
        return None
    if _lower_wick(c) / r >= PIN_BAR_TAIL_RATIO and _bullish(c):
        return {"pattern": "Bullish Pin Bar", "direction": "BUY", "strength": 80}
    return None


def _pin_bar_bear(c) -> Optional[Dict]:
    r = _range(c)
    if r == 0:
        return None
    if _upper_wick(c) / r >= PIN_BAR_TAIL_RATIO and _bearish(c):
        return {"pattern": "Bearish Pin Bar", "direction": "SELL", "strength": 80}
    return None


def _doji(c) -> Optional[Dict]:
    r = _range(c)
    if r == 0:
        return None
    if _body(c) / r < DOJI_BODY_RATIO:
        return {"pattern": "Doji", "direction": "NEUTRAL", "strength": 50}
    return None


def _morning_star(df) -> Optional[Dict]:
    if len(df) < 3:
        return None
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if _bearish(c1) and _body(c2) < _body(c1) * 0.3 and _bullish(c3):
        if c3["close"] > (c1["open"] + c1["close"]) / 2:
            return {"pattern": "Morning Star", "direction": "BUY", "strength": 90}
    return None


def _evening_star(df) -> Optional[Dict]:
    if len(df) < 3:
        return None
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if _bullish(c1) and _body(c2) < _body(c1) * 0.3 and _bearish(c3):
        if c3["close"] < (c1["open"] + c1["close"]) / 2:
            return {"pattern": "Evening Star", "direction": "SELL", "strength": 90}
    return None
