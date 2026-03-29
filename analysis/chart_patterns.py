# ============================================================
# analysis/chart_patterns.py — Chart pattern detector
# ============================================================
import logging
from typing import Optional, Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TOLERANCE = 0.003   # 0.3% price tolerance for level matching


def detect_chart_patterns(df: pd.DataFrame) -> Optional[Dict]:
    """
    Detect high-timeframe chart patterns on recent candle data.
    Returns dict: { pattern, direction, strength } or None.
    """
    if df is None or len(df) < 30:
        return None

    checks = [
        _double_top(df),
        _double_bottom(df),
        _head_and_shoulders(df),
        _inv_head_and_shoulders(df),
        _ascending_triangle(df),
        _descending_triangle(df),
    ]

    valid = [p for p in checks if p is not None]
    if not valid:
        return None
    return max(valid, key=lambda p: p["strength"])


def _swings(series: pd.Series, mode: str, window: int = 5) -> list:
    """Return (index, value) tuples for local peaks or troughs."""
    results = []
    arr = series.values
    for i in range(window, len(arr) - window):
        seg = arr[i - window: i + window + 1]
        if mode == "high" and arr[i] == max(seg):
            results.append((i, arr[i]))
        elif mode == "low" and arr[i] == min(seg):
            results.append((i, arr[i]))
    return results


def _near(a: float, b: float) -> bool:
    return abs(a - b) / max(abs(b), 1e-9) < TOLERANCE


def _double_top(df: pd.DataFrame) -> Optional[Dict]:
    highs  = _swings(df["high"], "high")
    lows   = _swings(df["low"],  "low")
    if len(highs) < 2 or len(lows) < 1:
        return None
    h1, h2 = highs[-2], highs[-1]
    if _near(h1[1], h2[1]) and h1[0] < h2[0]:
        return {"pattern": "Double Top", "direction": "SELL", "strength": 80}
    return None


def _double_bottom(df: pd.DataFrame) -> Optional[Dict]:
    lows = _swings(df["low"], "low")
    if len(lows) < 2:
        return None
    l1, l2 = lows[-2], lows[-1]
    if _near(l1[1], l2[1]) and l1[0] < l2[0]:
        return {"pattern": "Double Bottom", "direction": "BUY", "strength": 80}
    return None


def _head_and_shoulders(df: pd.DataFrame) -> Optional[Dict]:
    highs = _swings(df["high"], "high")
    if len(highs) < 3:
        return None
    l, h, r = highs[-3], highs[-2], highs[-1]
    if h[1] > l[1] and h[1] > r[1] and _near(l[1], r[1]):
        return {"pattern": "Head & Shoulders", "direction": "SELL", "strength": 85}
    return None


def _inv_head_and_shoulders(df: pd.DataFrame) -> Optional[Dict]:
    lows = _swings(df["low"], "low")
    if len(lows) < 3:
        return None
    l, h, r = lows[-3], lows[-2], lows[-1]
    if h[1] < l[1] and h[1] < r[1] and _near(l[1], r[1]):
        return {"pattern": "Inv Head & Shoulders", "direction": "BUY", "strength": 85}
    return None


def _ascending_triangle(df: pd.DataFrame) -> Optional[Dict]:
    highs = _swings(df["high"], "high")
    lows  = _swings(df["low"],  "low")
    if len(highs) < 3 or len(lows) < 3:
        return None
    # Flat top + rising lows
    flat_top    = _near(highs[-1][1], highs[-2][1]) and _near(highs[-2][1], highs[-3][1])
    rising_lows = lows[-1][1] > lows[-2][1] > lows[-3][1]
    if flat_top and rising_lows:
        return {"pattern": "Ascending Triangle", "direction": "BUY", "strength": 75}
    return None


def _descending_triangle(df: pd.DataFrame) -> Optional[Dict]:
    highs = _swings(df["high"], "high")
    lows  = _swings(df["low"],  "low")
    if len(highs) < 3 or len(lows) < 3:
        return None
    # Flat bottom + falling highs
    flat_bot     = _near(lows[-1][1], lows[-2][1]) and _near(lows[-2][1], lows[-3][1])
    falling_highs= highs[-1][1] < highs[-2][1] < highs[-3][1]
    if flat_bot and falling_highs:
        return {"pattern": "Descending Triangle", "direction": "SELL", "strength": 75}
    return None
