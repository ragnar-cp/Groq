# ============================================================
# analysis/indicators.py — RSI, MACD, ATR, ADX, BB, S&R
# ============================================================
import logging
from typing import Optional, Dict

import pandas as pd
import numpy as np

import config
import shared_state

logger = logging.getLogger(__name__)


def calculate_indicators(df: pd.DataFrame) -> Optional[Dict]:
    """
    Calculate all indicators for a given candle DataFrame.
    Returns a dict of key→value pairs that get written to shared_state.
    """
    if df is None or len(df) < 50:
        return None

    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    result = {}

    # ── EMA ──────────────────────────────────────────────────
    result["ema20_h1"]  = float(close.ewm(span=config.EMA_FAST, adjust=False).mean().iloc[-1])
    result["ema50_h1"]  = float(close.ewm(span=config.EMA_MID,  adjust=False).mean().iloc[-1])
    result["ema200_h1"] = float(close.ewm(span=config.EMA_SLOW, adjust=False).mean().iloc[-1])

    # ── RSI ──────────────────────────────────────────────────
    result["rsi_h1"] = float(_rsi(close, 14))

    # ── MACD ─────────────────────────────────────────────────
    macd_line, signal_line, histogram = _macd(close)
    result["macd_histogram"] = float(histogram)
    if histogram > 0 and histogram > _prev(histogram, df, close):
        result["macd_h1"] = "BULLISH"
    elif histogram < 0 and histogram < _prev(histogram, df, close):
        result["macd_h1"] = "BEARISH"
    else:
        result["macd_h1"] = "NEUTRAL"

    # ── ATR ──────────────────────────────────────────────────
    atr_series        = _atr(high, low, close, 14)
    result["atr_h1"]  = float(atr_series.iloc[-1])
    atr_avg           = float(atr_series.rolling(50).mean().iloc[-1])
    result["atr_ratio"] = result["atr_h1"] / atr_avg if atr_avg > 0 else 1.0

    # ── ADX ──────────────────────────────────────────────────
    result["adx_h1"] = float(_adx(high, low, close, 14))

    # ── Bollinger Bands ──────────────────────────────────────
    bb_upper, bb_mid, bb_lower = _bollinger(close, 20, 2)
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        result["bb_position"] = float((close.iloc[-1] - bb_lower) / bb_range)
    else:
        result["bb_position"] = 0.5

    # ── Support & Resistance ─────────────────────────────────
    support, resistance = _support_resistance(high, low, close)
    result["key_support"]    = float(support)
    result["key_resistance"] = float(resistance)

    # ── Daily High / Low ─────────────────────────────────────
    result["daily_high"] = float(high.tail(96).max())   # ~1 day of M15
    result["daily_low"]  = float(low.tail(96).min())

    # ── ML features (also stored for predict.py) ─────────────
    last_close = float(close.iloc[-1])
    result["dist_ema50_pct"]  = (last_close - result["ema50_h1"])  / result["ema50_h1"]  * 100
    result["dist_ema200_pct"] = (last_close - result["ema200_h1"]) / result["ema200_h1"] * 100
    result["prev_candle_range"] = float(high.iloc[-2] - low.iloc[-2])

    return result


# ── Private helpers ──────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    plus_dm  = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    atr_s    = _atr(high, low, close, period)
    plus_di  = 100 * plus_dm.ewm(span=period,  adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s

    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / dx_denom
    adx = dx.ewm(span=period, adjust=False).mean()
    return float(adx.iloc[-1])


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2):
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])


def _support_resistance(
    high: pd.Series,
    low:  pd.Series,
    close: pd.Series,
    lookback: int = 50,
) -> tuple:
    """Simple pivot-based S&R: nearest swing high/low to current price."""
    recent_high = high.tail(lookback)
    recent_low  = low.tail(lookback)
    current     = float(close.iloc[-1])

    swing_highs = [float(recent_high.iloc[i])
                   for i in range(1, len(recent_high) - 1)
                   if recent_high.iloc[i] > recent_high.iloc[i-1]
                   and recent_high.iloc[i] > recent_high.iloc[i+1]]

    swing_lows  = [float(recent_low.iloc[i])
                   for i in range(1, len(recent_low) - 1)
                   if recent_low.iloc[i] < recent_low.iloc[i-1]
                   and recent_low.iloc[i] < recent_low.iloc[i+1]]

    resistances = [h for h in swing_highs if h > current]
    supports    = [l for l in swing_lows  if l < current]

    resistance = min(resistances) if resistances else float(high.max())
    support    = max(supports)    if supports    else float(low.min())

    return support, resistance


def _prev(value: float, df: pd.DataFrame, close: pd.Series) -> float:
    """Helper — returns previous MACD histogram value (simplified)."""
    _, _, hist = _macd(close.iloc[:-1])
    return float(hist)
