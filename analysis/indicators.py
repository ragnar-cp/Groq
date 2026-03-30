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
    Returns a dict of key→value pairs AND writes them to shared_state.

    BUG FIX: previously this function returned a dict but never wrote
    it to shared_state, so predict() was always reading zeros.
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
    result["daily_high"] = float(high.tail(96).max())
    result["daily_low"]  = float(low.tail(96).min())

    # ── ML features ──────────────────────────────────────────
    last_close = float(close.iloc[-1])
    e50  = result["ema50_h1"]
    e200 = result["ema200_h1"]
    result["dist_ema50_pct"]    = (last_close - e50)  / e50  * 100 if e50  != 0 else 0.0
    result["dist_ema200_pct"]   = (last_close - e200) / e200 * 100 if e200 != 0 else 0.0
    result["prev_candle_range"] = float(high.iloc[-2] - low.iloc[-2])

    # ── BUG FIX: write every indicator into shared_state ─────
    # Without this, shared_state retains its initial 0.0 values and
    # predict() receives a zero vector regardless of market conditions.
    for key, value in result.items():
        shared_state.set(key, value)

    logger.debug(
        "Indicators updated → RSI=%.1f  ADX=%.1f  ATR=%.2f  BB=%.2f  "
        "dist_ema50=%.3f  dist_ema200=%.3f",
        result["rsi_h1"], result["adx_h1"], result["atr_h1"],
        result["bb_position"], result["dist_ema50_pct"], result["dist_ema200_pct"],
    )

    return result


def calculate_and_run_ml(df: pd.DataFrame) -> tuple:
    """
    Convenience wrapper: calculate indicators then immediately run ML predict.
    Call this from candle_watcher.py instead of calling them separately.

    Returns (ml_score, ml_direction) and writes both to shared_state.

    BUG FIX: the original flow called calculate_indicators() and predict()
    independently, with predict() reading from shared_state which was still
    all zeros. This function ensures the freshly-computed indicator dict is
    passed directly to predict() in the same call.
    """
    indicators = calculate_indicators(df)
    if indicators is None:
        logger.warning("ML skipped — indicators returned None (not enough candles?)")
        return 0.0, "NONE"

    from ml.predict import predict
    score, direction = predict(df, indicators)

    shared_state.set("ml_score",     score)
    shared_state.set("ml_direction", direction)

    logger.info(
        "ML result → %.1f%%  %s  (RSI=%.1f  ADX=%.1f  dist50=%.3f)",
        score * 100, direction,
        indicators["rsi_h1"], indicators["adx_h1"], indicators["dist_ema50_pct"],
    )
    return score, direction


# ── Private helpers ───────────────────────────────────────────

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
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
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
    _, _, hist = _macd(close.iloc[:-1])
    return float(hist)
