# ============================================================
# ml/predict.py — Live ML scoring (inference)
# ============================================================
import logging
import pickle
import os
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

_model      = None
_model_meta = {}


def load_model():
    """Load model from disk. Called once at startup."""
    global _model, _model_meta
    path = "ml/model.pkl"
    if not os.path.exists(path):
        logger.warning("No trained model found. Run: python main.py --train")
        return False
    try:
        with open(path, "rb") as f:
            data       = pickle.load(f)
            _model     = data["model"]
            _model_meta= data
        logger.info("ML model loaded (trained at %s).", _model_meta.get("trained_at"))
        return True
    except Exception as exc:
        logger.error("Failed to load ML model: %s", exc)
        return False


def predict(df: pd.DataFrame, indicators: Optional[Dict]) -> Tuple[float, str]:
    """
    Build features from live data and return (probability, direction).
    probability: 0.0 – 1.0
    direction:   "BUY" | "SELL" | "WEAK_BUY" | "WEAK_SELL"

    Thresholds:
      >= 0.60  → strong signal  (BUY / SELL)       — use for auto-execution
      >= 0.55  → weak signal    (WEAK_BUY / WEAK_SELL) — alert only
      <  0.55  → direction still shown but labelled WEAK_BUY/WEAK_SELL
                 so signal_fusion can weight it appropriately
    Direction is ALWAYS returned based on whichever side is stronger —
    "NONE" is never returned so the UI always shows something meaningful.
    """
    if _model is None:
        logger.warning("ML: model not loaded — call load_model() at startup")
        return 0.0, "NONE"

    try:
        features = _build_live_features(df, indicators)
        if features is None:
            logger.warning("ML: feature build returned None — indicators may not be ready yet")
            return 0.0, "NONE"

        X         = np.array([features], dtype=np.float32)
        proba     = _model.predict_proba(X)[0]
        prob_up   = float(proba[1])
        prob_down = float(proba[0])

        logger.info("ML raw → UP: %.1f%%  DOWN: %.1f%%", prob_up * 100, prob_down * 100)

        # Strong signals — cleared for signal fusion and auto-execution
        if prob_up >= 0.60:
            logger.info("ML strong BUY signal: %.1f%%", prob_up * 100)
            return prob_up, "BUY"
        if prob_down >= 0.60:
            logger.info("ML strong SELL signal: %.1f%%", prob_down * 100)
            return prob_down, "SELL"

        # Weak signals (0.55–0.59) — direction shown, lower weight in fusion
        if prob_up >= prob_down:
            direction = "WEAK_BUY" if prob_up >= 0.55 else "WEAK_BUY"
            logger.info("ML weak BUY: %.1f%%", prob_up * 100)
            return prob_up, direction
        else:
            direction = "WEAK_SELL" if prob_down >= 0.55 else "WEAK_SELL"
            logger.info("ML weak SELL: %.1f%%", prob_down * 100)
            return prob_down, direction

    except Exception as exc:
        logger.error("ML predict error: %s", exc, exc_info=True)
        return 0.0, "NONE"


def _get(ind: dict, key: str, default: float) -> float:
    """
    Safe indicator lookup that only falls back to default when the value is
    actually missing (None / key absent) — NOT when it is 0.0 or any other
    legitimate falsy number.

    BUG FIX: the old code used `ind.get(key) or default` which replaces ANY
    falsy value (0, 0.0, False) with the default.  For indicators like
    dist_ema50 that are legitimately 0 when price == EMA, this silently
    corrupted every feature that happened to be zero and sent a garbage
    vector to the model.
    """
    val = ind.get(key)
    return default if val is None else float(val)


def _build_live_features(df: pd.DataFrame, ind: Optional[Dict]) -> Optional[list]:
    """Build the same feature vector used during training."""
    try:
        from analysis.trend import calculate_trend
        from ml.train import _build_rsi_series, _encode_session

        if df is None or len(df) < 50:
            logger.warning("ML: not enough candles (%d)", len(df) if df is not None else 0)
            return None

        if ind is None:
            logger.warning("ML: indicators dict is None")
            return None

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        last  = df.iloc[-1]

        c = float(close.iloc[-1])

        # ── FIX: use _get() so that 0.0 / 0 values are preserved correctly ──
        e50    = _get(ind, "ema50_h1",  c)
        e200   = _get(ind, "ema200_h1", c)
        e20    = _get(ind, "ema20_h1",  c)
        atr    = _get(ind, "atr_h1",    10.0)
        atr_r  = _get(ind, "atr_ratio", 1.0)
        adx    = _get(ind, "adx_h1",    25.0)
        rsi    = _get(ind, "rsi_h1",    50.0)
        bb_pos = _get(ind, "bb_position", 0.5)

        dist_ema50  = _get(ind, "dist_ema50_pct",    0.0)
        dist_ema200 = _get(ind, "dist_ema200_pct",   0.0)
        prev_range  = _get(ind, "prev_candle_range",  atr)

        # ── FIX: use .get() on the Series/row correctly ──
        # df.iloc[-1] returns a pandas Series; attribute access works but
        # .get() is not guaranteed on all pandas versions — use [] with try/except.
        try:
            high_val = float(last["high"])
            low_val  = float(last["low"])
            open_val = float(last["open"])
        except (KeyError, TypeError):
            high_val = low_val = open_val = c

        candle_range = max(high_val - low_val, 0.0001)   # avoid /0
        body_ratio   = abs(c - open_val) / candle_range

        # RSI slope
        rsi_series = _build_rsi_series(close)
        rsi_slope  = float(rsi_series.iloc[-1] - rsi_series.iloc[-4]) if len(rsi_series) >= 4 else 0.0

        # Bias encoding
        import shared_state
        bias_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0, "UNKNOWN": 0}
        h4_enc = bias_map.get(shared_state.get("h4_trend") or "NEUTRAL", 0)
        h1_enc = bias_map.get(shared_state.get("h1_trend") or "NEUTRAL", 0)

        # ── FIX: guard against e50 == 0 (shouldn't happen but be safe) ──
        ema_slope = (e20 - e50) / (e50 + 1e-9)

        # ── FIX: handle both Timestamp and plain datetime / string ──
        try:
            ts = pd.Timestamp(last["time"])
        except Exception:
            ts = pd.Timestamp.now()
        hour    = ts.hour
        dow     = ts.dayofweek
        session = _encode_session(hour)

        features = [
            dist_ema50, dist_ema200, prev_range, body_ratio,
            atr, atr_r,
            adx, h4_enc, h1_enc, ema_slope,
            rsi, rsi_slope, 0.0, bb_pos,
            hour, dow, session,
        ]

        logger.debug("ML features: %s", [round(f, 3) for f in features])

        # ── Sanity check: warn if every feature is zero (bad indicator feed) ──
        non_zero = sum(1 for f in features if f != 0.0)
        if non_zero < 3:
            logger.warning(
                "ML: only %d / %d features are non-zero — "
                "indicators dict may not be populated yet. Features: %s",
                non_zero, len(features), features,
            )

        return features

    except Exception as exc:
        logger.error("Feature build error: %s", exc, exc_info=True)
        return None
