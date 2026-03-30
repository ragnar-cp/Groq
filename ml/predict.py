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
    direction:   "BUY" | "SELL" | "NONE"
    """
    if _model is None:
        return 0.0, "NONE"

    try:
        features = _build_live_features(df, indicators)
        if features is None:
            logger.warning("ML: feature build returned None — indicators may not be ready yet")
            return 0.0, "NONE"

        X       = np.array([features], dtype=np.float32)
        proba   = _model.predict_proba(X)[0]
        prob_up   = float(proba[1])
        prob_down = float(proba[0])

        logger.info("ML raw → UP: %.1f%%  DOWN: %.1f%%", prob_up*100, prob_down*100)

        if prob_up >= 0.60:
            return prob_up, "BUY"
        elif prob_down >= 0.60:
            return prob_down, "SELL"
        else:
            # Uncertain zone — return stronger probability for display, direction NONE
            stronger = prob_up if prob_up >= prob_down else prob_down
            return stronger, "NONE"

    except Exception as exc:
        logger.error("ML predict error: %s", exc)
        return 0.0, "NONE"


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

        c      = float(close.iloc[-1])
        e50    = ind.get("ema50_h1")  or c
        e200   = ind.get("ema200_h1") or c
        e20    = ind.get("ema20_h1")  or c
        atr    = ind.get("atr_h1")    or 10.0
        atr_r  = ind.get("atr_ratio") or 1.0
        adx    = ind.get("adx_h1")    or 25.0
        rsi    = ind.get("rsi_h1")    or 50.0
        bb_pos = ind.get("bb_position") or 0.5

        dist_ema50  = ind.get("dist_ema50_pct",  0) or 0
        dist_ema200 = ind.get("dist_ema200_pct", 0) or 0
        prev_range  = ind.get("prev_candle_range", atr) or atr

        high_val = last.get("high", c) if hasattr(last, "get") else float(last["high"])
        low_val  = last.get("low",  c) if hasattr(last, "get") else float(last["low"])
        open_val = last.get("open", c) if hasattr(last, "get") else float(last["open"])
        body_ratio = abs(c - open_val) / max(high_val - low_val, 0.01)

        # RSI slope
        rsi_series = _build_rsi_series(close)
        rsi_slope  = float(rsi_series.iloc[-1] - rsi_series.iloc[-4]) if len(rsi_series) >= 4 else 0.0

        # Bias encoding
        import shared_state
        h4_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0, "UNKNOWN": 0}
        h4_enc = h4_map.get(shared_state.get("h4_trend") or "NEUTRAL", 0)
        h1_enc = h4_map.get(shared_state.get("h1_trend") or "NEUTRAL", 0)

        ema_slope = (e20 - e50) / (e50 + 1e-9)

        ts      = pd.Timestamp(last["time"])
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
        return features

    except Exception as exc:
        logger.error("Feature build error: %s", exc, exc_info=True)
        return None
