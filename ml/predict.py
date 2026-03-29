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
    if not os.path.exists(config.MODEL_PATH if hasattr(config, "MODEL_PATH") else "ml/model.pkl"):
        logger.warning("No trained model found. Run ml/train.py first.")
        return False
    try:
        with open("ml/model.pkl", "rb") as f:
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
    probability: 0.0 – 1.0 (probability of price going UP)
    direction:   "BUY" | "SELL" | "NONE"
    """
    if _model is None:
        return 0.5, "NONE"

    try:
        features = _build_live_features(df, indicators)
        if features is None:
            return 0.5, "NONE"

        X = np.array([features], dtype=np.float32)
        prob_up = float(_model.predict_proba(X)[0][1])

        if prob_up >= 0.60:
            return prob_up, "BUY"
        elif prob_up <= 0.40:
            return 1 - prob_up, "SELL"
        else:
            return 0.5, "NONE"

    except Exception as exc:
        logger.error("ML predict error: %s", exc)
        return 0.5, "NONE"


def _build_live_features(df: pd.DataFrame, ind: Optional[Dict]) -> Optional[list]:
    """Build the same feature vector used during training."""
    try:
        from analysis.trend import calculate_trend
        from ml.train import _build_rsi_series, _build_adx_series, _encode_session

        if ind is None or len(df) < 50:
            return None

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        last  = df.iloc[-1]

        c      = float(close.iloc[-1])
        e50    = ind.get("ema50_h1", c)
        e200   = ind.get("ema200_h1", c)
        e20    = ind.get("ema20_h1", c)
        atr    = ind.get("atr_h1", 10)
        atr_r  = ind.get("atr_ratio", 1.0)
        adx    = ind.get("adx_h1", 25)
        rsi    = ind.get("rsi_h1", 50)
        bb_pos = ind.get("bb_position", 0.5)

        dist_ema50  = ind.get("dist_ema50_pct",  0)
        dist_ema200 = ind.get("dist_ema200_pct", 0)
        prev_range  = ind.get("prev_candle_range", atr)
        body_ratio  = abs(last["close"] - last["open"]) / (last["high"] - last["low"] + 1e-9)

        # RSI slope
        rsi_series = _build_rsi_series(close)
        rsi_slope  = float(rsi_series.iloc[-1] - rsi_series.iloc[-4]) if len(rsi_series) >= 4 else 0

        # Bias encoding
        import shared_state
        h4_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0, "UNKNOWN": 0}
        h4_enc = h4_map.get(shared_state.get("h4_trend") or "NEUTRAL", 0)
        h1_enc = h4_map.get(shared_state.get("h1_trend") or "NEUTRAL", 0)

        # EMA slope
        ema_slope = (e20 - e50) / (e50 + 1e-9)

        # Time
        ts      = pd.Timestamp(last["time"])
        hour    = ts.hour
        dow     = ts.dayofweek
        session = _encode_session(hour)

        return [
            dist_ema50, dist_ema200, prev_range, body_ratio,
            atr, atr_r,
            adx, h4_enc, h1_enc, ema_slope,
            rsi, rsi_slope, 0.0, bb_pos,
            hour, dow, session,
        ]

    except Exception as exc:
        logger.error("Feature build error: %s", exc)
        return None
