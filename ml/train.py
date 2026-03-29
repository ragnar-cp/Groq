# ============================================================
# ml/train.py — XGBoost model trainer on MT5 historical data
# ============================================================
import os
import logging
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = "ml/model.pkl"


def train():
    """
    Full training pipeline:
    1. Fetch labelled history from MT5
    2. Build feature matrix
    3. Train XGBoost
    4. Save model to disk
    """
    logger.info("Starting ML training pipeline…")

    # ── Ensure MT5 is connected ───────────────────────────────
    from data.mt5_feed import connect
    import shared_state
    if not shared_state.get("mt5_connected"):
        logger.info("Connecting to MT5…")
        if not connect():
            logger.error(
                "Cannot connect to MT5. Training aborted.\n"
                "Fix checklist:\n"
                "  1. Open MetaTrader 5 desktop app and log in\n"
                "  2. Check MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in config.py\n"
                "  3. Server name must match exactly (check File > Open Account in MT5)"
            )
            return

    from data.historical_fetcher import fetch_for_training
    df = fetch_for_training()
    if df is None or df.empty:
        logger.error("Training aborted — no data.")
        return

    X, y = build_features(df)
    if X is None:
        logger.error("Feature building failed.")
        return

    logger.info("Training XGBoost on %d samples…", len(X))
    model = _train_xgboost(X, y)

    os.makedirs("ml", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "trained_at": datetime.now()}, f)

    logger.info("Model saved to %s", MODEL_PATH)
    _evaluate(model, X, y)


def build_features(df: pd.DataFrame) -> tuple:
    """
    Build feature matrix from labelled candle DataFrame.
    All features match exactly what predict.py builds at inference time.
    """
    try:
        from analysis.indicators import (
            _rsi, _macd, _atr, _adx, _bollinger, _support_resistance
        )
        from analysis.trend import calculate_trend
        import config

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        n     = len(df)

        features = []
        labels   = []

        # Pre-compute series
        ema20  = close.ewm(span=config.EMA_FAST, adjust=False).mean()
        ema50  = close.ewm(span=config.EMA_MID,  adjust=False).mean()
        ema200 = close.ewm(span=config.EMA_SLOW, adjust=False).mean()
        atr14  = _atr(high, low, close, 14)
        atr_avg= atr14.rolling(50).mean()

        rsi_series = _build_rsi_series(close)
        adx_series = _build_adx_series(high, low, close)

        for i in range(config.EMA_SLOW + 20, n):
            row   = df.iloc[i]
            label = row.get("label", -1)
            if label == -1:
                continue

            c = float(close.iloc[i])

            # ── Price features ────────────────────────────────
            e20  = float(ema20.iloc[i])
            e50  = float(ema50.iloc[i])
            e200 = float(ema200.iloc[i])
            dist_ema50  = (c - e50)  / e50  * 100 if e50  != 0 else 0
            dist_ema200 = (c - e200) / e200 * 100 if e200 != 0 else 0
            prev_range  = float(high.iloc[i-1] - low.iloc[i-1])
            body_ratio  = abs(row["close"] - row["open"]) / (row["high"] - row["low"] + 1e-9)

            # ── Volatility features ───────────────────────────
            atr_val   = float(atr14.iloc[i])
            atr_r     = atr_val / float(atr_avg.iloc[i]) if float(atr_avg.iloc[i]) != 0 else 1.0

            # ── Trend / momentum ─────────────────────────────
            adx_val   = float(adx_series.iloc[i]) if i < len(adx_series) else 25.0
            rsi_val   = float(rsi_series.iloc[i]) if i < len(rsi_series) else 50.0
            rsi_slope = float(rsi_series.iloc[i] - rsi_series.iloc[i-3]) if i >= 3 else 0.0

            # ── H4 bias encoded ───────────────────────────────
            # Use last 96 M15 candles (~1 day) as proxy for H4 trend
            slice_h4 = df.iloc[max(0, i-96):i+1]
            h4_trend = calculate_trend(slice_h4)
            h4_enc   = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}.get(h4_trend, 0)

            # H1 proxy (last 4 M15 candles)
            slice_h1 = df.iloc[max(0, i-4):i+1]
            h1_trend = calculate_trend(slice_h1)
            h1_enc   = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}.get(h1_trend, 0)

            # ── Time/session features ─────────────────────────
            ts         = pd.Timestamp(row["time"])
            hour       = ts.hour
            dow        = ts.dayofweek
            session    = _encode_session(hour)

            # ── Bollinger position ────────────────────────────
            slice20 = close.iloc[max(0,i-20):i+1]
            if len(slice20) >= 20:
                bb_mean = float(slice20.mean())
                bb_std  = float(slice20.std())
                bb_pos  = (c - (bb_mean - 2*bb_std)) / (4*bb_std + 1e-9)
            else:
                bb_pos = 0.5

            feat_row = [
                dist_ema50, dist_ema200, prev_range, body_ratio,
                atr_val, atr_r,
                adx_val, h4_enc, h1_enc,
                (e20 - e50) / (e50 + 1e-9),   # EMA slope
                rsi_val, rsi_slope,
                0.0,   # MACD placeholder (expensive to compute per-row)
                bb_pos,
                hour, dow, session,
            ]
            features.append(feat_row)
            labels.append(int(label))

        if not features:
            return None, None

        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)

    except Exception as exc:
        logger.error("build_features error: %s", exc, exc_info=True)
        return None, None


def _train_xgboost(X, y):
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False   # No shuffle — time series
        )

        # H4 bias feature is index 7 — give it extra importance
        sample_weight = np.ones(len(y_train))

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        return model

    except ImportError:
        logger.warning("XGBoost not installed — using RandomForest fallback.")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        model.fit(X, y)
        return model


def _evaluate(model, X, y):
    try:
        from sklearn.metrics import classification_report
        preds = model.predict(X[-1000:])
        report = classification_report(y[-1000:], preds, target_names=["SELL","BUY"])
        logger.info("Model evaluation (last 1000 samples):\n%s", report)
    except Exception as exc:
        logger.warning("Evaluation failed: %s", exc)


def _build_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _build_adx_series(high, low, close, period=14) -> pd.Series:
    import numpy as np
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    plus_dm    = (high - prev_high).clip(lower=0)
    minus_dm   = (prev_low - low).clip(lower=0)
    plus_dm    = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm   = minus_dm.where(minus_dm > plus_dm, 0)
    prev_close = close.shift(1)
    tr         = pd.concat([high - low,
                             (high - prev_close).abs(),
                             (low  - prev_close).abs()], axis=1).max(axis=1)
    atr_s      = tr.ewm(span=period, adjust=False).mean()
    plus_di    = 100 * plus_dm.ewm(span=period,  adjust=False).mean() / atr_s
    minus_di   = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s
    dx_denom   = (plus_di + minus_di).replace(0, float("nan"))
    dx         = 100 * (plus_di - minus_di).abs() / dx_denom
    return dx.ewm(span=period, adjust=False).mean()


def _encode_session(hour: int) -> int:
    """Encode trading session from UTC hour."""
    if 0  <= hour < 8:  return 0   # Asian
    if 8  <= hour < 12: return 1   # London open
    if 12 <= hour < 16: return 3   # NY/London overlap (highest volume)
    if 16 <= hour < 21: return 2   # NY session
    return 4                        # Close


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
