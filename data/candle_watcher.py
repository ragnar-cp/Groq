# ============================================================
# data/candle_watcher.py — Candle close detector + Replay engine
# ============================================================
import time
import logging
import threading
from datetime import datetime, timedelta

import pandas as pd

import shared_state
import config

logger = logging.getLogger(__name__)

_stop_event = threading.Event()

_last_candle_time = {"M15": None, "H1": None, "H4": None}


def start():
    """Entry point — called as a daemon thread from main.py."""
    shared_state.set_thread_status("candle_watcher", "running")

    if config.MODE == "replay":
        logger.info("Starting in REPLAY mode (%s → %s)",
                    config.REPLAY_START_DATE, config.REPLAY_END_DATE)
        _run_replay()
    else:
        logger.info("Starting in LIVE mode — watching M15/H1/H4 closes.")
        _run_live()


def stop():
    _stop_event.set()


# ── Warm-up helpers ───────────────────────────────────────────

def _warm_up_indicators(get_candles_fn):
    """Pre-populate shared_state indicators immediately on live startup."""
    try:
        from analysis.trend import calculate_trend
        from analysis.indicators import calculate_indicators

        for tf in ["H4", "H1"]:
            df = get_candles_fn(tf, count=250)
            if df is not None and not df.empty:
                trend = calculate_trend(df)
                shared_state.set(f"{tf.lower()}_trend", trend)
                logger.info("Startup %s trend: %s", tf, trend)

        m15_df = get_candles_fn("M15", count=250)
        if m15_df is not None and not m15_df.empty:
            mid = float(m15_df.iloc[-1]["close"])
            spread = 0.3
            shared_state.update_tick(mid - spread / 2, mid + spread / 2)
            indicators = calculate_indicators(m15_df)
            if indicators:
                for k, v in indicators.items():
                    shared_state.set(k, v)
            logger.info("Startup indicators populated. Price: %.2f", mid)

        _update_bias_state()

        # ── BUG FIX: run ML immediately on startup so score is never 0 ──
        if m15_df is not None and not m15_df.empty:
            _run_ml_only(m15_df)

    except Exception as exc:
        logger.warning("Warm-up failed: %s", exc)


def _warm_up_indicators_from_df(m15_df, h1_df, h4_df):
    """Pre-populate shared_state from replay history slices."""
    try:
        from analysis.trend import calculate_trend
        from analysis.indicators import calculate_indicators

        if h4_df is not None and not h4_df.empty:
            trend = calculate_trend(h4_df.iloc[:50] if len(h4_df) >= 50 else h4_df)
            shared_state.set("h4_trend", trend)

        if h1_df is not None and not h1_df.empty:
            trend = calculate_trend(h1_df.iloc[:50] if len(h1_df) >= 50 else h1_df)
            shared_state.set("h1_trend", trend)

        indicators = calculate_indicators(m15_df)
        if indicators:
            for k, v in indicators.items():
                shared_state.set(k, v)

        mid = float(m15_df.iloc[-1]["close"])
        spread = 0.3
        shared_state.update_tick(mid - spread / 2, mid + spread / 2)
        _update_bias_state()

        # ── BUG FIX: run ML on warm-up so score is never 0 ──
        _run_ml_only(m15_df)

        logger.info("Replay warm-up done. Price: %.2f RSI: %.1f ML: %.1f%%",
                    mid,
                    indicators.get("rsi_h1", 0) if indicators else 0,
                    shared_state.get("ml_score") * 100)
    except Exception as exc:
        logger.warning("Replay warm-up failed: %s", exc)


# ── Live mode ─────────────────────────────────────────────────

def _run_live():
    """Poll MT5 every 5 seconds; trigger analysis when a candle closes."""
    from data.mt5_feed import get_candles

    _warm_up_indicators(get_candles)

    while not _stop_event.is_set():
        try:
            for tf in ["M15", "H1", "H4"]:
                df = get_candles(tf, count=5)
                if df is None or df.empty:
                    continue

                latest_time = df.iloc[-1]["time"]

                if _last_candle_time[tf] is None:
                    _last_candle_time[tf] = latest_time
                    continue

                if latest_time > _last_candle_time[tf]:
                    logger.info("%s candle closed at %s", tf, latest_time)
                    _last_candle_time[tf] = latest_time
                    _on_candle_close(tf, df)

            time.sleep(5)

        except Exception as exc:
            logger.error("Candle watcher error: %s", exc, exc_info=True)
            shared_state.set_thread_status("candle_watcher", f"error: {exc}")
            time.sleep(10)

    shared_state.set_thread_status("candle_watcher", "stopped")


# ── Replay mode ───────────────────────────────────────────────

def _run_replay():
    from data.historical_fetcher import fetch_range_for_replay

    shared_state.set("mode", "replay")

    start_dt = datetime.strptime(config.REPLAY_START_DATE, "%Y-%m-%d")
    end_dt   = datetime.strptime(config.REPLAY_END_DATE,   "%Y-%m-%d")

    logger.info("Loading replay data %s → %s …", start_dt.date(), end_dt.date())
    m15_df = fetch_range_for_replay("M15", start_dt, end_dt)
    h1_df  = fetch_range_for_replay("H1",  start_dt, end_dt)
    h4_df  = fetch_range_for_replay("H4",  start_dt, end_dt)

    if m15_df is None or m15_df.empty:
        logger.error("No replay data available. Check MT5 connection and dates.")
        shared_state.set_thread_status("candle_watcher", "error: no replay data")
        return

    logger.info("Replay loaded — %d M15 candles. Starting playback…", len(m15_df))

    if len(m15_df) >= 200:
        _warm_up_indicators_from_df(m15_df.iloc[:200], h1_df, h4_df)

    for i, row in m15_df.iterrows():
        if _stop_event.is_set():
            break

        candle_time = row["time"]
        mid = row["close"]
        spread = shared_state.get("spread") or 0.3
        shared_state.update_tick(mid - spread / 2, mid + spread / 2)
        shared_state.set("replay_progress", str(candle_time))

        m15_slice = m15_df.iloc[: i + 1].copy()
        h1_slice  = h1_df[h1_df["time"] <= candle_time].copy() if h1_df is not None else None
        h4_slice  = h4_df[h4_df["time"] <= candle_time].copy() if h4_df is not None else None

        if h4_slice is not None and not h4_slice.empty:
            last_h4 = h4_slice.iloc[-1]["time"]
            if _last_candle_time["H4"] != last_h4:
                _last_candle_time["H4"] = last_h4
                _on_candle_close("H4", h4_slice)

        if h1_slice is not None and not h1_slice.empty:
            last_h1 = h1_slice.iloc[-1]["time"]
            if _last_candle_time["H1"] != last_h1:
                _last_candle_time["H1"] = last_h1
                _on_candle_close("H1", h1_slice)

        _on_candle_close("M15", m15_slice)
        time.sleep(config.REPLAY_SPEED_SECONDS)

    logger.info("Replay complete.")
    shared_state.set_thread_status("candle_watcher", "replay complete")
    from execution.alert import send_message
    send_message("✅ *Replay complete.* Review results with /trades and /memory.")


# ── Shared close handler ──────────────────────────────────────

def _on_candle_close(timeframe: str, df: pd.DataFrame):
    try:
        from analysis.trend import calculate_trend

        if timeframe in ("H1", "H4"):
            trend = calculate_trend(df)
            shared_state.set(f"{timeframe.lower()}_trend", trend)
            _update_bias_state()
            return

        # M15 — always run indicators + ML first (no gates),
        # then run trade logic only if gates pass
        _run_m15_analysis(df)

    except Exception as exc:
        logger.error("on_candle_close(%s) error: %s", timeframe, exc, exc_info=True)


# ── ML-only update (no gates, always runs) ────────────────────

def _run_ml_only(m15_df: pd.DataFrame):
    """
    Run indicators + ML and write results to shared_state.
    Called unconditionally — no trading gates applied here.
    This ensures ml_score and ml_direction are always current,
    even when the bot is paused, in consolidation, or a trade is open.
    """
    try:
        from analysis.indicators import calculate_indicators
        from ml.predict import predict

        indicators = calculate_indicators(m15_df)
        if not indicators:
            logger.warning("ML-only update skipped — indicators returned None")
            return

        ml_score, ml_direction = predict(m15_df, indicators)
        shared_state.set("ml_score",     ml_score)
        shared_state.set("ml_direction", ml_direction)

        logger.info(
            "ML → %.1f%% %s  (RSI=%.1f  ADX=%.1f  bb=%.2f)",
            ml_score * 100, ml_direction,
            indicators.get("rsi_h1",    0),
            indicators.get("adx_h1",    0),
            indicators.get("bb_position", 0),
        )

    except Exception as exc:
        logger.error("ML-only update error: %s", exc, exc_info=True)


# ── Full M15 analysis pipeline ────────────────────────────────

def _run_m15_analysis(m15_df: pd.DataFrame):
    """
    BUG FIX: ML and indicators now run BEFORE any trading gates.
    Gates only block signal generation and trade execution.
    This means ml_score is always up to date in the UI, and
    breakout detections still fire even when trading is gated.
    """
    # ── Step 1: Always run indicators + ML (no gates) ─────────
    try:
        from analysis.indicators import calculate_indicators
        from ml.predict import predict

        indicators = calculate_indicators(m15_df)
        if not indicators:
            logger.warning("Indicators returned None — skipping M15 analysis")
            return

        ml_score, ml_direction = predict(m15_df, indicators)
        shared_state.set("ml_score",     ml_score)
        shared_state.set("ml_direction", ml_direction)

        logger.info(
            "M15 ML → %.1f%% %s  (RSI=%.1f  ADX=%.1f  bb=%.2f  dist50=%.3f)",
            ml_score * 100, ml_direction,
            indicators.get("rsi_h1",       0),
            indicators.get("adx_h1",       0),
            indicators.get("bb_position",  0),
            indicators.get("dist_ema50_pct", 0),
        )

    except Exception as exc:
        logger.error("Indicators/ML error: %s", exc, exc_info=True)
        return

    # ── Step 2: Trading gates — block signal/execution only ───
    # ML score above is already written regardless of these gates.
    if shared_state.get("bot_paused"):
        logger.debug("Signal skipped — bot paused")
        return
    if shared_state.get("daily_paused"):
        logger.debug("Signal skipped — daily loss limit hit")
        return
    if shared_state.is_trade_open():
        logger.debug("Signal skipped — trade already open")
        return
    if shared_state.get("bias_state") == "CONSOLIDATING":
        logger.debug("Signal skipped — bias is CONSOLIDATING")
        return
    if shared_state.get("in_news_lockout"):
        logger.debug("Signal skipped — news lockout active")
        return

    # ── Step 3: Pattern detection + signal fusion ─────────────
    try:
        from analysis.candle_patterns import detect_candle_patterns
        from analysis.chart_patterns  import detect_chart_patterns
        from analysis.sentiment       import get_sentiment_score
        from engine.signal_fusion     import fuse_signals

        candle_pattern = detect_candle_patterns(m15_df)
        chart_pattern  = detect_chart_patterns(m15_df)
        sentiment      = shared_state.get("sentiment_score") or "NEUTRAL"

        signal = fuse_signals(
            indicators=indicators,
            candle_pattern=candle_pattern,
            chart_pattern=chart_pattern,
            ml_score=ml_score,
            ml_direction=ml_direction,
            sentiment=sentiment,
        )

        if signal:
            _handle_signal(signal)
        else:
            logger.debug(
                "No signal this candle (ML=%.1f%% %s, bias=%s, pattern=%s)",
                ml_score * 100, ml_direction,
                shared_state.get("bias_state"), candle_pattern,
            )

    except Exception as exc:
        logger.error("Signal fusion error: %s", exc, exc_info=True)


def _update_bias_state():
    """Derive FULL_BULL / FULL_BEAR / CONSOLIDATING from H4 + H1 trends."""
    h4 = shared_state.get("h4_trend")
    h1 = shared_state.get("h1_trend")
    prev_bias = shared_state.get("bias_state")

    if h4 == "BULLISH" and h1 == "BULLISH":
        bias, side = "FULL_BULL", "BUY"
    elif h4 == "BEARISH" and h1 == "BEARISH":
        bias, side = "FULL_BEAR", "SELL"
    else:
        bias, side = "CONSOLIDATING", "NONE"

    shared_state.set("bias_state",   bias)
    shared_state.set("allowed_side", side)

    if bias != prev_bias:
        shared_state.set("bias_changed_at", datetime.now())
        logger.info("Bias changed: %s → %s", prev_bias, bias)
        _send_bias_change_alert(prev_bias, bias, h4, h1)


def _send_bias_change_alert(old: str, new: str, h4: str, h1: str):
    try:
        from execution.alert import send_message
        emoji_map = {"FULL_BULL": "🟢", "FULL_BEAR": "🔴", "CONSOLIDATING": "🟡"}
        old_e = emoji_map.get(old, "⚪")
        new_e = emoji_map.get(new, "⚪")
        side_note = {
            "FULL_BULL":     "M15 engine active — scanning BUY setups only.",
            "FULL_BEAR":     "M15 engine active — scanning SELL setups only.",
            "CONSOLIDATING": "H4/H1 conflict — standing down to protect capital.",
        }.get(new, "")
        msg = (
            f"🔔 *BIAS CHANGE*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Previous: {old_e} {old}\n"
            f"New:      {new_e} {new}\n\n"
            f"H4: {h4}   H1: {h1}\n\n"
            f"_{side_note}_"
        )
        send_message(msg)
    except Exception as exc:
        logger.warning("Could not send bias change alert: %s", exc)


def _handle_signal(signal: dict):
    """Route a confirmed signal to alert and optionally execute."""
    from execution.alert import send_signal_alert
    from engine.risk_manager import calculate_position

    signal = calculate_position(signal)

    logger.info("Signal: %s confidence=%.0f%%", signal["direction"], signal["confidence"])
    shared_state.set("last_signal",      signal)
    shared_state.set("last_signal_time", datetime.now())

    if signal["confidence"] >= config.ALERT_CONFIDENCE_MIN:
        send_signal_alert(signal)

    if (
        shared_state.get("auto_exec")
        and signal["confidence"] >= config.AUTOEXEC_CONFIDENCE_MIN
        and config.MODE != "replay"
    ):
        from execution.mt5_executor import execute_trade
        execute_trade(signal)
