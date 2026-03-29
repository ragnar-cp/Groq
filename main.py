# ============================================================
# main.py — XAUUSD AI Trading Bot — Entry Point
# ============================================================
#
# Starts 4 threads:
#   1. Tick Stream       — real-time price (live) / passive (replay)
#   2. Candle Watcher    — fires analysis on M15/H1/H4 close + replay engine
#   3. News Watcher      — polls news + ForexFactory every 3 min
#   4. Telegram Bot      — listens for commands and chat messages
#
# Usage:
#   python main.py            → uses MODE from config.py
#   python main.py --live     → override to live mode
#   python main.py --replay   → override to replay mode
#   python main.py --train    → train ML model only, then exit
# ============================================================

import os
import sys
import time
import logging
import threading
import argparse
from datetime import datetime, time as dtime

import config
import shared_state
from execution.alert import send_message

# ── Create required directories before anything else ──────────
os.makedirs("logs", exist_ok=True)
os.makedirs("ml",   exist_ok=True)

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def main():
    args = _parse_args()

    # ── CLI mode override ──────────────────────────────────────
    if args.train:
        logger.info("Training ML model…")
        from ml.train import train
        train()
        logger.info("Training complete. Exiting.")
        return

    if args.live:
        config.MODE = "live"
    elif args.replay:
        config.MODE = "replay"

    shared_state.set("mode", config.MODE)
    logger.info("=" * 60)
    logger.info("XAUUSD AI Trading Bot starting in %s mode", config.MODE.upper())
    logger.info("=" * 60)

    # ── MT5 connection (live mode only) ───────────────────────
    if config.MODE == "live":
        from data.mt5_feed import connect, disconnect
        if not connect():
            logger.critical("Cannot connect to MT5. Exiting.")
            sys.exit(1)
    else:
        logger.info("Replay mode — MT5 connection skipped.")
        # Connect anyway to pull historical data for replay
        from data.mt5_feed import connect
        if not connect():
            logger.warning(
                "MT5 not available — replay will use stub data. "
                "Connect MT5 for real historical candles."
            )

    # ── Load ML model ─────────────────────────────────────────
    from ml.predict import load_model
    if not load_model():
        logger.warning(
            "No ML model found. Run: python main.py --train\n"
            "Bot will operate on Rule-based + LLM only until model is trained."
        )

    # ── Start threads ─────────────────────────────────────────
    threads = [
        threading.Thread(
            target=_run_tick_stream,
            name="tick_stream",
            daemon=True,
        ),
        threading.Thread(
            target=_run_candle_watcher,
            name="candle_watcher",
            daemon=True,
        ),
        threading.Thread(
            target=_run_news_watcher,
            name="news_watcher",
            daemon=True,
        ),
        threading.Thread(
            target=_run_telegram_bot,
            name="telegram_bot",
            daemon=True,
        ),
    ]

    for t in threads:
        t.start()
        logger.info("Thread started: %s", t.name)
        time.sleep(0.5)   # Stagger startup slightly

    # ── Notify user ───────────────────────────────────────────
    time.sleep(3)   # Give threads a moment to initialise
    _send_startup_message()

    # ── Main loop — daily reset + health watchdog ─────────────
    logger.info("All threads running. Main loop active.")
    last_day = datetime.now().date()

    try:
        while True:
            time.sleep(30)

            # Daily reset at midnight
            today = datetime.now().date()
            if today != last_day:
                last_day = today
                logger.info("New trading day — resetting daily stats.")
                from engine.risk_manager import reset_daily_stats
                reset_daily_stats()
                send_message("🌅 *New trading day* — daily stats reset. Bot scanning.")

            # Health watchdog — warn if tick stream goes stale (live mode)
            if config.MODE == "live":
                last_tick = shared_state.get("last_tick_received")
                if last_tick:
                    stale_secs = (datetime.now() - last_tick).total_seconds()
                    if stale_secs > config.HEALTH_TICK_STALE_SECONDS:
                        logger.warning("Tick stream stale (%.0f s). Check MT5 connection.", stale_secs)

            # Restart dead non-daemon threads (basic resilience)
            for t in threads:
                if not t.is_alive():
                    logger.error("Thread %s died — check logs.", t.name)
                    shared_state.set_thread_status(t.name, "dead")

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
        _shutdown()


# ── Thread entry points ───────────────────────────────────────

def _run_tick_stream():
    try:
        from data.tick_stream import start
        start()
    except Exception as exc:
        logger.error("tick_stream crashed: %s", exc, exc_info=True)
        shared_state.set_thread_status("tick_stream", f"crashed: {exc}")


def _run_candle_watcher():
    try:
        from data.candle_watcher import start
        start()
    except Exception as exc:
        logger.error("candle_watcher crashed: %s", exc, exc_info=True)
        shared_state.set_thread_status("candle_watcher", f"crashed: {exc}")


def _run_news_watcher():
    try:
        import time as _time
        from data.news_feed import start as news_start
        from data.economic_calendar import _refresh_calendar, _update_lockout_state

        # Run calendar + news together on the same thread
        def combined():
            while True:
                try:
                    _refresh_calendar()
                    _update_lockout_state()
                except Exception as exc:
                    logger.warning("Calendar update error: %s", exc)
                try:
                    from data.news_feed import _fetch_all, _check_breaking
                    from analysis.sentiment import get_sentiment_score
                    headlines = _fetch_all()
                    if headlines:
                        shared_state.set("latest_headlines", headlines[:15])
                        _check_breaking(headlines)
                        get_sentiment_score()   # Update sentiment on each news poll
                except Exception as exc:
                    logger.warning("News fetch error: %s", exc)
                shared_state.set_thread_status("news_watcher", "running")
                _time.sleep(180)   # Poll every 3 min

        combined()
    except Exception as exc:
        logger.error("news_watcher crashed: %s", exc, exc_info=True)
        shared_state.set_thread_status("news_watcher", f"crashed: {exc}")


def _run_telegram_bot():
    try:
        from telegram_bot.bot import start
        start()
    except Exception as exc:
        logger.error("telegram_bot crashed: %s", exc, exc_info=True)
        shared_state.set_thread_status("telegram_bot", f"crashed: {exc}")


# ── Helpers ───────────────────────────────────────────────────

def _send_startup_message():
    mode      = config.MODE.upper()
    mode_note = (
        f"🎭 *Replay:* {config.REPLAY_START_DATE} → {config.REPLAY_END_DATE}\n"
        f"Speed: {config.REPLAY_SPEED_SECONDS}s per candle\n"
        if config.MODE == "replay"
        else "📡 *Live market data active*\n"
    )
    send_message(
        f"🚀 *XAUUSD Bot Online* — `{mode}` mode\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"{mode_note}"
        f"• Bias gate: H4→H1→M15\n"
        f"• Auto-exec: OFF _(send /autoon to enable)_\n"
        f"• Risk: {config.BASE_RISK_PERCENT}% per trade\n"
        f"• Daily limit: {config.MAX_DAILY_LOSS_PCT}%\n\n"
        f"_Send /help for all commands or just ask me anything!_"
    )


def _shutdown():
    logger.info("Shutting down…")
    try:
        if config.MODE == "live":
            from data.mt5_feed import disconnect
            disconnect()
    except Exception:
        pass
    send_message("🛑 *Bot offline.* See you next session.")
    logger.info("Shutdown complete.")


def _parse_args():
    parser = argparse.ArgumentParser(description="XAUUSD AI Trading Bot")
    parser.add_argument("--live",   action="store_true", help="Run in live mode")
    parser.add_argument("--replay", action="store_true", help="Run in replay mode")
    parser.add_argument("--train",  action="store_true", help="Train ML model and exit")
    return parser.parse_args()


if __name__ == "__main__":
    main()
