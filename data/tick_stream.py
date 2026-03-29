# ============================================================
# data/tick_stream.py — Real-time tick monitoring thread
# ============================================================
import time
import logging
import threading

import shared_state
import config

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

_stop_event = threading.Event()


def start():
    """Entry point — called as a daemon thread from main.py."""
    shared_state.set_thread_status("tick_stream", "running")
    logger.info("Tick stream started.")

    if config.MODE == "replay":
        # In replay mode ticks are simulated by the candle watcher.
        logger.info("Replay mode — tick stream is passive (candle watcher drives price).")
        while not _stop_event.is_set():
            time.sleep(1)
        shared_state.set_thread_status("tick_stream", "stopped")
        return

    _run_live()


def stop():
    _stop_event.set()


def _run_live():
    """Continuously poll MT5 for the latest tick."""
    symbol = config.SYMBOL
    last_bid = 0.0

    while not _stop_event.is_set():
        try:
            if not MT5_AVAILABLE or not shared_state.get("mt5_connected"):
                time.sleep(1)
                continue

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                time.sleep(0.1)
                continue

            bid, ask = tick.bid, tick.ask

            # Only process if price actually changed
            if bid != last_bid:
                shared_state.update_tick(bid, ask)
                last_bid = bid
                _check_open_trade(bid, ask)
                _check_breakout(bid)

            time.sleep(0.05)   # ~20 polls per second — enough for gold

        except Exception as exc:
            logger.error("Tick stream error: %s", exc, exc_info=True)
            shared_state.set_thread_status("tick_stream", f"error: {exc}")
            time.sleep(2)

    shared_state.set_thread_status("tick_stream", "stopped")
    logger.info("Tick stream stopped.")


def _check_open_trade(bid: float, ask: float):
    """Monitor an open trade's SL/TP on every tick."""
    trade = shared_state.get("open_trade")
    if trade is None:
        return

    direction = trade.get("direction")
    sl        = trade.get("sl")
    tp1       = trade.get("tp1")
    tp2       = trade.get("tp2")
    tp1_hit   = trade.get("tp1_hit", False)
    trail_sl  = trade.get("trail_sl")

    price = bid if direction == "BUY" else ask   # exit price depends on direction

    if direction == "BUY":
        # ── Check TP1 ──
        if not tp1_hit and tp1 and price >= tp1:
            _on_tp1_hit(trade, price)
            return

        # ── Update trailing stop after TP1 ──
        if tp1_hit and trail_sl:
            new_trail = price - trade["atr"] * config.ATR_TRAIL_MULTIPLIER
            if new_trail > trail_sl:
                trade["trail_sl"] = new_trail
                shared_state.set_open_trade(trade)

        # ── Check TP2 ──
        if tp1_hit and tp2 and price >= tp2:
            _on_tp2_hit(trade, price)
            return

        # ── Check trailing stop / SL ──
        effective_sl = trail_sl if (tp1_hit and trail_sl) else sl
        if effective_sl and price <= effective_sl:
            _on_sl_hit(trade, price)

    elif direction == "SELL":
        if not tp1_hit and tp1 and price <= tp1:
            _on_tp1_hit(trade, price)
            return

        if tp1_hit and trail_sl:
            new_trail = price + trade["atr"] * config.ATR_TRAIL_MULTIPLIER
            if new_trail < trail_sl:
                trade["trail_sl"] = new_trail
                shared_state.set_open_trade(trade)

        if tp1_hit and tp2 and price <= tp2:
            _on_tp2_hit(trade, price)
            return

        effective_sl = trail_sl if (tp1_hit and trail_sl) else sl
        if effective_sl and price >= effective_sl:
            _on_sl_hit(trade, price)


def _on_tp1_hit(trade: dict, price: float):
    from execution.alert import send_tp1_alert
    logger.info("TP1 hit at %.2f", price)
    trade["tp1_hit"]  = True
    trade["trail_sl"] = trade["entry"]   # Move SL to breakeven
    shared_state.set_open_trade(trade)
    send_tp1_alert(trade, price)


def _on_tp2_hit(trade: dict, price: float):
    from execution.alert import send_tp2_alert
    from engine.trade_memory import record_trade
    logger.info("TP2 hit at %.2f", price)
    pnl_pips = abs(price - trade["entry"]) / 0.1
    record_trade(trade, result="WIN", exit_price=price,
                 exit_reason="TP2 hit", pnl_pips=pnl_pips)
    send_tp2_alert(trade, price)
    shared_state.clear_open_trade()
    _update_daily_pnl(pnl_pips, win=True)


def _on_sl_hit(trade: dict, price: float):
    from execution.alert import send_sl_alert
    from engine.trade_memory import record_trade
    logger.info("SL/Trail hit at %.2f", price)
    pnl_pips = -abs(price - trade["entry"]) / 0.1 if not trade.get("tp1_hit") \
               else abs(price - trade["entry"]) / 0.1
    win = trade.get("tp1_hit", False)   # If TP1 was hit we still locked profit
    reason = "Trailing stop hit" if trade.get("tp1_hit") else "SL hit"
    record_trade(trade, result="WIN" if win else "LOSS",
                 exit_price=price, exit_reason=reason, pnl_pips=pnl_pips)
    send_sl_alert(trade, price, reason)
    shared_state.clear_open_trade()
    _update_daily_pnl(pnl_pips, win=win)


def _check_breakout(price: float):
    """Alert if price crosses a key S&R level."""
    resistance = shared_state.get("key_resistance")
    support    = shared_state.get("key_support")
    last_price = shared_state.get("live_price")

    if resistance and last_price and last_price < resistance <= price:
        from execution.alert import send_breakout_alert
        send_breakout_alert("resistance", resistance)

    if support and last_price and last_price > support >= price:
        from execution.alert import send_breakout_alert
        send_breakout_alert("support", support)


def _update_daily_pnl(pnl_pips: float, win: bool):
    account = None
    try:
        from data.mt5_feed import get_account_info
        account = get_account_info()
    except Exception:
        pass

    current = shared_state.get("daily_pnl") or 0.0
    shared_state.set("daily_pnl", current + pnl_pips)

    if not win:
        losses = shared_state.get("consecutive_losses") or 0
        shared_state.set("consecutive_losses", losses + 1)
    else:
        shared_state.set("consecutive_losses", 0)

    # Daily loss limit check
    if account:
        balance = account.get("balance", 1)
        daily_loss_pct = abs(min(shared_state.get("daily_pnl"), 0)) / balance * 100
        if daily_loss_pct >= config.MAX_DAILY_LOSS_PCT:
            shared_state.set("daily_paused", True)
            from execution.alert import send_message
            send_message(
                f"🚨 *Daily loss limit hit* ({config.MAX_DAILY_LOSS_PCT}%)\n"
                f"Bot is paused for today. Will resume tomorrow automatically."
            )
            logger.warning("Daily loss limit reached — pausing for today.")
