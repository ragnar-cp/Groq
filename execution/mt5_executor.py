# ============================================================
# execution/mt5_executor.py — Trade placement on MT5
# ============================================================
import logging
from typing import Dict, Optional

import shared_state
import config

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


def execute_trade(signal: Dict) -> bool:
    """
    Place a trade on MT5 based on a confirmed signal.
    Returns True if order was placed successfully.
    Never called in replay mode.
    """
    if config.MODE == "replay":
        logger.info("Replay mode — skipping real execution.")
        return False

    if not MT5_AVAILABLE or not shared_state.get("mt5_connected"):
        logger.error("MT5 not connected — cannot execute trade.")
        return False

    if shared_state.is_trade_open():
        logger.warning("Trade already open — skipping new signal.")
        return False

    if not _pre_trade_checks(signal):
        return False

    direction  = signal["direction"]
    lot_size   = signal.get("lot_size", 0.01)
    sl         = signal["sl"]
    tp1        = signal["tp1"]

    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    price      = mt5.symbol_info_tick(config.SYMBOL).ask if direction == "BUY" \
                 else mt5.symbol_info_tick(config.SYMBOL).bid

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       config.SYMBOL,
        "volume":       lot_size,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp1,        # MT5 TP set to TP1; TP2 managed by tick_stream
        "deviation":    10,
        "magic":        20240101,
        "comment":      f"XAUBOT {direction} {signal.get('confidence',0)}%",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = result.retcode if result else "None"
        logger.error("Order failed: retcode=%s", err)
        from execution.alert import send_message
        send_message(f"❌ *Order failed* — retcode: {err}\nSignal: {direction} @ {price:.2f}")
        return False

    logger.info("Order placed: %s %.2f lots @ %.2f | Ticket: %s",
                direction, lot_size, price, result.order)

    # Store open trade in shared state for tick_stream monitoring
    trade = {
        **signal,
        "ticket":   result.order,
        "entry":    price,
        "lot_size": lot_size,
        "tp1_hit":  False,
        "trail_sl": None,
    }
    shared_state.set_open_trade(trade)

    from execution.alert import send_execution_confirmation
    send_execution_confirmation(trade)
    return True


def close_trade(ticket: int, reason: str = "Manual close") -> bool:
    """Close an open position by ticket number."""
    if not MT5_AVAILABLE or not shared_state.get("mt5_connected"):
        return False

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning("No position found with ticket %s", ticket)
        return False

    pos  = positions[0]
    tick = mt5.symbol_info_tick(config.SYMBOL)
    close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       config.SYMBOL,
        "volume":       pos.volume,
        "type":         mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY
                        else mt5.ORDER_TYPE_BUY,
        "position":     ticket,
        "price":        close_price,
        "deviation":    10,
        "magic":        20240101,
        "comment":      f"XAUBOT close: {reason}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info("Position %s closed: %s", ticket, reason)
        shared_state.clear_open_trade()
        return True

    logger.error("Close failed: retcode=%s", result.retcode if result else "None")
    return False


def close_partial(ticket: int, volume: float, reason: str = "TP1") -> bool:
    """Close a portion of an open position (used for TP1 partial close)."""
    if not MT5_AVAILABLE:
        return False

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False

    pos  = positions[0]
    tick = mt5.symbol_info_tick(config.SYMBOL)
    close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

    volume = min(round(volume, 2), pos.volume)
    if volume <= 0:
        return False

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       config.SYMBOL,
        "volume":       volume,
        "type":         mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY
                        else mt5.ORDER_TYPE_BUY,
        "position":     ticket,
        "price":        close_price,
        "deviation":    10,
        "magic":        20240101,
        "comment":      f"XAUBOT partial: {reason}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info("Partial close %.2f lots: %s", volume, reason)
        return True

    logger.error("Partial close failed: %s", result.retcode if result else "None")
    return False


def modify_sl(ticket: int, new_sl: float) -> bool:
    """Modify stop loss of an open position (for breakeven / trail)."""
    if not MT5_AVAILABLE:
        return False

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl":       new_sl,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info("SL modified to %.2f for ticket %s", new_sl, ticket)
        return True

    logger.error("SL modify failed: %s", result.retcode if result else "None")
    return False


def _pre_trade_checks(signal: Dict) -> bool:
    """Final safety checks before execution."""
    # Daily loss limit
    from engine.risk_manager import check_daily_limit
    if not check_daily_limit():
        logger.warning("Pre-trade check failed: daily limit hit.")
        return False

    # Lot size sanity
    lot = signal.get("lot_size", 0)
    if lot < 0.01 or lot > 100:
        logger.warning("Pre-trade check failed: lot size %.2f out of range.", lot)
        return False

    # Spread check
    spread = shared_state.get("spread") or 0
    if spread > 5.0:   # $5 spread is extreme for gold
        logger.warning("Pre-trade check failed: spread too wide ($%.2f).", spread)
        from execution.alert import send_message
        send_message(f"⚠️ Trade skipped — spread too wide: ${spread:.2f}")
        return False

    return True
