# ============================================================
# engine/risk_manager.py — Dynamic position sizing + drawdown guard
# ============================================================
import logging
from typing import Dict

import shared_state
import config

logger = logging.getLogger(__name__)

# XAUUSD pip value per 0.01 lot (varies by broker, typical value)
PIP_VALUE_PER_LOT = 1.0      # USD per pip per 0.01 lot for XAUUSD


def calculate_position(signal: Dict) -> Dict:
    """
    Attach lot_size and final risk metadata to a signal dict.
    Uses dynamic sizing based on confidence + market conditions.
    """
    account = _get_account_balance()
    balance = account if account else 10_000   # Fallback for replay mode

    sl_pips = abs(signal["entry"] - signal["sl"]) / 0.1
    if sl_pips <= 0:
        sl_pips = 50   # Fallback

    # ── Base risk ─────────────────────────────────────────
    base_risk_usd = balance * (config.BASE_RISK_PERCENT / 100)

    # ── Confidence modifier ───────────────────────────────
    conf = signal.get("confidence", 70)
    if conf >= config.CONFIDENCE_HIGH_THRESHOLD:
        conf_mod = config.SIZE_MODIFIER_HIGH
    elif conf >= config.CONFIDENCE_LOW_THRESHOLD:
        conf_mod = config.SIZE_MODIFIER_NORMAL
    else:
        conf_mod = config.SIZE_MODIFIER_LOW

    # ── Consecutive loss modifier ─────────────────────────
    losses = shared_state.get("consecutive_losses") or 0
    loss_mod = config.SIZE_MODIFIER_LOSS_STREAK if losses >= 2 else 1.0

    # ── Volatility modifier ───────────────────────────────
    atr_ratio = shared_state.get("atr_ratio") or 1.0
    vol_mod   = config.SIZE_MODIFIER_HIGH_VOLATILITY if atr_ratio > 1.5 else 1.0

    # ── Combined modifier ─────────────────────────────────
    total_mod = conf_mod * loss_mod * vol_mod
    risk_usd  = base_risk_usd * total_mod

    # ── Lot size calculation ──────────────────────────────
    # lot_size = risk_usd / (sl_pips × pip_value_per_lot × 100)
    lot_size = risk_usd / (sl_pips * PIP_VALUE_PER_LOT * 100)
    lot_size = max(0.01, round(lot_size, 2))

    signal["lot_size"]    = lot_size
    signal["risk_usd"]    = round(risk_usd, 2)
    signal["risk_pct"]    = round(config.BASE_RISK_PERCENT * total_mod, 2)
    signal["conf_mod"]    = round(conf_mod, 2)
    signal["loss_mod"]    = round(loss_mod, 2)
    signal["vol_mod"]     = round(vol_mod,  2)

    logger.info(
        "Position sized: %.2f lots | Risk: $%.2f (%.1f%%) | Mods: conf=%.1f loss=%.1f vol=%.1f",
        lot_size, risk_usd, signal["risk_pct"], conf_mod, loss_mod, vol_mod,
    )
    return signal


def check_daily_limit() -> bool:
    """Return True if we're within daily loss limit (OK to trade)."""
    if shared_state.get("daily_paused"):
        return False

    account = _get_account_balance()
    if not account:
        return True   # Can't check, allow

    daily_pnl = shared_state.get("daily_pnl") or 0.0
    if daily_pnl < 0:
        loss_pct = abs(daily_pnl) / account * 100
        if loss_pct >= config.MAX_DAILY_LOSS_PCT:
            shared_state.set("daily_paused", True)
            logger.warning("Daily loss limit hit (%.1f%%) — pausing.", loss_pct)
            return False
    return True


def reset_daily_stats():
    """Called at start of new trading day."""
    shared_state.set("daily_pnl",    0.0)
    shared_state.set("daily_pnl_pct", 0.0)
    shared_state.set("daily_paused", False)
    logger.info("Daily stats reset for new trading day.")


def _get_account_balance() -> float:
    try:
        from data.mt5_feed import get_account_info
        info = get_account_info()
        return info["balance"] if info else 0.0
    except Exception:
        return 0.0
