# ============================================================
# execution/alert.py — Telegram message formatter + sender
# ============================================================
import logging
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

_bot = None   # Lazy-initialised


def _get_bot():
    """Return a telegram Bot instance (initialised once)."""
    global _bot
    if _bot is None:
        try:
            from telegram import Bot
            _bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
        except Exception as exc:
            logger.error("Telegram Bot init failed: %s", exc)
    return _bot


def send_message(text: str, parse_mode: str = "Markdown"):
    """
    Send a plain message to all allowed Telegram users.
    Works from any thread — handles asyncio event loop safely.
    """
    import asyncio

    bot = _get_bot()
    if bot is None:
        logger.warning("Telegram not available — message not sent:\n%s", text)
        return

    async def _send():
        for uid in config.TELEGRAM_ALLOWED_UIDS:
            try:
                await bot.send_message(chat_id=uid, text=text, parse_mode=parse_mode)
            except Exception as exc:
                logger.warning("Telegram send failed to %s: %s", uid, exc)

    # If there's a running event loop (Telegram bot thread), schedule on it.
    # Otherwise create a new one (background threads like news_watcher).
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_send(), loop)
        else:
            loop.run_until_complete(_send())
    except RuntimeError:
        # No event loop at all — create a fresh one
        asyncio.run(_send())


# ── Signal alert ─────────────────────────────────────────────

def send_signal_alert(signal: Dict):
    direction = signal["direction"]
    emoji     = "🟢" if direction == "BUY" else "🔴"
    arrow     = "▲" if direction == "BUY" else "▼"

    # Confidence bar (10 chars)
    conf      = signal.get("confidence", 0)
    filled    = round(conf / 10)
    bar       = "█" * filled + "░" * (10 - filled)

    rule  = signal.get("rule_score", 0)
    ml    = signal.get("ml_score",   0)
    llm   = signal.get("llm_score",  0)
    bias  = signal.get("bias_state", "").replace("_", " ").title()
    auto  = "✅ YES" if config.MODE != "replay" else "🎭 Replay (no real orders)"

    msg = (
        f"{emoji} *XAUUSD {direction} SIGNAL* {arrow}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Entry:      `${signal['entry']:.2f}`\n"
        f"🛑 Stop Loss:  `${signal['sl']:.2f}`  _(ATR-based)_\n"
        f"🎯 TP1:        `${signal['tp1']:.2f}`  _(50% close)_\n"
        f"🎯 TP2:        `${signal['tp2']:.2f}`  _(trail rest)_\n"
        f"📐 Lot Size:   `{signal.get('lot_size', 0):.2f}`\n"
        f"⚠️ Risk:       `{signal.get('risk_pct', 0):.1f}%`\n\n"
        f"📊 *Confidence Breakdown:*\n"
        f"   Rule-based:  `{rule}%`  `{_bar(rule)}`\n"
        f"   ML Model:    `{ml}%`   `{_bar(ml)}`\n"
        f"   LLM \\(AI\\): `{llm}%`   `{_bar(llm)}`\n"
        f"   Combined:    `{conf}%`  `{bar}`\n"
        f"   Bias:        _{bias}_ ✅\n\n"
        f"📈 Setup:   _{signal.get('pattern', 'N/A')}_\n"
        f"⏰ Session: _{signal.get('session', 'N/A')}_\n"
        f"🧠 AI:      _{signal.get('llm_reasoning', '')}_\n\n"
        f"⚡ Auto-exec: {auto}\n"
        f"_Send /autoon to enable auto-execution_"
    )
    send_message(msg)
    logger.info("Signal alert sent: %s conf=%d%%", direction, conf)


# ── Trade lifecycle alerts ────────────────────────────────────

def send_execution_confirmation(trade: Dict):
    d = trade["direction"]
    send_message(
        f"⚡ *Trade Executed* — {d}\n"
        f"Entry: `${trade['entry']:.2f}` | "
        f"SL: `${trade['sl']:.2f}` | "
        f"TP1: `${trade['tp1']:.2f}` | "
        f"TP2: `${trade['tp2']:.2f}`\n"
        f"Lots: `{trade.get('lot_size',0):.2f}` | "
        f"Ticket: `{trade.get('ticket','N/A')}`"
    )


def send_tp1_alert(trade: Dict, price: float):
    pips = abs(price - trade["entry"]) / 0.1
    send_message(
        f"🎯 *TP1 Hit!*\n"
        f"Price: `${price:.2f}` | +`{pips:.0f}` pips\n"
        f"✅ 50% position closed — profit locked\n"
        f"🔒 Stop Loss moved to *breakeven* `${trade['entry']:.2f}`\n"
        f"📍 Trailing stop now active on remaining 50%"
    )


def send_tp2_alert(trade: Dict, price: float):
    total_pips = abs(price - trade["entry"]) / 0.1
    send_message(
        f"🏆 *TP2 Hit — Full Close!*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Exit: `${price:.2f}` | +`{total_pips:.0f}` pips\n"
        f"🎉 Trade closed in full profit!"
    )


def send_sl_alert(trade: Dict, price: float, reason: str):
    pips = abs(price - trade["entry"]) / 0.1
    won  = trade.get("tp1_hit", False)

    if won:
        send_message(
            f"🔒 *Trailing Stop Hit*\n"
            f"Exit: `${price:.2f}` | +`{pips:.0f}` pips\n"
            f"✅ Trade closed in profit after TP1 lock"
        )
    else:
        send_message(
            f"🛑 *Stop Loss Hit*\n"
            f"Exit: `${price:.2f}` | -`{pips:.0f}` pips\n"
            f"_{reason}_\n"
            f"_Bot continues scanning for next setup…_"
        )


def send_breakout_alert(level_type: str, price: float):
    emoji = "🚀" if level_type == "resistance" else "📉"
    send_message(
        f"{emoji} *Breakout Detected!*\n"
        f"Price crossed {level_type}: `${price:.2f}`"
    )


# ── News alert ───────────────────────────────────────────────

def send_news_alert(headlines: List[Dict]):
    if not headlines:
        return
    lines = ["📰 *Breaking News:*"]
    for h in headlines:
        lines.append(f"• _{h['title']}_ — {h.get('source','')}")
    send_message("\n".join(lines))


# ── Bias change alert ────────────────────────────────────────

def send_bias_change(old: str, new: str, h4: str, h1: str):
    emoji_map = {"FULL_BULL": "🟢", "FULL_BEAR": "🔴", "CONSOLIDATING": "🟡"}
    old_e = emoji_map.get(old, "⚪")
    new_e = emoji_map.get(new, "⚪")
    note  = {
        "FULL_BULL":     "M15 engine active — BUY setups only.",
        "FULL_BEAR":     "M15 engine active — SELL setups only.",
        "CONSOLIDATING": "H4/H1 conflict — standing down.",
    }.get(new, "")
    send_message(
        f"🔔 *Bias Change*\n"
        f"{old_e} {old} → {new_e} {new}\n"
        f"H4: {h4}  |  H1: {h1}\n"
        f"_{note}_"
    )


# ── Helpers ──────────────────────────────────────────────────

def _bar(pct: float, width: int = 8) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)
