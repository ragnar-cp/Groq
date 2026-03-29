# ============================================================
# telegram_bot/chat_handler.py — Free text → Gemini → reply
# ============================================================
import logging
from typing import Optional

import shared_state
import config

logger = logging.getLogger(__name__)


def handle_message(text: str, user_id: int) -> str:
    """
    Send a free-text message to Gemini with full live market context injected.
    Returns Gemini's response as a string.
    """
    try:
        context  = _build_system_context()
        response = _call_gemini(text, context)
        return response
    except Exception as exc:
        logger.error("Chat handler error: %s", exc)
        return f"⚠️ Sorry, I couldn't process that right now: {exc}"


def _call_gemini(user_message: str, system_context: str) -> str:
    from groq import Groq
    from engine.trade_memory import get_memory_context

    client = Groq(api_key=config.GROQ_API_KEY)

    memory_ctx = get_memory_context(n=10)

    system_prompt = (
        "You are an expert XAUUSD (Gold) trading assistant embedded in a live trading bot. "
        "You have access to real-time market data shown below. "
        "Answer the trader's question specifically, concisely, and actionably. "
        "Use plain English — no fluff. Reference actual numbers from the market state. "
        "If asked about commands, mention /help. "
        "If the market is consolidating, explain why you're not trading. "
        "Keep responses under 200 words.\n\n"
        f"=== LIVE MARKET STATE ===\n{system_context}\n\n"
        f"=== TRADE MEMORY (last 10 trades) ===\n{memory_ctx}"
    )

    try:
        shared_state.set("anthropic_ok", True)
        import time as _time
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    max_tokens=400,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = 20 * (attempt + 1)
                    logger.warning("Groq rate limit — waiting %ds", wait)
                    _time.sleep(wait)
                else:
                    raise
        return "⚠️ Groq rate limit hit. Please try again in a minute." 
    except Exception as exc:
        shared_state.set("anthropic_ok", False)
        raise exc


def _build_system_context() -> str:
    """Compile the full live market state into a readable string for Claude."""
    price   = shared_state.get("live_price")        or 0.0
    bid     = shared_state.get("bid")               or 0.0
    ask     = shared_state.get("ask")               or 0.0
    spread  = shared_state.get("spread")            or 0.0
    h4      = shared_state.get("h4_trend")          or "UNKNOWN"
    h1      = shared_state.get("h1_trend")          or "UNKNOWN"
    bias    = shared_state.get("bias_state")        or "UNKNOWN"
    side    = shared_state.get("allowed_side")      or "NONE"
    rsi     = shared_state.get("rsi_h1")            or 50.0
    macd    = shared_state.get("macd_h1")           or "NEUTRAL"
    atr     = shared_state.get("atr_h1")            or 0.0
    adx     = shared_state.get("adx_h1")            or 0.0
    ema20   = shared_state.get("ema20_h1")          or 0.0
    ema50   = shared_state.get("ema50_h1")          or 0.0
    ema200  = shared_state.get("ema200_h1")         or 0.0
    supp    = shared_state.get("key_support")       or 0.0
    res     = shared_state.get("key_resistance")    or 0.0
    d_hi    = shared_state.get("daily_high")        or 0.0
    d_lo    = shared_state.get("daily_low")         or 0.0
    ml_sc   = shared_state.get("ml_score")          or 0.0
    ml_dir  = shared_state.get("ml_direction")      or "NONE"
    sent    = shared_state.get("sentiment_score")   or "NEUTRAL"
    sent_sm = shared_state.get("sentiment_summary") or ""
    lockout = shared_state.get("in_news_lockout")   or False
    auto    = shared_state.get("auto_exec")         or False
    paused  = shared_state.get("bot_paused")        or False
    d_pause = shared_state.get("daily_paused")      or False
    d_pnl   = shared_state.get("daily_pnl")         or 0.0
    mode    = shared_state.get("mode")              or "live"
    losses  = shared_state.get("consecutive_losses")or 0

    trade     = shared_state.get("open_trade")
    trade_str = "None"
    if trade:
        trade_str = (
            f"{trade['direction']} entered at ${trade['entry']:.2f} | "
            f"SL: ${trade['sl']:.2f} | TP1: ${trade['tp1']:.2f} | TP2: ${trade['tp2']:.2f} | "
            f"TP1 hit: {trade.get('tp1_hit', False)}"
        )

    next_ev   = shared_state.get("next_news_event")
    news_str  = f"{next_ev['name']}" if next_ev else "None"

    headlines = shared_state.get("latest_headlines") or []
    hl_str    = " | ".join(h["title"] for h in headlines[:4]) or "None"

    sig       = shared_state.get("last_signal")
    sig_str   = "None"
    if sig:
        t = shared_state.get("last_signal_time")
        sig_str = (
            f"{sig['direction']} @ ${sig['entry']:.2f} | "
            f"Conf: {sig['confidence']}% | {t.strftime('%H:%M') if t else '?'}"
        )

    return (
        f"Bot mode: {mode.upper()}\n"
        f"Price: ${price:.2f} (Bid {bid:.2f} / Ask {ask:.2f} / Spread ${spread:.2f})\n"
        f"H4 Trend: {h4} | H1 Trend: {h1}\n"
        f"Bias state: {bias} → Allowed side: {side}\n"
        f"RSI(H1): {rsi:.1f} | MACD: {macd} | ADX: {adx:.1f} | ATR: ${atr:.2f}\n"
        f"EMA20: ${ema20:.2f} | EMA50: ${ema50:.2f} | EMA200: ${ema200:.2f}\n"
        f"Support: ${supp:.2f} | Resistance: ${res:.2f}\n"
        f"Daily High: ${d_hi:.2f} | Daily Low: ${d_lo:.2f}\n"
        f"ML Score: {ml_sc*100:.0f}% ({ml_dir})\n"
        f"News Sentiment: {sent} — {sent_sm}\n"
        f"Top Headlines: {hl_str}\n"
        f"Next High-Impact Event: {news_str}\n"
        f"News lockout active: {lockout}\n"
        f"Open trade: {trade_str}\n"
        f"Last signal: {sig_str}\n"
        f"Auto-execution: {auto}\n"
        f"Bot paused: {paused}\n"
        f"Daily paused (loss limit): {d_pause}\n"
        f"Daily P&L: {d_pnl:+.1f} pips\n"
        f"Consecutive losses: {losses}"
    )
