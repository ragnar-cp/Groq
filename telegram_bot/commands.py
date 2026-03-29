# ============================================================
# telegram_bot/commands.py — All slash command handlers
# ============================================================
import logging
from datetime import datetime

import shared_state
import config

logger = logging.getLogger(__name__)


def handle_command(command: str, user_id: int) -> str:
    """
    Route a slash command to its handler.
    Returns the response string.
    """
    cmd = command.split()[0].lower().lstrip("/")

    handlers = {
        "start":    _cmd_start,
        "status":   _cmd_status,
        "analysis": _cmd_analysis,
        "news":     _cmd_news,
        "trades":   _cmd_trades,
        "memory":   _cmd_memory,
        "autoon":   _cmd_autoon,
        "autooff":  _cmd_autooff,
        "pause":    _cmd_pause,
        "resume":   _cmd_resume,
        "risk":     _cmd_risk,
        "health":   _cmd_health,
        "help":     _cmd_help,
    }

    handler = handlers.get(cmd, _cmd_unknown)
    try:
        return handler()
    except Exception as exc:
        logger.error("Command /%s error: %s", cmd, exc)
        return f"❌ Error handling /{cmd}: {exc}"


# ── Individual command handlers ───────────────────────────────

def _cmd_start() -> str:
    mode = shared_state.get("mode") or "live"
    return (
        "🤖 *XAUUSD AI Trading Bot*\n"
        f"Mode: `{mode.upper()}`\n\n"
        "I'm monitoring gold 24/5 using:\n"
        "• Real-time tick data\n"
        "• 3-layer AI (Rules + ML + Claude)\n"
        "• H4→H1→M15 bias cascade\n\n"
        "Use /help to see all commands.\n"
        "_Ask me anything about the market in plain English!_"
    )


def _cmd_status() -> str:
    price   = shared_state.get("live_price")  or 0.0
    bid     = shared_state.get("bid")         or 0.0
    ask     = shared_state.get("ask")         or 0.0
    spread  = shared_state.get("spread")      or 0.0
    h4      = shared_state.get("h4_trend")    or "UNKNOWN"
    h1      = shared_state.get("h1_trend")    or "UNKNOWN"
    bias    = shared_state.get("bias_state")  or "UNKNOWN"
    side    = shared_state.get("allowed_side")or "NONE"
    ml      = shared_state.get("ml_score")    or 0.0
    ml_dir  = shared_state.get("ml_direction")or "NONE"
    auto    = "✅ ON" if shared_state.get("auto_exec") else "❌ OFF"
    paused  = shared_state.get("bot_paused")
    d_pause = shared_state.get("daily_paused")
    d_pnl   = shared_state.get("daily_pnl")  or 0.0
    trade   = shared_state.get("open_trade")
    mode    = shared_state.get("mode")        or "live"
    replay  = shared_state.get("replay_progress") or ""
    lockout = shared_state.get("in_news_lockout")

    bias_emoji = {"FULL_BULL": "🟢", "FULL_BEAR": "🔴", "CONSOLIDATING": "🟡"}.get(bias, "⚪")
    status_emoji = "⏸️" if (paused or d_pause) else "🔄"

    trade_str = "None"
    if trade:
        trade_str = (
            f"{trade['direction']} @ ${trade['entry']:.2f} "
            f"| SL: ${trade['sl']:.2f} | TP2: ${trade['tp2']:.2f}"
        )

    next_news = shared_state.get("next_news_event")
    news_str  = f"{next_news['name']}" if next_news else "None upcoming"

    msg = (
        f"📊 *Bot Status* {status_emoji}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Mode:    `{mode.upper()}`"
        + (f" · _{replay}_" if mode == "replay" and replay else "") + "\n"
        f"Price:   `${price:.2f}` (Bid: {bid:.2f} / Ask: {ask:.2f})\n"
        f"Spread:  `${spread:.2f}`\n\n"
        f"H4 Trend: `{h4}`\n"
        f"H1 Trend: `{h1}`\n"
        f"Bias:     {bias_emoji} `{bias}` → *{side} only*\n\n"
        f"ML Score: `{ml*100:.0f}%` ({ml_dir})\n"
        f"Auto-exec: {auto}\n"
        f"Lockout:   {'🔒 YES' if lockout else '🔓 No'}\n\n"
        f"Open Trade: _{trade_str}_\n"
        f"Daily P&L:  `{d_pnl:+.1f} pips`\n"
        f"Next Event: _{news_str}_\n"
        + (f"\n⚠️ *Daily loss limit hit — paused until tomorrow*" if d_pause else "")
        + (f"\n⏸️ *Bot manually paused* — /resume to restart" if paused else "")
    )
    return msg


def _cmd_analysis() -> str:
    ind = {
        "rsi":    shared_state.get("rsi_h1")          or 0.0,
        "macd":   shared_state.get("macd_h1")         or "N/A",
        "atr":    shared_state.get("atr_h1")          or 0.0,
        "adx":    shared_state.get("adx_h1")          or 0.0,
        "ema20":  shared_state.get("ema20_h1")        or 0.0,
        "ema50":  shared_state.get("ema50_h1")        or 0.0,
        "ema200": shared_state.get("ema200_h1")       or 0.0,
        "supp":   shared_state.get("key_support")     or 0.0,
        "res":    shared_state.get("key_resistance")  or 0.0,
        "d_hi":   shared_state.get("daily_high")      or 0.0,
        "d_lo":   shared_state.get("daily_low")       or 0.0,
        "bb_pos": shared_state.get("bb_position")     or 0.5,
    }
    price = shared_state.get("live_price") or 0.0
    sig   = shared_state.get("last_signal")
    sig_str = "None yet"
    if sig:
        t = shared_state.get("last_signal_time")
        t_str = t.strftime("%H:%M") if t else "?"
        sig_str = f"{sig['direction']} @ ${sig['entry']:.2f} (conf: {sig['confidence']}%) at {t_str}"

    rsi_note = "Oversold 🔵" if ind["rsi"] < 35 else ("Overbought 🔴" if ind["rsi"] > 65 else "Neutral ✅")
    adx_note = "Strong trend 💪" if ind["adx"] > 25 else "Weak/ranging 😴"
    bb_note  = "Upper zone" if ind["bb_pos"] > 0.8 else ("Lower zone" if ind["bb_pos"] < 0.2 else "Mid zone")

    return (
        f"📈 *Full Market Analysis*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Price:  `${price:.2f}`\n"
        f"EMA20:  `${ind['ema20']:.2f}`  EMA50: `${ind['ema50']:.2f}`  EMA200: `${ind['ema200']:.2f}`\n\n"
        f"RSI(H1):  `{ind['rsi']:.1f}` — _{rsi_note}_\n"
        f"MACD(H1): `{ind['macd']}`\n"
        f"ADX:      `{ind['adx']:.1f}` — _{adx_note}_\n"
        f"ATR:      `${ind['atr']:.2f}`\n"
        f"BB Pos:   `{ind['bb_pos']:.2f}` — _{bb_note}_\n\n"
        f"Support:    `${ind['supp']:.2f}`\n"
        f"Resistance: `${ind['res']:.2f}`\n"
        f"Daily High: `${ind['d_hi']:.2f}`\n"
        f"Daily Low:  `${ind['d_lo']:.2f}`\n\n"
        f"Last Signal: _{sig_str}_"
    )


def _cmd_news() -> str:
    headlines  = shared_state.get("latest_headlines") or []
    sentiment  = shared_state.get("sentiment_score")  or "NEUTRAL"
    summary    = shared_state.get("sentiment_summary")or "No summary yet"
    next_event = shared_state.get("next_news_event")
    lockout    = shared_state.get("in_news_lockout")

    s_emoji = {
        "STRONG_BULLISH": "🟢🟢",
        "BULLISH":        "🟢",
        "NEUTRAL":        "⚪",
        "BEARISH":        "🔴",
        "STRONG_BEARISH": "🔴🔴",
    }.get(sentiment, "⚪")

    lines = [
        f"📰 *News & Sentiment*",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"Sentiment: {s_emoji} `{sentiment}`",
        f"_{summary}_\n",
        f"*Latest Headlines:*",
    ]
    for h in headlines[:5]:
        lines.append(f"• {h['title']}")

    if next_event:
        t = next_event["time"]
        t_str = t.strftime("%H:%M UTC") if hasattr(t, "strftime") else str(t)
        lines.append(f"\n📅 Next Event: *{next_event['name']}* at `{t_str}`")

    if lockout:
        lines.append(f"\n🔒 *News lockout currently active*")

    return "\n".join(lines)


def _cmd_trades() -> str:
    from engine.trade_memory import _load
    memory = _load()
    if not memory:
        return "📋 No trades recorded yet."

    today_str = datetime.now().strftime("%Y-%m-%d")
    today     = [t for t in memory if t.get("timestamp","").startswith(today_str)]

    if not today:
        return "📋 No trades today yet.\n_Use /memory for full history._"

    lines = [f"📋 *Today's Trades* ({len(today)}):", "━━━━━━━━━━━━━━━━━━"]
    total_pips = 0.0
    for t in today:
        sign = "✅" if t["result"] == "WIN" else "❌"
        pips = t.get("pnl_pips", 0)
        total_pips += pips
        lines.append(
            f"{sign} {t['direction']} | {t.get('pattern','?')} | "
            f"`{pips:+.0f}` pips | _{t.get('exit_reason','?')}_"
        )
    lines.append(f"\n💰 *Total today: `{total_pips:+.1f}` pips*")
    return "\n".join(lines)


def _cmd_memory() -> str:
    from engine.trade_memory import get_stats
    stats = get_stats()
    if not stats or stats.get("total", 0) == 0:
        return "🧠 No trade memory yet. Trades will be recorded after first signals."

    lines = [
        "🧠 *Trade Memory Stats*",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Total trades: `{stats['total']}`",
        f"Win rate:     `{stats['win_rate']}%` ({stats['wins']}W / {stats['losses']}L)",
        f"Last 10 WR:   `{stats.get('recent_10_wr','N/A')}`\n",
        f"*By Session:*",
    ]
    for sess, wr in (stats.get("session_wr") or {}).items():
        lines.append(f"  {sess}: `{wr}`")

    lines += [
        f"\n✅ Best setup:  _{stats.get('best_pattern','N/A')}_",
        f"❌ Worst setup: _{stats.get('worst_pattern','N/A')}_",
    ]
    return "\n".join(lines)


def _cmd_autoon() -> str:
    if config.MODE == "replay":
        return "🎭 *Replay mode* — auto-execution disabled (no real orders placed)."
    shared_state.set("auto_exec", True)
    threshold = config.AUTOEXEC_CONFIDENCE_MIN
    return (
        f"✅ *Auto-execution ENABLED*\n"
        f"Trades will be placed automatically when confidence ≥ `{threshold}%`\n"
        f"_Send /autooff to disable_"
    )


def _cmd_autooff() -> str:
    shared_state.set("auto_exec", False)
    return "❌ *Auto-execution DISABLED*\n_You will receive alerts only. Send /autoon to re-enable._"


def _cmd_pause() -> str:
    shared_state.set("bot_paused", True)
    return "⏸️ *Bot paused* — no new signals will be generated.\n_Send /resume to restart scanning._"


def _cmd_resume() -> str:
    shared_state.set("bot_paused", False)
    shared_state.set("daily_paused", False)
    return "▶️ *Bot resumed* — scanning for setups."


def _cmd_risk() -> str:
    import config as cfg
    cons_losses = shared_state.get("consecutive_losses") or 0
    atr_ratio   = shared_state.get("atr_ratio") or 1.0

    loss_mod = cfg.SIZE_MODIFIER_LOSS_STREAK if cons_losses >= 2 else 1.0
    vol_mod  = cfg.SIZE_MODIFIER_HIGH_VOLATILITY if atr_ratio > 1.5 else 1.0
    eff_risk = cfg.BASE_RISK_PERCENT * loss_mod * vol_mod

    return (
        f"⚙️ *Risk Settings*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Base risk:       `{cfg.BASE_RISK_PERCENT}%` per trade\n"
        f"Effective risk:  `{eff_risk:.2f}%` (after modifiers)\n"
        f"Daily limit:     `{cfg.MAX_DAILY_LOSS_PCT}%` drawdown\n"
        f"ATR SL mult:     `{cfg.ATR_SL_MULTIPLIER}×`\n"
        f"Trailing mult:   `{cfg.ATR_TRAIL_MULTIPLIER}×`\n"
        f"TP1 RR:          `1:{cfg.TP1_RR}`\n"
        f"TP2 RR:          `1:{cfg.TP2_RR}`\n\n"
        f"Consecutive losses: `{cons_losses}`\n"
        f"ATR ratio:          `{atr_ratio:.2f}` "
        + ("_(high vol — size reduced)_" if atr_ratio > 1.5 else "_(normal)_") + "\n"
        f"Loss streak mod:    `{loss_mod:.1f}×`\n"
        f"Volatility mod:     `{vol_mod:.1f}×`"
    )


def _cmd_health() -> str:
    threads = shared_state.get("thread_status") or {}
    last_tick = shared_state.get("last_tick_received")
    mt5_ok    = shared_state.get("mt5_connected")
    api_ok    = shared_state.get("anthropic_ok")
    mode      = shared_state.get("mode") or "live"

    now = datetime.now()
    if last_tick:
        secs_ago = (now - last_tick).total_seconds()
        tick_str = f"`{secs_ago:.0f}s ago`"
        tick_ok  = secs_ago < config.HEALTH_TICK_STALE_SECONDS
        tick_emoji = "✅" if tick_ok else "⚠️"
    else:
        tick_str  = "`never`"
        tick_emoji= "⚠️"

    def t_emoji(status: str) -> str:
        if "running" in status:      return "✅"
        if "stopped" in status:      return "🔴"
        if "complete" in status:     return "✅"
        return "⚠️"

    lines = [
        f"🏥 *Bot Health Check*",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"Mode: `{mode.upper()}`",
        f"MT5 Connected:  {'✅' if mt5_ok else '🔴'}",
        f"Anthropic API:  {'✅' if api_ok else '🔴'}",
        f"Last Tick:      {tick_emoji} {tick_str}",
        f"\n*Threads:*",
    ]
    for name, status in threads.items():
        lines.append(f"  {t_emoji(status)} `{name}`: _{status}_")

    lines.append(f"\n_Checked at {now.strftime('%H:%M:%S')}_")
    return "\n".join(lines)


def _cmd_help() -> str:
    return (
        "🤖 *XAUUSD Bot Commands*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "/status   — Price, bias, ML score, open trade\n"
        "/analysis — Full TA breakdown (RSI, MACD, ATR…)\n"
        "/news     — Latest headlines + sentiment\n"
        "/trades   — Today's trade results\n"
        "/memory   — Win rate stats by setup & session\n"
        "/autoon   — Enable auto-execution\n"
        "/autooff  — Disable auto-execution\n"
        "/pause    — Pause bot scanning\n"
        "/resume   — Resume bot scanning\n"
        "/risk     — View risk settings & modifiers\n"
        "/health   — Thread status & connection check\n\n"
        "_You can also just talk to me in plain English!_\n"
        "_Example: 'What's gold doing?' or 'Why no signals?'_"
    )


def _cmd_unknown() -> str:
    return "❓ Unknown command. Use /help to see all available commands."
