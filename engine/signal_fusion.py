# ============================================================
# engine/signal_fusion.py — 3-layer signal combiner
# ============================================================
import logging
import json
from typing import Optional, Dict

import shared_state
import config

logger = logging.getLogger(__name__)


def fuse_signals(
    indicators:     Dict,
    candle_pattern: Optional[Dict],
    chart_pattern:  Optional[Dict],
    ml_score:       float,
    ml_direction:   str,
    sentiment:      str,
) -> Optional[Dict]:
    """
    Combine rule-based, ML, and LLM layers into one final signal.
    Returns a signal dict if confidence >= ALERT_CONFIDENCE_MIN, else None.
    """
    allowed_side = shared_state.get("allowed_side")
    if allowed_side == "NONE":
        return None

    # ── Layer 1: Rule-based score ──────────────────────────
    rule_score, rule_direction = _rule_based_score(
        indicators, candle_pattern, chart_pattern
    )

    # ── Layer 2: ML score ──────────────────────────────────
    # ml_score is already a 0-100 probability for ml_direction
    ml_score_pct = ml_score * 100

    # ── Layer 3: LLM score ─────────────────────────────────
    llm_score, llm_direction, llm_reasoning = _llm_score(indicators, sentiment)

    # ── Determine agreed direction ─────────────────────────
    votes = {
        "BUY":  0,
        "SELL": 0,
    }
    if rule_direction in votes: votes[rule_direction] += rule_score * config.WEIGHT_RULE_BASED
    if ml_direction   in votes: votes[ml_direction]   += ml_score_pct * config.WEIGHT_ML
    if llm_direction  in votes: votes[llm_direction]  += llm_score * config.WEIGHT_LLM

    if votes["BUY"] > votes["SELL"]:
        direction     = "BUY"
        confidence    = votes["BUY"]
    elif votes["SELL"] > votes["BUY"]:
        direction     = "SELL"
        confidence    = votes["SELL"]
    else:
        return None   # Tie — no trade

    # ── Bias gate — hard filter ────────────────────────────
    if direction != allowed_side:
        logger.debug("Signal %s blocked by bias gate (allowed: %s)", direction, allowed_side)
        return None

    confidence = min(round(confidence), 100)

    if confidence < config.ALERT_CONFIDENCE_MIN:
        return None

    # ── Build signal dict ──────────────────────────────────
    price    = shared_state.get("live_price") or 0.0
    atr      = indicators.get("atr_h1", 10)
    sl_dist  = atr * config.ATR_SL_MULTIPLIER
    tp1_dist = sl_dist * config.TP1_RR
    tp2_dist = sl_dist * config.TP2_RR

    if direction == "BUY":
        sl  = price - sl_dist
        tp1 = price + tp1_dist
        tp2 = price + tp2_dist
    else:
        sl  = price + sl_dist
        tp1 = price - tp1_dist
        tp2 = price - tp2_dist

    pattern_name = (candle_pattern or {}).get("pattern") or \
                   (chart_pattern  or {}).get("pattern") or "No pattern"

    signal = {
        "direction":       direction,
        "entry":           round(price, 2),
        "sl":              round(sl,    2),
        "tp1":             round(tp1,   2),
        "tp2":             round(tp2,   2),
        "atr":             round(atr,   2),
        "confidence":      confidence,
        "rule_score":      round(rule_score),
        "ml_score":        round(ml_score_pct),
        "llm_score":       round(llm_score),
        "llm_reasoning":   llm_reasoning,
        "bias_state":      shared_state.get("bias_state"),
        "pattern":         pattern_name,
        "session":         _current_session(),
        "tp1_hit":         False,
        "trail_sl":        None,
    }

    logger.info(
        "Signal fused: %s @ %.2f | Conf: %d%% (R:%d ML:%d LLM:%d)",
        direction, price, confidence,
        round(rule_score), round(ml_score_pct), round(llm_score),
    )
    return signal


# ── Rule-based scoring ────────────────────────────────────────

def _rule_based_score(
    ind: Dict,
    candle: Optional[Dict],
    chart:  Optional[Dict],
) -> tuple:
    """
    Score the technical setup 0-100 and determine direction.
    Returns (score, direction).
    """
    buy_score  = 0
    sell_score = 0

    rsi    = ind.get("rsi_h1",    50)
    macd   = ind.get("macd_h1",   "NEUTRAL")
    adx    = ind.get("adx_h1",    25)
    atr_r  = ind.get("atr_ratio", 1.0)
    price  = shared_state.get("live_price") or 0.0
    e20    = ind.get("ema20_h1",  price)
    e50    = ind.get("ema50_h1",  price)
    e200   = ind.get("ema200_h1", price)
    supp   = ind.get("key_support",    0)
    res    = ind.get("key_resistance", 999999)

    # Trend alignment (+25)
    if price > e200 and e20 > e50:
        buy_score  += 25
    elif price < e200 and e20 < e50:
        sell_score += 25

    # RSI not extreme (+15)
    if 30 < rsi < 60:
        buy_score  += 15
    elif 40 < rsi < 70:
        sell_score += 15
    if rsi < 35:
        buy_score  += 10   # Oversold bonus
    if rsi > 65:
        sell_score += 10   # Overbought bonus

    # MACD (+15)
    if macd == "BULLISH":
        buy_score  += 15
    elif macd == "BEARISH":
        sell_score += 15

    # ADX trend strength (+10)
    if adx > 25:
        buy_score  += 5
        sell_score += 5   # Strong trend in either direction is good

    # Candle pattern (+20)
    if candle:
        if candle["direction"] == "BUY":
            buy_score  += candle["strength"] * 0.2
        elif candle["direction"] == "SELL":
            sell_score += candle["strength"] * 0.2

    # Chart pattern (+15)
    if chart:
        if chart["direction"] == "BUY":
            buy_score  += chart["strength"] * 0.15
        elif chart["direction"] == "SELL":
            sell_score += chart["strength"] * 0.15

    # Near S&R (+10)
    tolerance = ind.get("atr_h1", 10) * 0.5
    if price and abs(price - supp) < tolerance:
        buy_score  += 10
    if price and abs(price - res) < tolerance:
        sell_score += 10

    buy_score  = min(buy_score,  100)
    sell_score = min(sell_score, 100)

    if buy_score > sell_score:
        return buy_score, "BUY"
    elif sell_score > buy_score:
        return sell_score, "SELL"
    return 50, "NONE"


# ── LLM scoring ───────────────────────────────────────────────

def _llm_score(ind: Dict, sentiment: str) -> tuple:
    """
    Send full market context to Gemini and get a score + direction + reasoning.
    Returns (score 0-100, direction BUY/SELL/WAIT, reasoning str)
    """
    try:
        from groq import Groq
        from engine.trade_memory import get_memory_context

        client = Groq(api_key=config.GROQ_API_KEY)

        context = _build_market_context(ind, sentiment)
        memory  = get_memory_context()

        prompt = f"""You are an expert XAUUSD trader making a real-time trade decision.

CURRENT MARKET STATE:
{context}

RECENT TRADE MEMORY (last 10 similar setups):
{memory}

Based on all available information, provide your trading decision.

Respond ONLY with this JSON (no markdown, no extra text):
{{
  "direction": "BUY" | "SELL" | "WAIT",
  "confidence": 0-100,
  "reasoning": "max 25 words explaining your decision"
}}
"""
        import time as _time
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.1,
                )
                break
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = 20 * (attempt + 1)
                    logger.warning("Groq rate limit — waiting %ds", wait)
                    _time.sleep(wait)
                else:
                    raise
        else:
            return 50, "NONE", "Rate limit — skipped"
        raw       = response.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        data      = json.loads(raw)
        direction = data.get("direction", "WAIT")
        score     = float(data.get("confidence", 50))
        reasoning = data.get("reasoning", "")

        if direction == "WAIT":
            return 40, "NONE", reasoning

        return score, direction, reasoning

    except Exception as exc:
        logger.warning("Gemini LLM score error: %s", exc)
        return 50, "NONE", "LLM unavailable"


def _build_market_context(ind: Dict, sentiment: str) -> str:
    h4    = shared_state.get("h4_trend")     or "UNKNOWN"
    h1    = shared_state.get("h1_trend")     or "UNKNOWN"
    bias  = shared_state.get("bias_state")   or "UNKNOWN"
    price = shared_state.get("live_price")   or 0.0
    news  = shared_state.get("next_news_event")
    news_str = f"{news['name']} at {news['time']}" if news else "None upcoming"
    headlines = shared_state.get("latest_headlines") or []
    top_headlines = "; ".join(h["title"] for h in headlines[:3])

    return (
        f"Price: ${price:.2f}\n"
        f"H4 Trend: {h4}  H1 Trend: {h1}  Bias: {bias}\n"
        f"RSI(H1): {ind.get('rsi_h1',50):.1f}\n"
        f"MACD(H1): {ind.get('macd_h1','NEUTRAL')}\n"
        f"ADX: {ind.get('adx_h1',25):.1f}\n"
        f"ATR: {ind.get('atr_h1',10):.2f}\n"
        f"Support: ${ind.get('key_support',0):.2f}  "
        f"Resistance: ${ind.get('key_resistance',0):.2f}\n"
        f"News Sentiment: {sentiment}\n"
        f"Next High-Impact Event: {news_str}\n"
        f"Top Headlines: {top_headlines or 'None'}"
    )


def _current_session() -> str:
    from datetime import datetime, timezone
    hour = datetime.now(timezone.utc).hour
    if  0 <= hour < 8:  return "Asian"
    if  8 <= hour < 12: return "London"
    if 12 <= hour < 16: return "Overlap"
    if 16 <= hour < 21: return "New York"
    return "Close"
