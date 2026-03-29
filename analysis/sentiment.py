# ============================================================
# analysis/sentiment.py — News headlines → Claude → sentiment score
# ============================================================
import logging
import json
from typing import Optional

import shared_state
import config

logger = logging.getLogger(__name__)


def get_sentiment_score() -> str:
    """
    Send latest headlines to Claude and get a sentiment score.
    Returns: "STRONG_BULLISH" | "BULLISH" | "NEUTRAL" | "BEARISH" | "STRONG_BEARISH"
    Updates shared_state sentiment fields.
    Cached — only re-runs when headlines change.
    """
    headlines = shared_state.get("latest_headlines") or []
    if not headlines:
        return "NEUTRAL"

    try:
        summary, score = _call_groq_sentiment(headlines)
        shared_state.set("sentiment_score",   score)
        shared_state.set("sentiment_summary", summary)
        logger.info("Sentiment: %s — %s", score, summary[:80])
        return score

    except Exception as exc:
        logger.warning("Sentiment analysis failed: %s", exc)
        return shared_state.get("sentiment_score") or "NEUTRAL"


def _call_groq_sentiment(headlines: list) -> tuple:
    """Call Groq API to score market sentiment from headlines."""
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)

    headline_text = "\n".join(
        f"- {h['title']} ({h.get('source','')})"
        for h in headlines[:10]
    )

    prompt = f"""You are an expert gold (XAUUSD) market analyst.

Analyze these recent news headlines and rate the overall sentiment for gold price direction.

Headlines:
{headline_text}

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{{
  "score": "STRONG_BULLISH" | "BULLISH" | "NEUTRAL" | "BEARISH" | "STRONG_BEARISH",
  "summary": "One sentence explanation max 20 words",
  "key_factors": ["factor1", "factor2"]
}}

Consider: Fed policy (hawkish=bearish for gold), geopolitical risk (bullish),
USD strength (bearish for gold), inflation (bullish), risk-off sentiment (bullish).
"""

    import time as _time
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 20 * (attempt + 1)
                logger.warning("Groq rate limit — waiting %ds (attempt %d/3)", wait, attempt+1)
                _time.sleep(wait)
            else:
                raise
    else:
        logger.error("Groq rate limit — all retries failed. Using cached sentiment.")
        return "", "NEUTRAL"

    raw  = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    return data.get("summary", ""), data.get("score", "NEUTRAL")
