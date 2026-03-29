# ============================================================
# data/news_feed.py — Real-time gold news fetcher (thread)
# ============================================================
import time
import logging
import threading
from typing import List, Dict

import shared_state
import config

logger = logging.getLogger(__name__)

_stop_event   = threading.Event()
_seen_urls    = set()          # Avoid re-processing the same headline

NEWS_POLL_INTERVAL = 180       # seconds (3 min)

GOLD_KEYWORDS = [
    "gold", "xauusd", "federal reserve", "fed", "inflation", "cpi",
    "interest rate", "dollar", "dxy", "safe haven", "geopolit",
    "ukraine", "oil", "treasury", "jobs report", "nonfarm",
]


def start():
    shared_state.set_thread_status("news_watcher", "running")
    logger.info("News watcher started.")

    while not _stop_event.is_set():
        try:
            headlines = _fetch_all()
            if headlines:
                shared_state.set("latest_headlines", headlines[:15])
                _check_breaking(headlines)
            time.sleep(NEWS_POLL_INTERVAL)

        except Exception as exc:
            logger.error("News watcher error: %s", exc, exc_info=True)
            shared_state.set_thread_status("news_watcher", f"error: {exc}")
            time.sleep(30)

    shared_state.set_thread_status("news_watcher", "stopped")


def stop():
    _stop_event.set()


def _fetch_all() -> List[Dict]:
    headlines = []
    headlines += _fetch_newsapi()
    headlines += _fetch_rss()
    return headlines


def _fetch_newsapi() -> List[Dict]:
    """Fetch from NewsAPI — requires a free API key."""
    if not config.NEWS_API_KEY or config.NEWS_API_KEY == "your_newsapi_key":
        return []
    try:
        import requests
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        "gold OR XAUUSD OR 'Federal Reserve'",
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": 20,
                "apiKey":   config.NEWS_API_KEY,
            },
            timeout=10,
        )
        data = resp.json()
        items = []
        for art in data.get("articles", []):
            items.append({
                "title":     art.get("title", ""),
                "url":       art.get("url", ""),
                "source":    art.get("source", {}).get("name", ""),
                "published": art.get("publishedAt", ""),
            })
        return items
    except Exception as exc:
        logger.warning("NewsAPI fetch failed: %s", exc)
        return []


def _fetch_rss() -> List[Dict]:
    """Fetch from free gold-related RSS feeds."""
    try:
        import feedparser
        feeds = [
            "https://www.kitco.com/rss/kitconews.rss",
            "https://www.fxstreet.com/rss/news",
        ]
        items = []
        for url in feeds:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                title = entry.get("title", "")
                if _is_relevant(title):
                    items.append({
                        "title":     title,
                        "url":       entry.get("link", ""),
                        "source":    feed.feed.get("title", "RSS"),
                        "published": entry.get("published", ""),
                    })
        return items
    except Exception as exc:
        logger.warning("RSS fetch failed: %s", exc)
        return []


def _is_relevant(title: str) -> bool:
    tl = title.lower()
    return any(kw in tl for kw in GOLD_KEYWORDS)


def _check_breaking(headlines: List[Dict]):
    """Send Telegram alert for headlines we haven't seen before."""
    new_items = [h for h in headlines if h["url"] not in _seen_urls]
    for item in new_items[:3]:          # Max 3 new alerts per cycle
        _seen_urls.add(item["url"])
        logger.info("Breaking news: %s", item["title"])

    if new_items:
        from execution.alert import send_news_alert
        send_news_alert(new_items[:3])
