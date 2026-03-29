# ============================================================
# data/economic_calendar.py — High-impact event watcher
# ============================================================
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import shared_state
import config

logger = logging.getLogger(__name__)

_stop_event = threading.Event()
_events_cache: List[Dict] = []
_cache_updated_at: Optional[datetime] = None
CACHE_TTL_MINUTES = 60


def start():
    """Run as part of news_watcher thread — called periodically."""
    while not _stop_event.is_set():
        try:
            _refresh_calendar()
            _update_lockout_state()
        except Exception as exc:
            logger.warning("Calendar update error: %s", exc)
        time.sleep(300)   # Refresh every 5 min


def stop():
    _stop_event.set()


def get_upcoming_events(hours_ahead: int = 4) -> List[Dict]:
    """Return high-impact events in the next N hours."""
    now = datetime.utcnow()
    cutoff = now + timedelta(hours=hours_ahead)
    return [e for e in _events_cache if now <= e["time"] <= cutoff]


def _refresh_calendar():
    global _events_cache, _cache_updated_at
    events = _scrape_forexfactory()
    if events:
        _events_cache      = events
        _cache_updated_at  = datetime.utcnow()
        logger.info("Calendar refreshed — %d events loaded.", len(events))

    # Always find the next upcoming event and store it
    upcoming = get_upcoming_events(hours_ahead=8)
    if upcoming:
        next_event = min(upcoming, key=lambda e: e["time"])
        shared_state.set("next_news_event", next_event)
    else:
        shared_state.set("next_news_event", None)


def _update_lockout_state():
    """Set in_news_lockout flag based on upcoming events."""
    now = datetime.utcnow()
    lockout_before = timedelta(minutes=config.NEWS_LOCKOUT_MINUTES_BEFORE)
    lockout_after  = timedelta(minutes=config.NEWS_LOCKOUT_MINUTES_AFTER)

    in_lockout = False
    for event in _events_cache:
        event_time = event["time"]
        if (event_time - lockout_before) <= now <= (event_time + lockout_after):
            in_lockout = True
            logger.info("News lockout active: %s at %s", event["name"], event_time)
            break

    prev = shared_state.get("in_news_lockout")
    shared_state.set("in_news_lockout", in_lockout)

    if in_lockout and not prev:
        try:
            from execution.alert import send_message
            next_ev = shared_state.get("next_news_event")
            name    = next_ev["name"] if next_ev else "high-impact event"
            send_message(f"🔒 *News lockout active* — {name}\nNo trades until {config.NEWS_LOCKOUT_MINUTES_AFTER} min after release.")
        except Exception:
            pass

    if not in_lockout and prev:
        try:
            from execution.alert import send_message
            send_message("🔓 *News lockout lifted* — Resuming market scan.")
        except Exception:
            pass


def _scrape_forexfactory() -> List[Dict]:
    """
    Scrape ForexFactory calendar for USD + Gold high-impact events today.
    Falls back to empty list if scraping fails (site structure may change).
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://www.forexfactory.com/calendar"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        events  = []
        today   = datetime.utcnow().date()

        for row in soup.select("tr.calendar__row"):
            impact = row.select_one(".calendar__impact span")
            if not impact:
                continue
            impact_class = impact.get("class", [])
            if not any("red" in c for c in impact_class):
                continue   # Only high-impact (red) events

            currency = row.select_one(".calendar__currency")
            if not currency or currency.text.strip() not in ("USD", "XAU"):
                continue

            name_el = row.select_one(".calendar__event-title")
            time_el = row.select_one(".calendar__time")
            if not name_el or not time_el:
                continue

            time_str = time_el.text.strip()
            try:
                event_time = datetime.strptime(
                    f"{today} {time_str}", "%Y-%m-%d %I:%M%p"
                )
            except ValueError:
                continue

            events.append({
                "name":     name_el.text.strip(),
                "currency": currency.text.strip(),
                "time":     event_time,
                "impact":   "HIGH",
            })

        return events

    except Exception as exc:
        logger.warning("ForexFactory scrape failed: %s", exc)
        return []
