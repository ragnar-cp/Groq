# ============================================================
# engine/trade_memory.py — Persistent trade memory system
# ============================================================
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

MEMORY_FILE = config.TRADE_MEMORY_FILE
MAX_MEMORY  = 200   # Keep last N trades


def record_trade(
    signal:      Dict,
    result:      str,       # "WIN" | "LOSS"
    exit_price:  float,
    exit_reason: str,
    pnl_pips:    float,
    lesson:      Optional[str] = None,
):
    """Write a completed trade to trade_memory.json."""
    try:
        memory = _load()

        entry = {
            "id":           f"TM_{len(memory)+1:04d}",
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "setup":        _describe_setup(signal),
            "direction":    signal.get("direction"),
            "bias_state":   signal.get("bias_state"),
            "session":      signal.get("session"),
            "entry":        signal.get("entry"),
            "exit_price":   round(exit_price, 2),
            "sl":           signal.get("sl"),
            "tp1":          signal.get("tp1"),
            "tp2":          signal.get("tp2"),
            "confidence":   signal.get("confidence"),
            "rule_score":   signal.get("rule_score"),
            "ml_score":     signal.get("ml_score"),
            "llm_score":    signal.get("llm_score"),
            "pattern":      signal.get("pattern"),
            "result":       result,
            "pnl_pips":     round(pnl_pips, 1),
            "exit_reason":  exit_reason,
            "lesson":       lesson,
        }
        memory.append(entry)

        # Trim to max
        if len(memory) > MAX_MEMORY:
            memory = memory[-MAX_MEMORY:]

        _save(memory)
        logger.info("Trade memory updated: %s %s (%.1f pips)", result, exit_reason, pnl_pips)

    except Exception as exc:
        logger.error("Failed to record trade memory: %s", exc)


def get_memory_context(n: int = 20) -> str:
    """
    Return last N trades as a formatted string for LLM context injection.
    """
    try:
        memory = _load()
        recent = memory[-n:]
        if not recent:
            return "No trade history yet."

        wins   = sum(1 for t in recent if t["result"] == "WIN")
        losses = len(recent) - wins
        wr     = wins / len(recent) * 100 if recent else 0

        lines = [f"Last {len(recent)} trades: {wins}W/{losses}L ({wr:.0f}% WR)"]
        for t in recent[-5:]:   # Last 5 in detail
            sign = "✅" if t["result"] == "WIN" else "❌"
            lines.append(
                f"{sign} {t['direction']} | {t['setup']} | "
                f"{t['pnl_pips']:+.0f} pips | {t.get('lesson','')}"
            )
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("get_memory_context error: %s", exc)
        return "Memory unavailable."


def get_stats() -> Dict:
    """Return win rate stats broken down by setup, session, etc."""
    try:
        memory = _load()
        if not memory:
            return {"total": 0}

        total  = len(memory)
        wins   = sum(1 for t in memory if t["result"] == "WIN")
        wr     = wins / total * 100 if total else 0

        # By session
        sessions = {}
        for t in memory:
            s  = t.get("session", "Unknown")
            sessions.setdefault(s, {"wins": 0, "total": 0})
            sessions[s]["total"] += 1
            if t["result"] == "WIN":
                sessions[s]["wins"] += 1

        session_wr = {
            s: f"{v['wins']}/{v['total']} ({v['wins']/v['total']*100:.0f}%)"
            for s, v in sessions.items()
        }

        # By pattern
        patterns = {}
        for t in memory:
            p = t.get("pattern", "Unknown")
            patterns.setdefault(p, {"wins": 0, "total": 0})
            patterns[p]["total"] += 1
            if t["result"] == "WIN":
                patterns[p]["wins"] += 1

        best_pattern  = max(patterns, key=lambda p: patterns[p]["wins"] / patterns[p]["total"])
        worst_pattern = min(patterns, key=lambda p: patterns[p]["wins"] / patterns[p]["total"])

        return {
            "total":         total,
            "wins":          wins,
            "losses":        total - wins,
            "win_rate":      round(wr, 1),
            "session_wr":    session_wr,
            "best_pattern":  best_pattern,
            "worst_pattern": worst_pattern,
            "recent_10_wr":  _recent_wr(memory, 10),
        }

    except Exception as exc:
        logger.error("get_stats error: %s", exc)
        return {}


def _recent_wr(memory: list, n: int) -> str:
    recent = memory[-n:]
    if not recent:
        return "N/A"
    wins = sum(1 for t in recent if t["result"] == "WIN")
    return f"{wins}/{len(recent)} ({wins/len(recent)*100:.0f}%)"


def _describe_setup(signal: Dict) -> str:
    bias    = signal.get("bias_state", "")
    pattern = signal.get("pattern", "")
    session = signal.get("session", "")
    return f"{bias} + {pattern} @ {session}".strip(" +")


def _load() -> List[Dict]:
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save(memory: List[Dict]):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2, default=str)
