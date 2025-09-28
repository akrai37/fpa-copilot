
import re
from typing import Tuple, Dict, Any, Optional

# Prefer dateparser for robust fuzzy parsing, fallback to dateutil if missing
try:
    import dateparser
except Exception:
    dateparser = None
from dateutil import parser as dtparser

MONTH_REGEX = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}", re.IGNORECASE)
ISO_YYYY_MM = re.compile(r"\b(\d{4})-(0[1-9]|1[0-2])\b")
QUARTER_REGEX = re.compile(r"\b(q[1-4])\s*(\d{4})\b", re.IGNORECASE)

SYNONYMS = {
    "budget": {"budget", "plan", "target"},
    "revenue": {"revenue", "sales", "topline"},
    "gross_margin": {"gross margin", "gm%", "gm %", "gm", "margin%", "margin %"},
    "opex": {"opex", "operating expense", "operating expenses", "opex:"},
    "breakdown": {"breakdown", "split", "by category", "category"},
    "cash_runway": {"cash runway", "runway", "months runway"},
}
LAST_N_REGEX = re.compile(r"last\s+(\d+)\s+months?", re.IGNORECASE)

def _normalize_month(text: str) -> Optional[str]:
    # Qx YYYY support
    qm = QUARTER_REGEX.search(text)
    if qm:
        q = qm.group(1).lower()  # q1..q4
        year = int(qm.group(2))
        start_month = {"q1":1, "q2":4, "q3":7, "q4":10}[q]
        # choose last month of quarter as representative for month-specific functions
        month = start_month + 2
        return f"{year:04d}-{month:02d}"

    # ISO YYYY-MM directly
    im = ISO_YYYY_MM.search(text)
    if im:
        y, m = int(im.group(1)), int(im.group(2))
        return f"{y:04d}-{m:02d}"

    m = MONTH_REGEX.search(text)
    try:
        if m:
            dt = (dateparser.parse(m.group(0)) if dateparser else None)
            if dt is None:
                dt = dtparser.parse(m.group(0))
            return dt.strftime("%Y-%m")
        dt = (dateparser.parse(text) if dateparser else None)
        if dt is None:
            dt = dtparser.parse(text, fuzzy=True)
        if dt and getattr(dt, 'year', None) and getattr(dt, 'month', None):
            return dt.strftime("%Y-%m")
    except Exception:
        pass
    return None

def _extract_last_n(text: str, default: int = 3) -> int:
    m = LAST_N_REGEX.search(text)
    if m:
        try:
            return max(1, int(m.group(1)))
        except ValueError:
            return default
    return default

def route_query(question: str) -> Tuple[str, Dict[str, Any]]:
    q = question.lower()

    # EBITDA
    if "ebitda" in q:
        month = _normalize_month(question)
        return "ebitda", {"month": month}

    if any(s in q for s in SYNONYMS["cash_runway"]) or ("runway" in q and "cash" in q):
        return "cash_runway", {}

    if any(s in q for s in SYNONYMS["gross_margin"]):
        last_n = _extract_last_n(q, default=3)
        return "gross_margin_trend", {"last_n": last_n}

    if any(s in q for s in SYNONYMS["opex"]) and any(s in q for s in SYNONYMS["breakdown"]):
        month = _normalize_month(question)
        return "opex_breakdown", {"month": month}

    if any(s in q for s in SYNONYMS["revenue"]) and (any(s in q for s in SYNONYMS["budget"]) or " vs " in f" {q} "):
        month = _normalize_month(question)
        return "revenue_vs_budget", {"month": month}

    return "fallback", {}
