
import re
from dateutil import parser
from typing import Tuple, Dict, Any, Optional

MONTH_REGEX = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}", re.IGNORECASE)
LAST_N_REGEX = re.compile(r"last\s+(\d+)\s+months?", re.IGNORECASE)

def _normalize_month(text: str) -> Optional[str]:
    m = MONTH_REGEX.search(text)
    try:
        if m:
            dt = parser.parse(m.group(0))
            return dt.strftime("%Y-%m")
        dt = parser.parse(text, fuzzy=True, default=None)
        if dt and dt.year and dt.month:
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

    if "cash runway" in q or ("runway" in q and "cash" in q):
        return "cash_runway", {}

    if "gross margin" in q or "gm%" in q or "gm %" in q:
        last_n = _extract_last_n(q, default=3)
        return "gross_margin_trend", {"last_n": last_n}

    if "opex" in q and ("breakdown" in q or "category" in q):
        month = _normalize_month(question)
        return "opex_breakdown", {"month": month}

    if "revenue" in q and ("budget" in q or "vs" in q):
        month = _normalize_month(question)
        return "revenue_vs_budget", {"month": month}

    return "fallback", {}
