
import re
import pandas as pd
from typing import Tuple, Dict, Any, Optional

# Prefer dateparser for robust fuzzy parsing, fallback to dateutil if missing
try:
    import dateparser
except Exception:
    dateparser = None
from dateutil import parser as dtparser
# Optional LLM-assisted router then RAG fallback
try:
    from agent.llm_router import parse_query_with_llm
except Exception:
    parse_query_with_llm = None
try:
    from agent.rag import route_with_rag
except Exception:
    route_with_rag = None
# lazy import helper to avoid circular work at import time
def _latest_year_from_data() -> Optional[int]:
    try:
        # import here to avoid top-level dependency during tests
        from agent import tools as _tools
        dfs = _tools.load_data()
        months = []
        for k in ("actuals","budget","cash"):
            df = dfs.get(k)
            if df is not None and "month" in df.columns:
                months.extend(pd.to_datetime(df["month"], errors="coerce").dropna().dt.year.astype(int).tolist())
        return max(months) if months else None
    except Exception:
        return None

MONTH_REGEX = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}", re.IGNORECASE)
MONTH_WORD = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.IGNORECASE)
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
LAST_N_REGEX = re.compile(r"last\s+(\d+)\s*months?", re.IGNORECASE)
# Also accept generic forms like "6 months" or "6months"
GENERIC_N_MONTHS = re.compile(r"\b(\d+)\s*months?\b", re.IGNORECASE)

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
            # If parsed date doesn't include year (rare), try to infer latest year from data
            if getattr(dt, 'year', None) is None or dt.year < 1900:
                latest = _latest_year_from_data()
                if latest:
                    return f"{latest:04d}-{dt.month:02d}"
            return dt.strftime("%Y-%m")

        # Try full-text parse only if there's an explicit month word or a clear date-like indicator
        has_month_word = MONTH_WORD.search(text) is not None
        has_year_token = re.search(r"\b\d{4}\b", text) is not None
        if has_month_word or has_year_token:
            dt = (dateparser.parse(text) if dateparser else None)
            if dt is None:
                dt = dtparser.parse(text, fuzzy=True)
            if dt and getattr(dt, 'month', None):
                if has_year_token:
                    return dt.strftime("%Y-%m")
                latest = _latest_year_from_data()
                if latest:
                    return f"{latest:04d}-{dt.month:02d}"
                if getattr(dt, 'year', None):
                    return dt.strftime("%Y-%m")
    except Exception:
        pass
    return None

def _extract_last_n(text: str, default: int = 3) -> int:
    m = LAST_N_REGEX.search(text)
    if not m:
        m = GENERIC_N_MONTHS.search(text)
    if m:
        try:
            return max(1, int(m.group(1)))
        except ValueError:
            return default
    return default

def route_query(question: str) -> Tuple[str, Dict[str, Any]]:
    q = question.lower()
    # Phrase normalization map: common multi-word phrases -> canonical token
    PHRASE_MAP = {
        "break down": "breakdown",
        "run out of cash": "runway",
        "months left": "runway",
        "months remaining": "runway",
        "year on year": "yoy",
        "year-over-year": "yoy",
    }
    for a, b in PHRASE_MAP.items():
        q = q.replace(a, b)

    # 1) Optional: LLM-assisted parsing to map messy English to structured intent/params
    if parse_query_with_llm is not None:
        try:
            parsed = parse_query_with_llm(question)
            if parsed is not None:
                intent, params = parsed
                # Light normalization for month if present as free text
                if intent in {"revenue_vs_budget", "opex_breakdown", "ebitda"} and not params.get("month"):
                    m = _normalize_month(question)
                    if m:
                        params["month"] = m
                if intent == "gross_margin_trend" and not params.get("last_n"):
                    params["last_n"] = _extract_last_n(q, default=3)
                return intent, params
        except Exception:
            pass

    # 2) Deterministic keyword rules (safety & tests)
    if "ebitda" in q:
        month = _normalize_month(question)
        return "ebitda", {"month": month}

    if any(s in q for s in SYNONYMS["cash_runway"]) or ("runway" in q and "cash" in q):
        last_n = _extract_last_n(q, default=3)
        return "cash_runway", {"last_n": last_n}

    if any(s in q for s in SYNONYMS["gross_margin"]):
        last_n = _extract_last_n(q, default=3)
        return "gross_margin_trend", {"last_n": last_n}

    if any(s in q for s in SYNONYMS["opex"]) and any(s in q for s in SYNONYMS["breakdown"]):
        month = _normalize_month(question)
        return "opex_breakdown", {"month": month}

    # If user mentions 'breakdown' with revenue/budget, treat it as revenue vs budget (not Opex)
    if any(s in q for s in SYNONYMS["breakdown"]):
        if any(s in q for s in SYNONYMS["revenue"]) or any(s in q for s in SYNONYMS["budget"]):
            month = _normalize_month(question)
            return "revenue_vs_budget", {"month": month}
        # Otherwise, default 'breakdown' to Opex
        month = _normalize_month(question)
        return "opex_breakdown", {"month": month}

    # If the query mentions Opex at all, prefer Opex breakdown (month may be missing)
    if any(s in q for s in SYNONYMS["opex"]):
        month = _normalize_month(question)
        return "opex_breakdown", {"month": month}

    # Revenue intent: be permissive — if user says revenue and includes a month, assume revenue vs budget
    if any(s in q for s in SYNONYMS["revenue"]):
        month = _normalize_month(question)
        if month or any(s in q for s in SYNONYMS["budget"]) or " vs " in f" {q} ":
            return "revenue_vs_budget", {"month": month}

    # If user only provided a month/date and no clear metric, do not guess — let UI prompt
    has_metric = (
        any(s in q for s in SYNONYMS["revenue"]) or
        any(s in q for s in SYNONYMS["opex"]) or
        any(s in q for s in SYNONYMS["gross_margin"]) or
        ("ebitda" in q) or
        any(s in q for s in SYNONYMS["cash_runway"]) or
        any(s in q for s in SYNONYMS["breakdown"])  # breakdown implies opex
    )
    if not has_metric and _normalize_month(question):
        return "fallback", {}

    # 3) RAG retrieval only if rules above didn't determine intent
    if route_with_rag is not None:
        try:
            rag = route_with_rag(question)
            if rag is not None:
                intent, params = rag
                # refine params as above
                if intent in {"revenue_vs_budget", "opex_breakdown", "ebitda"}:
                    m = _normalize_month(question)
                    if m:
                        params["month"] = m
                if intent == "gross_margin_trend":
                    ln = _extract_last_n(q, default=params.get("last_n", 3))
                    params["last_n"] = ln
                # Guardrail: if there are no metric keywords and no obvious date/window tokens,
                # avoid routing to a metric based solely on RAG; return fallback instead.
                has_metric = (
                    any(s in q for s in SYNONYMS["revenue"]) or
                    any(s in q for s in SYNONYMS["opex"]) or
                    any(s in q for s in SYNONYMS["gross_margin"]) or
                    ("ebitda" in q) or
                    any(s in q for s in SYNONYMS["cash_runway"])
                )
                has_month = _normalize_month(question) is not None
                has_window = (LAST_N_REGEX.search(q) is not None) or (GENERIC_N_MONTHS.search(q) is not None)
                if not has_metric and not has_month and not has_window:
                    return "fallback", {}
                return intent, params
        except Exception:
            pass

    return "fallback", {}
