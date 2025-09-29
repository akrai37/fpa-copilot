"""
Minimal local RAG-style retriever for FP&A questions.

This module builds a tiny textual corpus from our existing tools' summaries
(e.g., revenue vs budget per month, opex breakdown per month, GM% trend, cash runway)
and uses a TF-IDF retriever to select the most relevant snippet for a user query.

We DO NOT generate answers here. Instead, we map the top retrieved snippet back
to an intent and parameters (like month), then delegate to our deterministic tools.

This gives us a flexible, synonym-friendly fallback without changing the model.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import functools
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional local embeddings via sentence-transformers
_EMBED_MODEL = None
_EMBED_MODEL_NAME = None
try:
    import os
    from sentence_transformers import SentenceTransformer
    _EMBED_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Lazy-load the model on first use
    def _get_embed_model():
        global _EMBED_MODEL
        if _EMBED_MODEL is None:
            _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
        return _EMBED_MODEL
except Exception:  # sentence-transformers not installed or failed to init
    def _get_embed_model():
        return None

from agent import tools

# Simple synonyms and month/quarter helpers to enrich documents for better recall
SYNONYMS = {
    "revenue": ["revenue", "sales", "topline", "turnover"],
    "budget": ["budget", "plan", "target"],
    "opex": ["opex", "operating expenses", "operating expense", "breakdown", "by category", "split"],
    "gm": ["gross margin", "gm", "gm%", "margin%"],
    "cash_runway": ["cash runway", "runway", "run out of cash", "months left"],
    "ebitda": ["ebitda", "earnings before interest tax depreciation amortization"],
}

MONTH_ALIASES = {
    1: ["jan", "january"], 2: ["feb", "february"], 3: ["mar", "march"],
    4: ["apr", "april"], 5: ["may"], 6: ["jun", "june"], 7: ["jul", "july"],
    8: ["aug", "august"], 9: ["sep", "september"], 10: ["oct", "october"],
    11: ["nov", "november"], 12: ["dec", "december"],
}

def _enrich_text(base: str, intent: str, month_str: Optional[str] = None) -> str:
    """Append synonym and date alias tokens to improve matching without changing content meaning."""
    extras: List[str] = []
    # intent synonym tokens
    if intent == "revenue_vs_budget":
        extras += SYNONYMS["revenue"] + SYNONYMS["budget"]
    elif intent == "opex_breakdown":
        extras += SYNONYMS["opex"]
    elif intent == "gross_margin_trend":
        extras += SYNONYMS["gm"]
    elif intent == "cash_runway":
        extras += SYNONYMS["cash_runway"]
    elif intent == "ebitda":
        extras += SYNONYMS["ebitda"]

    # month aliases (e.g., 2025-06 -> jun june Q2)
    if month_str:
        try:
            p = pd.Period(month_str, freq="M")
            mnum = p.month
            year = p.year
            extras += [month_str, f"{year}", f"{p.strftime('%b').lower()}", f"{p.strftime('%B').lower()}"]
            extras += MONTH_ALIASES.get(mnum, [])
            # quarter tag (Q2 2025)
            q = (mnum - 1) // 3 + 1
            extras += [f"q{q}", f"q{q} {year}", f"quarter {q}"]
        except Exception:
            pass

    return base + "\n\n" + " ".join(set(extras))


def _available_recent_months(limit: int = 12) -> List[str]:
    """Return up to `limit` recent YYYY-MM periods seen across actuals/budget.

    We choose the union of months present in actuals/budget, sorted descending,
    and return the top N as strings (YYYY-MM) for stable serialization.
    """
    dfs = tools.load_data()
    months = set()
    for k in ("actuals", "budget"):
        df = dfs.get(k)
        if df is not None and "month" in df.columns:
            s = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").dropna().astype(str)
            months.update(s.tolist())
    recent = sorted(months, reverse=True)[:limit]
    # ensure most recent first
    return recent


def _safe_tool_text(callable_fn, *args, **kwargs) -> Optional[str]:
    try:
        text, _ = callable_fn(*args, **kwargs)
        return text
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def build_index() -> Dict[str, Any]:
    """Build a TF-IDF index of tool-generated summaries with metadata.

    Returns dict containing:
      - vectorizer: TfidfVectorizer
      - matrix: TF-IDF matrix of shape (n_docs, n_terms)
      - docs: list of dicts with keys {id, text, intent, params, title}
    """
    docs: List[Dict[str, Any]] = []

    # Month-specific docs for revenue vs budget and opex breakdown
    for mstr in _available_recent_months(limit=12):
        # Revenue vs Budget
        t1 = _safe_tool_text(tools.get_revenue_vs_budget, month=mstr)
        if t1:
            t1 = _enrich_text(t1, intent="revenue_vs_budget", month_str=mstr)
            docs.append({
                "id": f"revbud-{mstr}",
                "title": f"Revenue vs Budget — {mstr}",
                "text": t1,
                "intent": "revenue_vs_budget",
                "params": {"month": mstr},
            })
        # Opex breakdown
        t2 = _safe_tool_text(tools.get_opex_breakdown, month=mstr)
        if t2:
            t2 = _enrich_text(t2, intent="opex_breakdown", month_str=mstr)
            docs.append({
                "id": f"opex-{mstr}",
                "title": f"Opex Breakdown — {mstr}",
                "text": t2,
                "intent": "opex_breakdown",
                "params": {"month": mstr},
            })

    # Non-month-specific docs
    t3 = _safe_tool_text(tools.get_gross_margin_trend, 3)
    if t3:
        t3 = _enrich_text(t3, intent="gross_margin_trend", month_str=None)
        docs.append({
            "id": "gm-trend",
            "title": "Gross Margin % Trend",
            "text": t3,
            "intent": "gross_margin_trend",
            "params": {"last_n": 3},
        })

    t4 = _safe_tool_text(tools.get_cash_runway)
    if t4:
        t4 = _enrich_text(t4, intent="cash_runway", month_str=None)
        docs.append({
            "id": "cash-runway",
            "title": "Cash Runway",
            "text": t4,
            "intent": "cash_runway",
            "params": {},
        })

    # EBITDA month docs (mirrors revenue/opex month-level content)
    for mstr in _available_recent_months(limit=12):
        t5 = _safe_tool_text(tools.get_ebitda, month=mstr)
        if t5:
            t5 = _enrich_text(t5, intent="ebitda", month_str=mstr)
            docs.append({
                "id": f"ebitda-{mstr}",
                "title": f"EBITDA — {mstr}",
                "text": t5,
                "intent": "ebitda",
                "params": {"month": mstr},
            })

    # Build TF-IDF index
    corpus = [d["text"] for d in docs]
    if not corpus:
        # empty corpus fallback
        return {"vectorizer": None, "matrix": None, "docs": [], "embeddings": None}

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    matrix = vectorizer.fit_transform(corpus)

    # Optional embeddings
    embeddings = None
    model = _get_embed_model()
    if model is not None:
        try:
            # Normalize for cosine via dot product
            embeddings = model.encode(corpus, normalize_embeddings=True)
        except Exception:
            embeddings = None

    return {"vectorizer": vectorizer, "matrix": matrix, "docs": docs, "embeddings": embeddings}


def retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    idx = build_index()
    if not idx["docs"]:
        return []

    # Quick path: if the query explicitly contains a month/year (e.g., 'June 2025' or '2025-06'),
    # prefer returning docs that have that month in their params (exact match).
    def _parse_month(text: str) -> Optional[str]:
        try:
            ts = pd.to_datetime(text, errors="coerce")
            if ts is not pd.NaT and not pd.isna(ts):
                return str(pd.Period(ts, freq="M"))
        except Exception:
            pass
        return None

    m = _parse_month(query)
    if m:
        # return month-matching docs first (score boosted)
        month_docs: List[Dict[str, Any]] = []
        # human-readable tokens to check (e.g., 'Jun 2025', 'June 2025')
        try:
            p = pd.Period(m, freq="M")
            hr1 = p.strftime("%b %Y").lower()
            hr2 = p.strftime("%B %Y").lower()
        except Exception:
            hr1 = hr2 = m

        for d in idx["docs"]:
            params_month = d.get("params", {}).get("month")
            text_lower = (d.get("text") or "").lower()
            title_lower = (d.get("title") or "").lower()
            id_lower = (d.get("id") or "").lower()
            if params_month == m or m in id_lower or m in title_lower or hr1 in text_lower or hr2 in text_lower or hr1 in title_lower or hr2 in title_lower:
                month_docs.append(dict(d, score=1.0))
        if month_docs:
            # Prefer a sensible intent priority for month-scoped queries
            intent_priority = {
                "revenue_vs_budget": 0,
                "opex_breakdown": 1,
                "ebitda": 2,
                "gross_margin_trend": 3,
                "cash_runway": 4,
            }
            month_docs.sort(key=lambda d: intent_priority.get(d.get("intent"), 99))
            return month_docs[:top_k]

    # Otherwise, prefer embeddings if available, else TF-IDF
    scores = None
    if idx.get("embeddings") is not None:
        try:
            model = _get_embed_model()
            if model is not None:
                q_emb = model.encode([query], normalize_embeddings=True)[0]
                # cosine for normalized vectors is dot product
                import numpy as _np
                scores = (idx["embeddings"] @ _np.asarray(q_emb))
        except Exception:
            scores = None

    if scores is None:
        if idx["vectorizer"] is None or idx["matrix"] is None:
            return []
        q_vec = idx["vectorizer"].transform([query])
        sims = cosine_similarity(q_vec, idx["matrix"])  # shape (1, n_docs)
        scores = sims.ravel()

    # if all similarities are essentially zero, try to fallback to month-matching by parsing
    if float(scores.max()) <= 1e-12 and m:
        month_docs = [dict(d, score=1.0) for d in idx["docs"] if d.get("params", {}).get("month") == m]
        if month_docs:
            return month_docs[:top_k]

    order = scores.argsort()[::-1]
    out = []
    for i in order[:top_k]:
        doc = dict(idx["docs"][i])
        doc["score"] = float(scores[i])
        out.append(doc)
    return out


def route_with_rag(question: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Use retrieval to infer the best intent+params for a question.

    Returns (intent, params) or None if low confidence or no docs.
    """
    cands = retrieve(question, top_k=3)
    if not cands:
        return None

    top = cands[0]
    # Basic low-confidence filter: require some minimal similarity
    if top.get("score", 0.0) < 0.05:
        return None
    return top["intent"], top.get("params", {})
