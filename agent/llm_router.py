"""
Optional LLM-assisted query parser.

This module attempts to parse a messy user query into (intent, params) using an LLM
if available. It is STRICTLY OPTIONAL and never blocks the app:
- If no API key or client library is available, it returns None.
- Calculators remain unchanged; this only helps the router.

Enable by setting OPENAI_API_KEY and installing 'openai' package:
  pip install openai

You can also set MODEL via env var LLM_ROUTER_MODEL (default: gpt-4o-mini).
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import os
import json

ALLOWED_INTENTS = {"revenue_vs_budget", "gross_margin_trend", "opex_breakdown", "cash_runway", "ebitda", "fallback"}


def parse_query_with_llm(question: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        # openai package not installed
        return None

    model = os.getenv("LLM_ROUTER_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    system = (
        "You convert a user's finance question into a JSON object with: "
        "intent (one of: revenue_vs_budget, gross_margin_trend, opex_breakdown, cash_runway, ebitda, fallback) "
        "and params (month 'YYYY-MM' if applicable or last_n for trends). Use null when unknown."
    )
    user = f"Question: {question}\nReturn ONLY JSON: {{\"intent\":..., \"params\":{{...}}}}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        content = resp.choices[0].message.content if resp and resp.choices else ""
        # Attempt to parse JSON
        data = json.loads(content)
        intent = data.get("intent")
        params = data.get("params", {}) if isinstance(data.get("params"), dict) else {}
        if intent not in ALLOWED_INTENTS:
            return None
        return intent, params
    except Exception:
        return None
