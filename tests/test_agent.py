
from agent.planner import route_query

def test_revenue_vs_budget():
    intent, params = route_query("What was June 2025 revenue vs budget in USD?")
    assert intent == "revenue_vs_budget"
    assert params["month"].startswith("2025-")

def test_gm_trend_last3():
    intent, params = route_query("Show Gross Margin % trend for the last 3 months.")
    assert intent == "gross_margin_trend"
    assert params["last_n"] == 3

def test_opex_breakdown():
    intent, params = route_query("Break down Opex by category for June 2025.")
    assert intent == "opex_breakdown"
    assert params["month"].startswith("2025-")

def test_cash_runway():
    intent, params = route_query("What is our cash runway right now?")
    assert intent == "cash_runway"

def test_ebitda():
    intent, params = route_query("What is EBITDA for June 2025?")
    assert intent == "ebitda"
    assert params["month"].startswith("2025-")

def test_synonyms_and_iso_month():
    # revenue vs plan, ISO month
    intent, params = route_query("Compare sales vs plan for 2025-06")
    assert intent == "revenue_vs_budget"
    assert params["month"] == "2025-06"

def test_quarter_parsing_for_month_pick():
    # Q2 2025 -> map to 2025-06 as representative month
    intent, params = route_query("Revenue vs budget for Q2 2025")
    assert intent == "revenue_vs_budget"
    assert params["month"] == "2025-06"
