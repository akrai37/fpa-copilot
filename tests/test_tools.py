from agent import tools


def test_revenue_vs_budget_text_contains_values():
    text, fig = tools.get_revenue_vs_budget('2025-06')
    assert 'revenue was' in text.lower()
    assert 'budget' in text.lower()
    assert fig is not None


def test_gm_trend_has_percentages():
    text, fig = tools.get_gross_margin_trend(3)
    # Expect something like "84.8%"
    assert '%' in text
    assert fig is not None or 'not enough data' in text.lower()


def test_opex_breakdown_month():
    text, fig = tools.get_opex_breakdown('2025-06')
    # If there is data, ensure the summary mentions Opex
    assert 'opex' in text.lower()
    # fig can be None if no data, but with provided fixtures it should exist
    assert fig is not None or 'no opex data' in text.lower()


def test_ebitda_month():
    text, fig = tools.get_ebitda('2025-06')
    assert 'ebitda' in text.lower()
    assert fig is not None or 'please specify a month' in text.lower()


def test_cash_runway_message():
    text, fig = tools.get_cash_runway()
    # Accept any of the three message types depending on data/state
    assert any(kw in text.lower() for kw in [
        'cash runway',
        'not enough burn data',
        'profitable or zero net burn',
    ])
    assert fig is not None
