# CFO Copilot — Mini FP&A Agent (Streamlit)

![tests](https://github.com/akrai37/fpa-copilot/actions/workflows/tests.yml/badge.svg)

CFO Copilot is a lightweight FP&A assistant built with Streamlit. It reads committed CSV fixtures (actuals, budget, FX, and cash) and answers a small set of common FP&A questions with text summaries and charts. It also supports a compact 2-page PDF export intended for quick board-pack snapshots.

## Project contents
- `app.py` — Streamlit application and UI glue.
- `agent/` — core agent logic:
  - `planner.py` — simple rule-based intent/parameter parsing and routing.
  - `tools.py` — data-loading helpers, finance computations (Revenue vs Budget, Gross Margin %, Opex breakdown, Cash runway), and PDF export.
- `fixtures/` — example CSV files used by the app and tests (`actuals.csv`, `budget.csv`, `fx.csv`, `cash.csv`).
- `tests/` — pytest unit tests for core functions.
- `requirements.txt` — Python dependencies for running the app and tests.

## Quickstart (run locally)

1. Clone the repository and change into the project folder:

```bash
git clone https://github.com/akrai37/fpa-copilot.git
cd fpa-copilot
```

2. Create and activate a virtual environment (macOS / Linux shown):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

4. The app expects sample CSVs in `fixtures/`. The repo includes example fixtures suitable for local testing.

## Supported queries
The agent supports a small set of structured FP&A queries. Examples:

- "What was June 2025 revenue vs budget in USD?"
- "Show Gross Margin % trend for the last 3 months."
- "Break down Opex by category for June 2025."
- "What is our cash runway right now?"
- "What is EBITDA for June 2025?"

If a question needs a month and the user omits it, the UI guides the user to provide a month (e.g., "June 2025").

## Data format (fixtures)
Place CSV files in the `fixtures/` directory. Expected files and minimal columns:

- `actuals.csv` — columns: `month`, `account` (or `account_category`), `amount`, `currency` (optional).
- `budget.csv` — same structure as `actuals.csv` for month-level budgets.
- `fx.csv` — columns: `month`, `currency`, `usd_rate` (or other common rate column names; the loader attempts to detect common variants). USD rates may be left out for months that are USD.
- `cash.csv` — columns: `month`, `amount` (or `cash_usd` / `balance`), `currency` (optional).

The code normalizes column names and period formats, and it attempts to forward-fill FX rates when needed.

## Cash runway note
The app's Cash runway computation is derived from P&L (Revenue − COGS − Opex averaged over the selected window) and therefore reflects operating profitability rather than observed changes in the bank balance. Non‑P&L items (capex, financing, timing of receivables/payables, one‑offs) can make the cash-balance runway differ materially. For a strict liquidity view, compute runway from observed cash-balance changes.

## Testing
Run the unit tests with pytest (recommended inside the virtualenv):

```bash
python -m pytest -q
```

The repository includes tests for the planner and tools modules.

## Deploy
To deploy to Streamlit Community Cloud:
1. Push the repository to GitHub.
2. On https://share.streamlit.io create a new app and point it to `app.py`.

The `requirements.txt` and `runtime.txt` are provided for reproducible builds.

## Troubleshooting
- If a required CSV is missing, the app will show a data-health warning in the sidebar.
- If plots do not render, ensure `matplotlib` is installed and that the environment supports GUI-less rendering (the included requirements are configured for headless use).

---

If someone wants changes to the README (more examples, sample output screenshots, or a CONTRIBUTING section), say which additions to make and they will be applied.
