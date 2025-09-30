# CFO Copilot — Mini FP&A Agent (Streamlit)

![tests](https://github.com/akrai37/fpa-copilot/actions/workflows/tests.yml/badge.svg)

CFO Copilot is a lightweight FP&A assistant built with Streamlit. It reads committed CSV fixtures (actuals, budget, FX, and cash) and answers a small set of common FP&A questions with text summaries and charts. It also supports a compact 2-page PDF export intended for quick board-pack snapshots.

## Project contents
- `app.py` — Streamlit application and UI glue.
- `agent/` — core agent logic:
  - `planner.py` — simple rule-based intent/parameter parsing and routing.
  - `tools.py` — data-loading helpers, finance computations (Revenue vs Budget, Gross Margin %, Opex breakdown, Cash runway), and PDF export.
  - `rag.py` — a minimal retrieval module used as a fallback to help map fuzzy queries to intents/params.
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

## Features
- Chat agent → intent classification → deterministic tools → text + charts (inline).
- Supported metrics: Revenue vs Budget, Gross Margin %, Opex breakdown, EBITDA, Cash runway.
- Optional local embeddings (Hugging Face sentence-transformers, default: `sentence-transformers/all-MiniLM-L6-v2`) to improve fuzzy matching; falls back to TF‑IDF.
- Tests included for planner and tool functions (pytest).
- Optional PDF export: quick 2-page board-style snapshot.

## Example queries
The agent supports a small set of structured FP&A queries. Examples:

- "What was June 2025 revenue vs budget in USD?"
- "Show Gross Margin % trend for the last 3 months."
- "Break down Opex by category for June 2025."
- "What is our cash runway right now?"
- "What is EBITDA for June 2025?"

If a question needs a month and the user omits it, the UI guides the user to provide a month (e.g., "June 2025").

## Design & model

This project evolved from a conservative, rule-based router to a small RAG-style retriever used as a robust fallback:

- Rule-based first: `agent/planner.py` contains deterministic, tested keyword-and-date heuristics that map queries to a small set of intents (revenue_vs_budget, opex_breakdown, gross_margin_trend, cash_runway, ebitda). This guarantees predictable behavior and avoids hallucination for core finance calculations.

- RAG as a focused fallback: `agent/rag.py` builds a tiny corpus of tool-generated summaries (recent revenue/budget summaries, opex breakdowns, GM trend text, cash runway summaries) and indexes it with TF‑IDF. Retrieval helps match synonyms and fuzzy phrasing (e.g., "topline vs plan", "run out of cash") without changing the deterministic tools.

- Embeddings & model used: if `sentence-transformers` is installed the code can optionally compute dense embeddings using a Hugging Face sentence-transformers model — the default in this project is `sentence-transformers/all-MiniLM-L6-v2`. When available the pipeline can also use FAISS for faster similarity search. TF‑IDF remains the portable fallback when embeddings are not available.

- Why the retriever delegates to deterministic tools (no generation): the system uses retrieval to infer intent + parameters, then delegates to the existing analytic functions in `agent/tools.py` to produce numeric answers and charts. This keeps numeric calculations authoritative and avoids LLM hallucination when returning financial metrics.

- Why not a larger open/paid LLM or large local model?
  - Data size & signal: the app operates on small, structured CSV tables and a handful of standard reports. A large, general-purpose LLM is overkill for mapping a small set of finance questions to deterministic computations.
  - Cost & latency: hosted paid LLMs incur API costs and add latency; for a compact internal demo the small TF‑IDF + optional lightweight embedding approach is cheaper and faster.
  - Privacy & control: using local retrieval and deterministic tools keeps financial figures in code and CSVs — there is no need to send sensitive data to external generative APIs for this use case.
  - Resources: running large local models (Llama, large Hugging Face weights) requires significant CPU/GPU memory and engineering overhead. Given the small corpus and the deterministic computation layer, they do not materially improve accuracy here.

If you later need to handle open-ended narrative QA over large documents (disclosures, long contracts, board minutes), the repo is structured so a larger retrieval+generation pipeline can be slotted in later (e.g., production embeddings + an authorised LLM), but for this assignment the current hybrid keeps behavior predictable and auditable.

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
The project is built with Streamlit, so it can be deployed to any platform that supports Streamlit apps (e.g., Streamlit Community Cloud, Hugging Face Spaces, or similar).

For this assignment, local quickstart instructions (pip install -r requirements.txt and streamlit run app.py) are provided to run and evaluate the app.

## Troubleshooting
- If a required CSV is missing, the app will show a data-health warning in the sidebar.
- If plots do not render, ensure `matplotlib` is installed and that the environment supports GUI-less rendering (the included requirements are configured for headless use).

---

If someone wants changes to the README (more examples, sample output screenshots, or a CONTRIBUTING section), say which additions to make and they will be applied.
