# 📊 CFO Copilot — Mini FP&A Agent (Streamlit)

End-to-end rule-based agent that answers FP&A questions from CSV data, with charts and optional PDF export.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Place CSVs in `fixtures/` named: `actuals.csv`, `budget.csv`, `fx.csv`, `cash.csv`.

## Supported questions
- What was June 2025 revenue vs budget in USD?
- Show Gross Margin % trend for the last 3 months.
- Break down Opex by category for June 2025.
- What is our cash runway right now?
 - What is EBITDA for June 2025?

PDF Export (optional extra): Sidebar button generates a compact 2-page Board Pack PDF containing Revenue vs Budget, GM% trend, Opex breakdown, and Cash trend with runway.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public recommended to start).
2. Go to https://share.streamlit.io and click “New app”.
3. Select your repo, branch, and set the main file to `app.py`.
4. Deploy. The first build takes a couple minutes.

Notes
- App reads data from committed CSVs in `fixtures/` – no secrets needed.
- `requirements.txt` installs Streamlit, pandas, numpy, matplotlib, reportlab, etc.
- `runtime.txt` pins Python (3.11). You can change it if needed.
- PDF export uses temporary files and provides a download button.

Troubleshooting
- If you hit a missing package error, add it to `requirements.txt` and redeploy.
- If charts don’t render, ensure matplotlib is installed and avoid LaTeX in text.
