
from __future__ import annotations
import os, math
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")

# ---------- helpers for robust CSV handling ----------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _standardize_fx(fx: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns: month (Period[M]), currency (UPPER), usd_rate (float).
    Accepts header variants and fills USD=1.0 if missing.
    """
    fx = _normalize_columns(fx)
    # pick a usable rate column
    candidates = [
        "usd_rate",
        "rate",
        "fx_rate",
        "usd",
        "to_usd",
        "usdconversion",
        "usd_conversion",
        "rate_to_usd",
    ]
    rate_col = next((c for c in candidates if c in fx.columns), None)
    if rate_col is None:
        fx["usd_rate"] = np.where(fx.get("currency", pd.Series("USD")).astype(str).str.upper().eq("USD"), 1.0, np.nan)
    else:
        if rate_col != "usd_rate":
            fx = fx.rename(columns={rate_col: "usd_rate"})
    # month normalization
    month_col = "month" if "month" in fx.columns else ("date" if "date" in fx.columns else None)
    if month_col is None:
        raise ValueError("fx.csv must have a 'month' or 'date' column.")
    fx["month"] = pd.to_datetime(fx[month_col], errors="coerce").dt.to_period("M")
    # currency normalization
    fx["currency"] = fx.get("currency", pd.Series("USD")).astype(str).str.strip().str.upper()
    fx.loc[fx["currency"].eq("USD") & fx["usd_rate"].isna(), "usd_rate"] = 1.0
    return fx[["month","currency","usd_rate"]]

def prepare_fx(fx: pd.DataFrame, months_needed: Optional[pd.Series] = None) -> pd.DataFrame:
    """Prepare FX table for use: standardize, ensure USD=1.0 for required months and forward-fill per currency.

    months_needed: Series-like of months (datetime/period/string) that we must ensure USD rows for.
    """
    fx_std = _standardize_fx(fx.copy())
    # build months list
    if months_needed is None:
        months = pd.Series([], dtype=object)
    else:
        months = pd.to_datetime(pd.Series(months_needed).dropna()).dt.to_period("M")
    months = months.drop_duplicates().sort_values()

    if not months.empty:
        usd_fill = pd.DataFrame({
            "month": months.astype(object),
            "currency": "USD",
            "usd_rate": 1.0,
        })
        fx_comb = pd.concat([fx_std, usd_fill], ignore_index=True)
    else:
        fx_comb = fx_std.copy()

    # drop duplicates keeping last (so explicit fx overrides fill)
    fx_comb = fx_comb.drop_duplicates(subset=["month", "currency"], keep="last")

    # forward-fill usd_rate within each currency (sort by currency+month first)
    fx_comb = fx_comb.sort_values(["currency", "month"]).reset_index(drop=True)
    fx_comb["usd_rate"] = fx_comb.groupby("currency")["usd_rate"].ffill()
    return fx_comb

def _account_col(df: pd.DataFrame) -> Optional[str]:
    """Return the name of the account column regardless of header variant."""
    if df is None:
        return None
    if "account" in df.columns:
        return "account"
    if "account_category" in df.columns:
        return "account_category"
    return None

# ---------- load & health ----------
def _read_csv_safe(filename: str) -> Optional[pd.DataFrame]:
    path = os.path.join(FIXTURES_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def load_data() -> Dict[str, Optional[pd.DataFrame]]:
    actuals = _read_csv_safe("actuals.csv")
    budget  = _read_csv_safe("budget.csv")
    fx      = _read_csv_safe("fx.csv")
    cash    = _read_csv_safe("cash.csv")

    dfs = {"actuals": actuals, "budget": budget, "fx": fx, "cash": cash}
    # normalize columns
    for k in ("actuals","budget","fx","cash"):
        if dfs[k] is not None:
            dfs[k] = _normalize_columns(dfs[k])
    # periods
    for k in ("actuals","budget","cash"):
        df = dfs[k]
        if df is not None and "month" in df.columns:
            dfs[k]["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M")
    # standardize fx
    if dfs["fx"] is not None:
        months_needed = pd.Series(dtype="datetime64[ns]")
        for k in ("actuals", "budget", "cash"):
            df = dfs.get(k)
            if df is not None and "month" in df.columns:
                months_needed = pd.concat([months_needed, pd.to_datetime(df["month"].astype(str), errors="coerce")])
        dfs["fx"] = prepare_fx(dfs["fx"], months_needed)
    return dfs

def data_health() -> Dict[str, object]:
    dfs = load_data()
    out = {"ok": False}
    missing_files = [n for n in ("actuals", "budget", "fx", "cash") if dfs.get(n) is None]
    if missing_files:
        out["missing_files"] = missing_files
        out["ok"] = False
        return out

    a, b, fx = dfs["actuals"], dfs["budget"], dfs["fx"]
    # Check FX coverage by merging and counting NaNs (should be zero after prepare_fx)
    results = {}
    for dfname, df in (("actuals", a), ("budget", b)):
        if df is None or not {"month","currency"}.issubset(df.columns):
            results[f"missing_cols_{dfname}"] = True
            continue
        # Ensure both sides use the same month dtype (Period[M]) to avoid merge errors
        df_local = df.copy()
        fx_local = fx.copy()
        if "month" in df_local.columns:
            df_local["month"] = pd.to_datetime(df_local["month"], errors="coerce").dt.to_period("M")
        if "month" in fx_local.columns:
            fx_local["month"] = pd.to_datetime(fx_local["month"], errors="coerce").dt.to_period("M")
        merged = df_local.merge(fx_local, on=["month", "currency"], how="left")
        missing = int(merged["usd_rate"].isna().sum())
        results[f"missing_fx_rows_{dfname}"] = missing
    out.update(results)
    out["ok"] = all(v == 0 for k, v in results.items() if k.startswith("missing_fx_rows_"))
    return out

# ---------- core finance helpers ----------
def _to_usd(df: pd.DataFrame, fx: pd.DataFrame, amount_col: str, out_col: str) -> pd.DataFrame:
    if df is None or fx is None:
        return df
    fx = _standardize_fx(fx.copy())
    if not {"month", "currency"}.issubset(df.columns):
        return df
    # ensure string month keys to avoid mismatches and normalize currency
    df2 = df.copy()
    df2["_mstr"] = pd.to_datetime(df2["month"], errors="coerce").dt.to_period("M").astype(str)
    df2["_cur"] = df2["currency"].astype(str).str.strip().str.upper()
    fx2 = fx.copy()
    fx2["_mstr"] = fx2["month"].astype(str)
    fx2["_cur"] = fx2["currency"].astype(str).str.strip().str.upper()
    merged = df2.merge(fx2[["_mstr","_cur","usd_rate"]], on=["_mstr","_cur"], how="left")
    merged = merged.drop(columns=["_mstr","_cur"])  # keep original month/currency columns
    amt = amount_col if amount_col in merged.columns else ("amount" if "amount" in merged.columns else None)
    if amt is None:
        return merged
    merged[out_col] = merged[amt] * merged["usd_rate"]
    return merged

def _is_revenue(a: str) -> bool: return str(a).lower().startswith("revenue")
def _is_cogs(a: str)    -> bool: return str(a).lower().startswith("cogs")
def _is_opex(a: str)    -> bool: return str(a).lower().startswith("opex")

def _fmt_money(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "$0"
    return f"${x:,.0f}" if abs(x) >= 10000 else f"${x:,.2f}"

# ---------- tools ----------
def get_revenue_vs_budget(month: Optional[str]) -> Tuple[str, Optional[plt.Figure]]:
    dfs = load_data()
    actuals, budget, fx = dfs["actuals"], dfs["budget"], dfs["fx"]
    if actuals is None or budget is None or fx is None:
        return "Data not found. Ensure actuals.csv, budget.csv, and fx.csv exist in fixtures/.", None

    a = _to_usd(actuals, fx, "amount", "usd_amount")
    b = _to_usd(budget,  fx, "amount", "usd_amount")

    if month is None:
        return "Please specify a month, e.g., 'June 2025'.", None
    m = pd.Period(month, freq="M")
    a_m = a[a["month"] == m]
    b_m = b[b["month"] == m]

    a_col = _account_col(a_m)
    b_col = _account_col(b_m)
    if a_col is None or b_col is None:
        return "Data missing account column (expected 'account' or 'account_category').", None

    rev_actual = a_m[a_m[a_col].apply(_is_revenue)]["usd_amount"].sum()
    rev_budget = b_m[b_m[b_col].apply(_is_revenue)]["usd_amount"].sum()

    variance = rev_actual - rev_budget
    pct = (variance / rev_budget * 100.0) if rev_budget else np.nan

    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(["Actual", "Budget"], [rev_actual, rev_budget])
    ax.set_title(f"Revenue vs Budget — {m.strftime('%b %Y')}")
    ax.set_ylabel("USD")
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()

    text = (
        f"{m.strftime('%B %Y')} revenue was **{_fmt_money(rev_actual)}** "
        f"vs budget **{_fmt_money(rev_budget)}** "
        f"({('+' if variance>=0 else '')}{pct:.1f}% vs plan)." if not np.isnan(pct) else
        f"{m.strftime('%B %Y')} revenue was **{_fmt_money(rev_actual)}** (budget unavailable)."
    )
    return text, fig

def get_gross_margin_trend(last_n: int = 3) -> Tuple[str, Optional[plt.Figure]]:
    dfs = load_data()
    actuals, fx = dfs["actuals"], dfs["fx"]
    if actuals is None or fx is None:
        return "Data not found. Ensure actuals.csv and fx.csv exist in fixtures/.", None

    a = _to_usd(actuals, fx, "amount", "usd_amount")
    a_col = _account_col(a)
    if a_col is None:
        return "Data missing account column (expected 'account' or 'account_category').", None
    grp = a.groupby(["month", a_col])["usd_amount"].sum().reset_index()

    rev  = grp[grp[a_col].apply(_is_revenue)].groupby("month")["usd_amount"].sum()
    cogs = grp[grp[a_col].apply(_is_cogs)].groupby("month")["usd_amount"].sum()

    idx = sorted(set(rev.index).union(set(cogs.index)))
    rev  = rev.reindex(idx, fill_value=0.0)
    cogs = cogs.reindex(idx, fill_value=0.0)

    gm_pct = (rev - cogs) / rev.replace(0, np.nan) * 100.0
    gm_pct = gm_pct.dropna()

    if gm_pct.empty:
        return "Not enough data to compute Gross Margin %.", None

    tail = gm_pct.sort_index().tail(last_n)

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot([p.to_timestamp() for p in tail.index], tail.values, marker="o")
    ax.set_title(f"Gross Margin % — last {len(tail)} months")
    ax.set_ylabel("%")
    ax.set_xlabel("Month")
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()

    seq = " → ".join([f"{v:.1f}%" for v in tail.values])
    text = f"Gross Margin % last {len(tail)} months: **{seq}**."
    return text, fig

def get_opex_breakdown(month: Optional[str]) -> Tuple[str, Optional[plt.Figure]]:
    dfs = load_data()
    actuals, fx = dfs["actuals"], dfs["fx"]
    if actuals is None or fx is None:
        return "Data not found. Ensure actuals.csv and fx.csv exist in fixtures/.", None

    if month is None:
        return "Please specify a month, e.g., 'June 2025'.", None
    m = pd.Period(month, freq="M")

    a = _to_usd(actuals, fx, "amount", "usd_amount")
    a_m = a[a["month"] == m]
    a_col = _account_col(a_m)
    if a_col is None:
        return "Data missing account column (expected 'account' or 'account_category').", None
    opex_rows = a_m[a_m[a_col].apply(_is_opex)].copy()
    if opex_rows.empty:
        return f"No Opex data for {m.strftime('%b %Y')}.", None

    def _cat(x: str) -> str:
        x = str(x)
        if x.lower().startswith("opex:"):
            return x.split(":", 1)[1].strip() or "Opex"
        return "Opex"

    opex_rows["category"] = opex_rows[a_col].apply(_cat)
    agg = opex_rows.groupby("category")["usd_amount"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(agg.index.tolist(), agg.values.tolist())
    ax.set_title(f"Opex Breakdown — {m.strftime('%b %Y')}")
    ax.set_ylabel("USD")
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()

    total = agg.sum()
    parts = ", ".join([f"{k} {_fmt_money(v)}" for k, v in agg.items()])
    text = f"{m.strftime('%B %Y')} Opex: **{_fmt_money(total)}** ({parts})."
    return text, fig

def get_cash_runway() -> Tuple[str, Optional[plt.Figure]]:
    dfs = load_data()
    actuals, fx, cash = dfs["actuals"], dfs["fx"], dfs["cash"]
    if actuals is None or fx is None or cash is None:
        return "Data not found. Ensure actuals.csv, fx.csv, and cash.csv exist in fixtures/.", None

    a = _to_usd(actuals, fx, "amount", "usd_amount")
    # Normalize cash to have 'usd_amount' directly
    cash_norm = cash.copy()
    # ensure period month
    if "month" in cash_norm.columns:
        cash_norm["month"] = pd.to_datetime(cash_norm["month"], errors="coerce").dt.to_period("M")
    # prefer 'cash_usd' if present
    if "cash_usd" in cash_norm.columns:
        cash_usd = cash_norm.rename(columns={"cash_usd": "usd_amount"})
    else:
        # unify to 'amount' and convert with FX if currency present; otherwise assume USD
        if "amount" not in cash_norm.columns:
            # accept 'balance' as amount alias
            if "balance" in cash_norm.columns:
                cash_norm = cash_norm.rename(columns={"balance": "amount"})
        if "currency" in cash_norm.columns and "amount" in cash_norm.columns:
            cash_usd = _to_usd(cash_norm, fx, "amount", "usd_amount")
        elif "amount" in cash_norm.columns:
            cash_usd = cash_norm.copy()
            cash_usd["usd_amount"] = cash_usd["amount"].astype(float)
        else:
            return "cash.csv must include one of: cash_usd, amount, or balance.", None

    a_col = _account_col(a)
    if a_col is None:
        return "Data missing account column (expected 'account' or 'account_category').", None
    grp  = a.groupby(["month", a_col])["usd_amount"].sum().reset_index()
    rev  = grp[grp[a_col].apply(_is_revenue)].groupby("month")["usd_amount"].sum()
    cogs = grp[grp[a_col].apply(_is_cogs)].groupby("month")["usd_amount"].sum()
    opex = grp[grp[a_col].apply(_is_opex)].groupby("month")["usd_amount"].sum()

    idx = sorted(set(rev.index).union(set(cogs.index)).union(set(opex.index)))
    rev  = rev.reindex(idx, fill_value=0.0)
    cogs = cogs.reindex(idx, fill_value=0.0)
    opex = opex.reindex(idx, fill_value=0.0)

    net  = rev - cogs - opex
    burn = (-net).clip(lower=0.0)

    last3 = burn.sort_index().tail(3)
    avg_burn = last3.mean() if not last3.empty else np.nan

    latest_cash_row = cash_usd.sort_values("month").tail(1)
    if latest_cash_row.empty:
        return "No cash data available.", None
    latest_cash = float(latest_cash_row["usd_amount"].iloc[0])
    latest_month = latest_cash_row["month"].iloc[0]

    runway = (latest_cash / avg_burn) if (avg_burn is not None and not np.isnan(avg_burn) and avg_burn > 0) else np.nan

    cash_series = cash_usd.groupby("month")["usd_amount"].sum().sort_index()
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot([p.to_timestamp() for p in cash_series.index], cash_series.values, marker="o")
    ax.set_title("Cash Balance (USD)")
    ax.set_ylabel("USD")
    ax.set_xlabel("Month")
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()

    if np.isnan(avg_burn) or avg_burn is None:
        try:
            import pandas as _pd
            lm = latest_month.strftime('%b %Y') if not _pd.isna(latest_month) else "N/A"
        except Exception:
            lm = str(latest_month)
        text = (
            f"Latest cash ({lm}): **{_fmt_money(latest_cash)}**. "
            f"Not enough burn data to compute runway."
        )
    elif avg_burn <= 0:
        try:
            import pandas as _pd
            lm = latest_month.strftime('%b %Y') if not _pd.isna(latest_month) else "N/A"
        except Exception:
            lm = str(latest_month)
        text = (
            f"Latest cash ({lm}): **{_fmt_money(latest_cash)}**. "
            f"Profitable or zero net burn over the last 3 months — runway not applicable."
        )
    elif np.isnan(runway):
        try:
            import pandas as _pd
            lm = latest_month.strftime('%b %Y') if not _pd.isna(latest_month) else "N/A"
        except Exception:
            lm = str(latest_month)
        text = (
            f"Latest cash ({lm}): **{_fmt_money(latest_cash)}**. "
            f"Not enough burn data to compute runway."
        )
    else:
        text = (
            f"Cash runway: **{runway:.1f} months** "
            f"(cash **{_fmt_money(latest_cash)}**, avg net burn last 3 months **{_fmt_money(avg_burn)}**/mo)."
        )
    return text, fig

# ---------- PDF export ----------
def export_pdf(path: str, month_str: Optional[str] = None):
    """
    2-page Board Pack:
    - Page 1: Revenue vs Budget (month), GM% trend
    - Page 2: Opex breakdown (month), Cash trend + runway
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.pdfbase.pdfmetrics import stringWidth

    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT

    def _md_to_rml(text: str) -> str:
        """Convert lightweight markdown bold (**text**) into ReportLab inline <b> tags.

        This keeps the original text content but removes literal asterisks so the PDF text is selectable/copyable.
        """
        import re
        if not text:
            return ""
        # Replace **bold** with <b>bold</b>
        r = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        # Also handle single *...* as italic-ish fallback -> use <b> for simplicity
        r = re.sub(r"\*(.*?)\*", r"<b>\1</b>", r)
        return r

    def wrap_and_draw_text(c, text: str, x: float, y: float, max_width: float, font_name: str = "Helvetica", font_size: int = 10, leading: float = 12.0) -> float:
        """Render wrapped text using ReportLab Paragraph; returns bottom y after text."""
        if not text:
            return y
        # Create a simple Paragraph style
        ps = ParagraphStyle(
            name="body",
            fontName=font_name,
            fontSize=font_size,
            leading=leading,
            alignment=TA_LEFT,
        )
        rml = _md_to_rml(text)
        # Paragraph draws from its baseline; we need to wrap and get height
        p = Paragraph(rml, ps)
        # p.wrap returns (width, height)
        w, h = p.wrap(max_width, y)
        # draw onto canvas at x,y-h (ReportLab coordinates)
        p.drawOn(c, x, y - h)
        return y - h - (leading * 0.1)

    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER

    # ---- Document metadata & outline ----
    # Try to build a friendly title with month context when available
    disp_month = None
    try:
        if month_str:
            pm = pd.Period(month_str, freq="M")
            disp_month = pm.strftime('%b %Y')
    except Exception:
        disp_month = None

    doc_title = "Board Pack — FP&A Summary" + (f" ({disp_month})" if disp_month else "")
    try:
        c.setTitle(doc_title)
        c.setAuthor("CFO Copilot")
        c.setSubject("FP&A Board Pack")
        c.setCreator("CFO Copilot")
        c.setKeywords(["FP&A", "Finance", "Board Pack", "Summary"])  # optional
    except Exception:
        pass

    # Page 1 (fixed grid to avoid overlap)
    left = 1*inch
    right = width - 1*inch
    content_width = right - left
    top_margin = 0.7*inch
    bottom_margin = 0.5*inch
    chart_h = 2.7*inch

    text, fig = get_revenue_vs_budget(month_str)
    c.setFont("Helvetica-Bold", 14)
    title_y = height - top_margin
    c.drawString(left, title_y, "Board Pack — FP&A Summary")
    # Add a bookmark/outline for Page 1
    try:
        c.bookmarkPage("page1")
        c.addOutlineEntry("Summary & GM Trend", "page1", level=0, closed=False)
    except Exception:
        pass

    # Subtitle below title, then place chart directly after wrapped text
    sub_start_y = title_y - 0.2*inch
    sub_end_y = wrap_and_draw_text(c, text, left, sub_start_y, content_width, font_size=10, leading=12)

    # First chart immediately after subtitle block
    chart1_top = sub_end_y - 0.15*inch
    if fig:
        tmp_png = os.path.join(FIXTURES_DIR, "_tmp_chart1.png")
        fig.savefig(tmp_png, dpi=150, bbox_inches="tight")
        c.drawImage(tmp_png, left, chart1_top - chart_h, width=content_width, height=chart_h, preserveAspectRatio=True, mask='auto')
        try: os.remove(tmp_png)
        except: pass

    # GM text and chart directly after first chart
    gm_text_y = chart1_top - chart_h - 0.2*inch
    text2, fig2 = get_gross_margin_trend(3)
    gm_end_y = wrap_and_draw_text(c, text2, left, gm_text_y, content_width, font_size=10, leading=12)
    if fig2:
        tmp_png2 = os.path.join(FIXTURES_DIR, "_tmp_chart2.png")
        fig2.savefig(tmp_png2, dpi=150, bbox_inches="tight")
        gm_chart_top = gm_end_y - 0.15*inch
        c.drawImage(tmp_png2, left, gm_chart_top - chart_h, width=content_width, height=chart_h, preserveAspectRatio=True, mask='auto')
        try: os.remove(tmp_png2)
        except: pass

    c.showPage()
    # Add bookmark for Page 2
    try:
        c.bookmarkPage("page2")
        c.addOutlineEntry("Opex & Cash Overview", "page2", level=0, closed=False)
    except Exception:
        pass

    # Page 2 (fixed grid)
    text3, fig3 = get_opex_breakdown(month_str or None)
    c.setFont("Helvetica-Bold", 12)
    title2_y = height - top_margin
    c.drawString(left, title2_y, "Opex & Cash Overview")

    body2_y = title2_y - 0.2*inch
    body2_end_y = wrap_and_draw_text(c, text3, left, body2_y, content_width, font_size=10, leading=12)

    chart2_top = body2_end_y - 0.15*inch
    if fig3:
        tmp_png3 = os.path.join(FIXTURES_DIR, "_tmp_chart3.png")
        fig3.savefig(tmp_png3, dpi=150, bbox_inches="tight")
        # Fit to remaining space above bottom margin if needed
        draw_h3 = chart_h
        # Target bottom Y if drawn at full height
        target_y3 = chart2_top - chart_h
        if target_y3 < bottom_margin:
            # shrink to fit
            available_h = max(0.0, chart2_top - bottom_margin)
            # keep a minimum reasonable height
            draw_h3 = max(1.5*inch, available_h)
            target_y3 = chart2_top - draw_h3
        if draw_h3 > 0.25*inch:
            c.drawImage(tmp_png3, left, target_y3, width=content_width, height=draw_h3, preserveAspectRatio=True, mask='auto')
        try: os.remove(tmp_png3)
        except: pass

    text4, fig4 = get_cash_runway()
    cash_text_y = chart2_top - chart_h - 0.2*inch
    cash_end_y = wrap_and_draw_text(c, text4, left, cash_text_y, content_width, font_size=10, leading=12)
    if fig4:
        tmp_png4 = os.path.join(FIXTURES_DIR, "_tmp_chart4.png")
        fig4.savefig(tmp_png4, dpi=150, bbox_inches="tight")
        cash_chart_top = cash_end_y - 0.15*inch
        # Fit to remaining space above bottom margin; shrink if necessary
        draw_h4 = chart_h
        target_y4 = cash_chart_top - chart_h
        if target_y4 < bottom_margin:
            available_h = max(0.0, cash_chart_top - bottom_margin)
            # If there's not enough space to render at least 1.5in, try to shrink; if still not enough, start a new page
            if available_h < 1.0*inch:
                # new page for cash chart
                c.showPage()
                try:
                    c.bookmarkPage("page3")
                    c.addOutlineEntry("Cash (continued)", "page3", level=0, closed=False)
                except Exception:
                    pass
                # reset common layout vars for new page context
                left = 1*inch
                right = width - 1*inch
                content_width = right - left
                # Start near top of new page
                cash_chart_top = height - top_margin
                available_h = cash_chart_top - bottom_margin
            draw_h4 = max(1.5*inch, available_h)
            target_y4 = cash_chart_top - draw_h4
        if draw_h4 > 0.25*inch:
            c.drawImage(tmp_png4, left, target_y4, width=content_width, height=draw_h4, preserveAspectRatio=True, mask='auto')
        try: os.remove(tmp_png4)
        except: pass

    c.save()

def get_ebitda(month: Optional[str]) -> Tuple[str, Optional[plt.Figure]]:
    """
    EBITDA (proxy) = Revenue – COGS – Opex, reported in USD for the specified month.
    Returns a small bar chart (Revenue, COGS, Opex, EBITDA) for that month.
    """
    dfs = load_data()
    actuals, fx = dfs["actuals"], dfs["fx"]
    if actuals is None or fx is None:
        return "Data not found. Ensure actuals.csv and fx.csv exist in fixtures/.", None

    if month is None:
        return "Please specify a month, e.g., 'June 2025'.", None
    m = pd.Period(month, freq="M")

    a = _to_usd(actuals, fx, "amount", "usd_amount")
    a_m = a[a["month"] == m]
    a_col = _account_col(a_m)
    if a_col is None:
        return "Data missing account column (expected 'account' or 'account_category').", None
    grp = a_m.groupby(a_col)["usd_amount"].sum()

    rev = grp[[k for k in grp.index if _is_revenue(k)]].sum() if not grp.empty else 0.0
    cgs = grp[[k for k in grp.index if _is_cogs(k)]].sum() if not grp.empty else 0.0
    opx = grp[[k for k in grp.index if _is_opex(k)]].sum() if not grp.empty else 0.0
    ebitda = rev - cgs - opx

    fig, ax = plt.subplots(figsize=(6,3))
    labels = ["Revenue", "COGS", "Opex", "EBITDA"]
    values = [rev, -cgs, -opx, ebitda]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4"]
    ax.bar(labels, values, color=colors)
    ax.set_title(f"EBITDA — {m.strftime('%b %Y')}")
    ax.set_ylabel("USD")
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()

    text = (
        f"{m.strftime('%B %Y')} EBITDA: **{_fmt_money(ebitda)}** "
        f"(Revenue {_fmt_money(rev)} − COGS {_fmt_money(cgs)} − Opex {_fmt_money(opx)})."
    )
    return text, fig
