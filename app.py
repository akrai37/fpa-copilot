import streamlit as st
from agent.planner import route_query, _normalize_month
from agent import tools
import re
import unicodedata
import html as _html
import tempfile, os
import pandas as pd

st.set_page_config(page_title="CFO Copilot", page_icon="📊")
st.title("📊 CFO Copilot — Mini FP&A Agent")

# Clear chat action
def _clear_chat():
    if "history" in st.session_state:
        st.session_state.history = []

# (no URL query param behavior) clear action is triggered via the sidebar button below

# (no URL query param behavior) clear action is triggered via the sidebar button below

with st.sidebar:
    # Render Clear button directly (avoid columns that can cause flex/wrap in narrow sidebars)
    st.button("Clear", on_click=_clear_chat, key="clear_chat")

    # add some space before the next section
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.header("Data Health")
    health = tools.data_health()

    # human-friendly summary + JSON view
    if isinstance(health, dict):
        if health.get("ok"):
            st.success("✅ Data health: OK — FX coverage looks good.")
        else:
            if health.get("missing_files"):
                st.error("❌ Missing files: " + ", ".join(health.get("missing_files")))
            # show any missing fx rows per source
            for k, v in health.items():
                if k.startswith("missing_fx_rows_") and v:
                    src = k.replace("missing_fx_rows_", "")
                    st.warning(f"⚠️ {src}: {v} rows without FX rate")
        with st.expander("Details", expanded=False):
            st.json(health)
    else:
        # older string message
        st.write(health)

    st.divider()
    # Debugging helpers for RAG
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    show_rag_debug = st.checkbox("Show routing & retrieval debug", value=False)
    # Option: if enabled, month-only queries (e.g., "June") will default to Revenue vs Budget
    prefer_month_default = st.checkbox("Auto-assume month-only queries as 'Revenue vs Budget'", value=False)
    # Add extra top spacing so the Export section is visually separated from the controls above
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.header("Export")
    month_for_pdf = st.text_input("Month for PDF (e.g., June 2025)", value="")
    if st.button("Export Board PDF"):
        # Normalize month input (accepts 'June', '2025-06', 'Jun 2025', etc.).
        month_input = (month_for_pdf or "").strip()
        norm_month = None
        if month_input:
            try:
                ts = pd.to_datetime(month_input, errors="coerce")
                if ts is not None and not pd.isna(ts):
                    norm_month = str(pd.Period(ts, freq="M"))  # 'YYYY-MM'
            except Exception:
                norm_month = None

        # If user typed something we couldn't parse, show a friendly error
        if month_input and norm_month is None:
            st.error("Invalid month. Please enter like 'June 2025' or '2025-06'.")
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            try:
                tools.export_pdf(tmp.name, norm_month)
                st.success("PDF generated.")
                st.download_button(
                    "Download PDF",
                    data=open(tmp.name, "rb").read(),
                    file_name="board_pack.pdf",
                    mime="application/pdf",
                )
            except Exception:
                st.error("Couldn't generate PDF. Please try again with a valid month like 'June 2025'.")
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    # ...sidebar continues (other items kept)

    # (Clear chat button moved to top) -- removed bottom spacer/button

st.markdown("""
Ask about **Revenue vs Budget**, **Gross Margin % trend**, **Opex breakdown**, or **Cash runway**.

**Examples**
- What was June 2025 revenue vs budget in USD?
- Show Gross Margin % trend for the last 3 months.
- Break down Opex by category for June 2025.
- What is our cash runway right now?
""")

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_intent" not in st.session_state:
    st.session_state.pending_intent = None
    st.session_state.pending_params = {}

query = st.chat_input("Type your question...")
if query:
    st.session_state.history.append(("user", query))
    intent, params = route_query(query)

    # Helper: intents that need a month
    def _needs_month(i: str) -> bool:
        return i in {"revenue_vs_budget", "opex_breakdown", "ebitda"}

    # Parse month from the raw query (month-only follow-ups like "June 2025")
    try:
        parsed_month_inline = _normalize_month(query)
    except Exception:
        parsed_month_inline = None

    # If we previously asked for a month (pending intent) and the user sent just a month,
    # fulfill the pending intent directly without disambiguation.
    if intent == "fallback" and st.session_state.get("pending_intent") and parsed_month_inline:
        intent = st.session_state.pending_intent
        params = dict(st.session_state.pending_params or {})
        if params.get("month") is None:
            params["month"] = parsed_month_inline
        # clear pending
        st.session_state.pending_intent = None
        st.session_state.pending_params = {}

    # If routing fell back and the user typed a month (e.g., 'June 2025'), offer quick choices
    # Only show the disambiguation UI when there is no pending intent to fulfill.
    if intent == "fallback":
        parsed_month = parsed_month_inline
        if parsed_month:
            if not st.session_state.get("pending_intent"):
                # If user prefers auto-default, route month-only queries directly to Revenue vs Budget
                if 'prefer_month_default' in locals() and prefer_month_default:
                    # announce the auto-selection and set intent accordingly
                    st.session_state.history.append(("assistant", f"I detected month: {parsed_month}. Auto-selecting Revenue vs Budget as your default.", None, "disambiguation"))
                    intent, params = "revenue_vs_budget", {"month": parsed_month}
                else:
                    # Format a friendly label alongside the normalized month
                    try:
                        _disp_month = pd.Period(parsed_month, freq="M").strftime('%B %Y')
                    except Exception:
                        _disp_month = parsed_month
                    # show quick choice buttons to disambiguate what the user wants for that month (use normalized YYYY-MM display)
                    st.session_state.history.append(("assistant", f"I detected month: {parsed_month} ({_disp_month}). What would you like to see for {parsed_month}?", None, "disambiguation"))
                # store the parsed month for reference; no quick-action buttons are rendered
                st.session_state._parsed_month = parsed_month

    # If user wants debug info, attach RAG retrieval hints into the session (non-sensitive)
    try:
        if 'show_rag_debug' in locals() and show_rag_debug:
            from agent import rag as _rag
            retrieved = _rag.retrieve(query, top_k=5)
            # store in session_state to render with assistant message
            st.session_state.last_retrieved = retrieved
        else:
            st.session_state.last_retrieved = []
    except Exception:
        st.session_state.last_retrieved = []

    # If this intent needs a month and it's missing, remember it as pending so a month-only follow-up fulfills it.
    if intent != "fallback" and _needs_month(intent) and not params.get("month"):
        st.session_state.pending_intent = intent
        st.session_state.pending_params = dict(params)

    # Determine whether to proceed with generating a response
    proceed_response = intent != "fallback"

    if proceed_response:
        if intent == "revenue_vs_budget":
            text, fig = tools.get_revenue_vs_budget(month=params.get("month"))
        elif intent == "gross_margin_trend":
            text, fig = tools.get_gross_margin_trend(last_n=params.get("last_n", 3))
        elif intent == "opex_breakdown":
            text, fig = tools.get_opex_breakdown(month=params.get("month"))
        elif intent == "cash_runway":
            text, fig = tools.get_cash_runway()
        elif intent == "ebitda":
            text, fig = tools.get_ebitda(month=params.get("month"))
        else:
            text, fig = ("I can answer: Revenue vs Budget, Gross Margin % trend, Opex breakdown, EBITDA, and Cash runway.", None)

        st.session_state.history.append(("assistant", text, fig, intent))

for item in st.session_state.history:
    role = item[0]
    if role == "user":
        with st.chat_message("user"):
            st.write(item[1])
    else:
        with st.chat_message("assistant"):
            # Render structured metrics first (clean, no markdown), then a plain sentence.
            intent = item[3] if len(item) > 3 else None

            def _strip_md(s: str) -> str:
                s2 = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
                s2 = re.sub(r"\*(.*?)\*", r"\1", s2)
                return s2

            def _sanitize_text(s: str) -> str:
                if not isinstance(s, str):
                    return s
                # Normalize unicode; remove zero-width characters; replace unicode asterisk
                s2 = unicodedata.normalize("NFKC", s)
                for zw in [
                    "\u200b",  # zero width space
                    "\u2009",  # thin space
                    "\u200a",  # hair space
                    "\u200c",  # zero width non-joiner
                    "\u200d",  # zero width joiner
                    "\u202f",  # narrow no-break space
                    "\u00a0",  # non-breaking space
                    "\u2060",  # word joiner
                    "\u180e",  # mongolian vowel separator (deprecated)
                    "\ufeff",  # BOM
                ]:
                    s2 = s2.replace(zw, "")
                s2 = s2.replace("∗", "*")
                return s2

            def _metrics_revenue_vs_budget(text: str):
                m = re.findall(r"\$([0-9,]+(?:\.\d{2})?)", text)
                p = re.search(r"([+-]?[0-9]+(?:\.[0-9]+)?)%", text)
                if len(m) >= 2:
                    col1, col2, col3 = st.columns(3)
                    def _small_metric(col, label, val, delta=None):
                        # Render a compact metric: label (small), value (moderate), optional delta (small)
                        with col:
                            st.markdown(f"<div style='font-size:12px;color:#6c757d'>{label}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size:18px;font-weight:600'>${val}</div>", unsafe_allow_html=True)
                            if delta is not None:
                                st.markdown(f"<div style='font-size:12px;color:#6c757d'>{delta}</div>", unsafe_allow_html=True)

                    _small_metric(col1, "Revenue (USD)", m[0])
                    _small_metric(col2, "Budget (USD)", m[1])
                    if p:
                        pct = p.group(1)
                        _small_metric(col3, "Var vs plan", f"{pct}%")

            def _metrics_ebitda(text: str):
                vals = re.findall(r"\$([0-9,]+(?:\.\d{2})?)", text)
                # order expected: EBITDA, Revenue, COGS, Opex
                if len(vals) >= 4:
                    ebitda, rev, cogs, opex = vals[0], vals[1], vals[2], vals[3]
                    col1, col2, col3, col4 = st.columns(4)
                    def _sm(col, label, val):
                        st.markdown(f"<div style='font-size:12px;color:#6c757d'>{label}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:18px;font-weight:600'>${val}</div>", unsafe_allow_html=True)
                    _sm(col1, "EBITDA", ebitda)
                    _sm(col2, "Revenue", rev)
                    _sm(col3, "COGS", cogs)
                    _sm(col4, "Opex", opex)

            def _metrics_cash_runway(text: str):
                runway = re.search(r"runway:\s*\*\*?([0-9]+(?:\.[0-9]+)?)\s*months\*\*?", text, re.IGNORECASE)
                cash = re.search(r"cash\s*\*\*?\$([0-9,]+)\*\*?", text, re.IGNORECASE)
                burn = re.search(r"burn.*?\*\*?\$([0-9,]+)\*\*?/mo", text, re.IGNORECASE)
                col1, col2, col3 = st.columns(3)
                def _sm_small(col, label, val):
                    with col:
                        st.markdown(f"<div style='font-size:12px;color:#6c757d'>{label}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:18px;font-weight:600'>{val}</div>", unsafe_allow_html=True)

                if runway:
                    _sm_small(col1, "Runway (months)", runway.group(1))
                if cash:
                    _sm_small(col2, "Cash", f"${cash.group(1)}")
                if burn:
                    _sm_small(col3, "Avg burn/mo", f"${burn.group(1)}")

            def _render_summary_code(s: str, pad: bool = True):
                # Revert to the original Streamlit code block rendering
                txt = _sanitize_text(_strip_md(s)) if isinstance(s, str) else s
                if pad:
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.code(txt)

            if intent == "revenue_vs_budget":
                _metrics_revenue_vs_budget(item[1])
                # Custom clean summary for revenue vs budget in a monospace code block
                stripped = _sanitize_text(_strip_md(item[1]))
                m = re.findall(r"\$([0-9,]+(?:\.\d{2})?)", stripped)
                p = re.search(r"([+-]?[0-9]+(?:\.[0-9]+)?)%", stripped)
                if len(m) >= 2 and p:
                    month_text = re.search(r"^([A-Za-z]+ \d{4})", stripped)
                    month = month_text.group(1) if month_text else "This month"
                    line = f"{month} revenue was ${m[0]} vs budget ${m[1]} ({p.group(1)}% vs plan)"
                    # Add a small padded container above the code block for breathing room
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    st.code(line)
                else:
                    st.code(stripped)
            elif intent == "ebitda":
                _metrics_ebitda(item[1])
                # Custom clean summary for EBITDA in monospace
                stripped = _sanitize_text(_strip_md(item[1]))
                vals = re.findall(r"\$([0-9,]+(?:\.\d{2})?)", stripped)
                if len(vals) >= 4:
                    month_text = re.search(r"^([A-Za-z]+ \d{4})", stripped)
                    month = month_text.group(1) if month_text else "This month"
                    line = f"{month} EBITDA: ${vals[0]} (Revenue ${vals[1]} - COGS ${vals[2]} - Opex ${vals[3]})"
                    _render_summary_code(line)
                else:
                    _render_summary_code(stripped)
            elif intent == "cash_runway":
                _metrics_cash_runway(item[1])
                # Custom clean summary for Cash Runway in monospace
                stripped = _sanitize_text(_strip_md(item[1]))
                vals = re.findall(r"(\d+(?:\.\d+)?)\s*months?", stripped)
                if vals:
                    line = f"Cash Runway: {vals[0]} months"
                    _render_summary_code(line)
                else:
                    _render_summary_code(stripped)
            elif intent == "gross_margin_trend":
                # Clean summary for GM trend (monospace)
                _render_summary_code(f"Gross margin trend: {_sanitize_text(_strip_md(item[1]))}")
            elif intent == "opex_breakdown":
                # Render Opex in a terminal/Jupyter-like monospaced box
                def _render_opex_code_block(raw: str):
                    stripped = _sanitize_text(_strip_md(raw))
                    # Collapse stray newlines/CRs that can appear inside amounts (e.g. "R&D $2\n,246,400")
                    # This keeps the parsing logic robust and prevents character-by-character wrapping in the UI.
                    stripped = stripped.replace('\r', '').replace('\n', ' ')
                    # Try to parse: Month, Total Opex, and category amounts from parentheses
                    month_m = re.search(r"^([A-Za-z]+\s+\d{4})", stripped)
                    total_m = re.search(r"Opex:\s*\$([0-9,]+(?:\.\d{2})?)", stripped, re.IGNORECASE)
                    cats_m = re.search(r"\(([^\)]+)\)", stripped)

                    try:
                        month = month_m.group(1) if month_m else "Opex"
                        total = total_m.group(1) if total_m else None
                        pairs = []
                        if cats_m:
                            inner = cats_m.group(1)
                            # Find all occurrences of "Name $amount" without splitting by commas in the amount
                            for m in re.finditer(r"([A-Za-z0-9 &/\-]+)\s*\$([0-9][0-9,]*(?:\.\d{2})?)", inner):
                                name = m.group(1).strip()
                                amt = m.group(2)
                                pairs.append((name, amt))

                        # If parsing fails, just show sanitized text in styled code block
                        if not total and not pairs:
                            _render_summary_code(stripped)
                            return

                        # Build aligned output
                        lines = [f"{month} Opex"]
                        if total:
                            lines.append(f"Total: ${total}")
                        if pairs:
                            lines.append("")
                            w = max(len(name) for name, _ in pairs)
                            for name, amt in pairs:
                                lines.append(f"{name.ljust(w)}  ${amt}")
                        _render_summary_code("\n".join(lines))
                    except Exception:
                        # Fallback to a styled code block
                        _render_summary_code(stripped)

                _render_opex_code_block(item[1])
            else:
                # Fallback for other intents: render monospace summary
                _render_summary_code(item[1])

            # Render RAG debug cards (if present and enabled)
            if show_rag_debug and st.session_state.get("last_retrieved"):
                with st.expander("Retrieved docs (RAG)"):
                    for d in st.session_state.get("last_retrieved", []):
                        st.write(f"- {d.get('title')} (id={d.get('id')}, score={d.get('score'):.3f})")
                        st.write(d.get('text')[:500] + ("..." if len(d.get('text',''))>500 else ""))

            if len(item) > 2 and item[2] is not None:
                st.pyplot(item[2])
