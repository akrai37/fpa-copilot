
import streamlit as st
from agent.planner import route_query
from agent import tools
import tempfile, os

st.set_page_config(page_title="CFO Copilot", page_icon="📊", layout="centered")
st.title("📊 CFO Copilot — Mini FP&A Agent")

with st.sidebar:
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
    st.header("Export")
    month_for_pdf = st.text_input("Month for PDF (e.g., June 2025)", value="")
    if st.button("Export Board PDF"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tools.export_pdf(tmp.name, month_for_pdf or None)
        st.success("PDF generated.")
        st.download_button("Download PDF", data=open(tmp.name, "rb").read(), file_name="board_pack.pdf", mime="application/pdf")
        os.unlink(tmp.name)

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

query = st.chat_input("Type your question...")
if query:
    st.session_state.history.append(("user", query))
    intent, params = route_query(query)

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

    st.session_state.history.append(("assistant", text, fig))

for item in st.session_state.history:
    role = item[0]
    if role == "user":
        with st.chat_message("user"):
            st.write(item[1])
    else:
        with st.chat_message("assistant"):
            st.write(item[1])
            if len(item) > 2 and item[2] is not None:
                st.pyplot(item[2])
