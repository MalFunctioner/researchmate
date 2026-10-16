"""ResearchMate - Streamlit frontend.  Run with:  streamlit run app.py"""
import streamlit as st

import config
import memory
from agent import ResearchAgent, build_final_report
from export import report_to_pdf

st.set_page_config(page_title="ResearchMate", page_icon="🔬", layout="wide")

NODE_LABELS = {
    "planner": "Planning research",
    "researcher": "Searching the web",
    "analyst": "Analysing sources",
    "writer": "Writing report",
    "reviewer": "Reviewing draft",
}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔬 ResearchMate")
    st.caption("Autonomous AI research assistant agent")

    with st.expander("⚙️ Settings", expanded=not (config.GOOGLE_API_KEY and config.TAVILY_API_KEY)):
        g_key = st.text_input("Gemini API key", value=config.GOOGLE_API_KEY, type="password")
        t_key = st.text_input("Tavily API key", value=config.TAVILY_API_KEY, type="password")
        model = st.text_input("Gemini model", value=config.GEMINI_MODEL)
        n_q = st.slider("Sub-questions", 2, 6, config.MAX_SUB_QUESTIONS)
        loops = st.slider("Max review loops", 1, 3, config.MAX_REVIEW_LOOPS)

    st.divider()
    st.subheader("📚 History")
    rows = memory.list_reports()
    if not rows:
        st.caption("No reports yet.")
    for rid, topic, n_src, created in rows[:15]:
        if st.button(f"{topic[:40]}", key=f"h{rid}", help=f"{created} · {n_src} sources",
                     use_container_width=True):
            st.session_state["view_id"] = rid
            st.session_state.pop("result", None)

# ---------------------------------------------------------------- main
st.header("What do you want to research?")
topic = st.text_input(
    "Research topic",
    placeholder="e.g. Impact of electric vehicles on urban air quality",
    label_visibility="collapsed",
)
go = st.button("🚀 Start research", type="primary", disabled=not topic.strip())


def show_report(topic_: str, report: str, key: str):
    st.markdown(report)
    c1, c2 = st.columns(2)
    fname = "".join(ch if ch.isalnum() else "_" for ch in topic_)[:40] or "report"
    c1.download_button("⬇️ Download Markdown", report, f"{fname}.md", "text/markdown",
                       key=f"md{key}", use_container_width=True)
    try:
        c2.download_button("⬇️ Download PDF", report_to_pdf(report), f"{fname}.pdf",
                           "application/pdf", key=f"pdf{key}", use_container_width=True)
    except Exception as e:
        c2.caption(f"PDF export unavailable: {e}")


if go:
    if not g_key or not t_key:
        st.error("Please add your Gemini and Tavily API keys in ⚙️ Settings (sidebar).")
        st.stop()
    config.GOOGLE_API_KEY, config.TAVILY_API_KEY, config.GEMINI_MODEL = g_key, t_key, model
    st.session_state.pop("view_id", None)

    import tools
    from agent import default_llm

    agent = ResearchAgent(llm=default_llm(),
                          search_fn=lambda q: tools.web_search(q),
                          max_sub_questions=n_q, max_loops=loops)
    final_state: dict = {}
    with st.status("🤖 Agent is working…", expanded=True) as status:
        try:
            for node, update in agent.stream(topic.strip()):
                status.update(label=f"🤖 {NODE_LABELS.get(node, node)}…")
                for line in update.get("log", []):
                    st.markdown(line)
                final_state.update({k: v for k, v in update.items() if k != "log"})
        except Exception as e:
            status.update(label="❌ Agent stopped with an error", state="error")
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                st.error(
                    "**Gemini free-tier quota reached.** The free tier allows a limited number of "
                    "requests per day *per model*. Either wait for it to reset, or switch to another "
                    "Gemini model in ⚙️ Settings (each model has its own quota)."
                )
            elif "NOT_FOUND" in msg and "model" in msg.lower():
                st.error("**Model not available.** Change the Gemini model name in ⚙️ Settings.")
            else:
                st.exception(e)
            st.stop()
        status.update(label="✅ Research complete", state="complete", expanded=False)

    sources = final_state.get("sources", [])
    report = build_final_report(final_state.get("draft", ""), sources)
    memory.save_report(topic.strip(), report, len(sources))
    st.session_state["result"] = (topic.strip(), report, len(sources))
    st.rerun()  # refresh sidebar history

if "result" in st.session_state:
    t, report, n = st.session_state["result"]
    st.success(f"Report ready · {n} sources consulted")
    show_report(t, report, "new")
elif "view_id" in st.session_state:
    row = memory.get_report(st.session_state["view_id"])
    if row:
        t, report, created = row
        st.caption(f"Saved report · {created}")
        show_report(t, report, f"h{st.session_state['view_id']}")
        if st.button("🗑️ Delete this report"):
            memory.delete_report(st.session_state.pop("view_id"))
            st.rerun()
else:
    st.info(
        "**How it works:** the agent plans sub-questions → searches the web → reads and "
        "extracts notes into memory → writes a cited report → reviews its own draft and "
        "researches more if it finds gaps."
    )
