# ==========================================
# STREAMLIT APP - Multi-Agent System
# Architecture: Layered (Reactive + Deliberative + BDI)
# Patterns: ReAct | Tool Use | Reflection | CoALA
# LLM: Claude Haiku 4.5  |  Memory: Pinecone
# ==========================================

import time
import streamlit as st

# --- Page Config (must be first Streamlit call) ---
st.set_page_config(
    page_title="Multi-Agent System",
    page_icon="🤖",
    layout="wide",
)


# ------------------------------------------
# Lazy backend loader
# ------------------------------------------
# The backend initializes Claude + Pinecone and builds a vector index on import,
# which can take a while. We cache it so it only runs once per Streamlit session.
@st.cache_resource(show_spinner="🔌 Initializing agents, Claude & Pinecone...")
def load_backend():
    """Import and return the backend module.

    Importing multiagent_backend triggers LLM + Pinecone setup. Because the
    interactive CLI loop is guarded behind `if __name__ == '__main__'`, importing
    here is safe and does not block.
    """
    import multiagent_backend as backend
    return backend


def get_backend():
    """Return the backend module, capturing any initialization error."""
    try:
        return load_backend(), None
    except Exception as e:  # noqa: BLE001 - surface any setup error to the UI
        return None, str(e)


backend, backend_error = get_backend()


# ------------------------------------------
# Header
# ------------------------------------------
st.title("🤖 Multi-Agent Research System")
st.caption(
    "Researcher → Writer → Analyzer, orchestrated by a Supervisor "
    "(ReAct · Tool Use · Reflection · CoALA)"
)


# ------------------------------------------
# Sidebar
# ------------------------------------------
with st.sidebar:
    st.header("📐 Architecture")
    st.markdown(
        """
        **Layered (Hybrid)**
        - **Reactive**: Web Search, Pinecone retrieval
        - **Deliberative**: Supervisor orchestration
        - **BDI**: Researcher → Writer → Analyzer
        """
    )

    st.divider()

    st.header("🧠 Design Patterns")
    st.markdown(
        """
        | Pattern | Role |
        |---------|------|
        | **ReAct** | Thought → Action → Observation |
        | **Tool Use** | Pinecone, Web, Summarize, FactCheck |
        | **Reflection** | Analyzer self-corrects |
        | **CoALA** | Reasoning + Memory + Tools |
        """
    )

    st.divider()

    st.subheader("🔌 System Status")
    if backend_error:
        st.error("❌ Backend failed to initialize")
        with st.expander("Error details"):
            st.code(backend_error)
    else:
        st.success("✅ Claude Haiku 4.5")
        st.success(f"✅ Pinecone: {backend.PINECONE_INDEX}")

    st.divider()

    st.subheader("⚙️ Options")
    show_stages = st.toggle("Show agent-by-agent breakdown", value=True)

    if st.button("🔄 Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ------------------------------------------
# Session State
# ------------------------------------------
if "history" not in st.session_state:
    # Each item: {"query", "research", "draft", "analysis", "final_answer"}
    st.session_state.history = []


# ------------------------------------------
# Render prior conversation
# ------------------------------------------
def render_result(result: dict, show_stages: bool):
    """Render a single multi-agent result inside the chat area."""
    with st.chat_message("user"):
        st.markdown(result["query"])

    with st.chat_message("assistant"):
        st.markdown("### ✅ Final Answer")
        st.markdown(result["final_answer"])

        if show_stages:
            with st.expander("🔍 Researcher Agent — findings"):
                st.markdown(result["research"])
            with st.expander("✍️ Writer Agent — draft"):
                st.markdown(result["draft"])
            with st.expander("🔬 Analyzer Agent — reflection"):
                st.markdown(result["analysis"])


for past in st.session_state.history:
    render_result(past, show_stages)


# ------------------------------------------
# Input + run pipeline
# ------------------------------------------
prompt = st.chat_input(
    "Ask the agents something (e.g. 'What are the benefits of Ashwagandha coffee?')",
    disabled=backend_error is not None,
)

if backend_error and not st.session_state.history:
    st.info(
        "The multi-agent backend could not start. Check that ANTHROPIC_API_KEY "
        "and PINECONE_API_KEY are set in your `.env`, then reload."
    )

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Running the multi-agent pipeline...", expanded=True)
        try:
            status.update(label="🔍 Researcher Agent working (ReAct)...", state="running")
            result = backend.run_multi_agent(prompt)
            status.update(label="✅ Pipeline complete", state="complete", expanded=False)

            st.markdown("### ✅ Final Answer")
            st.markdown(result["final_answer"])

            if show_stages:
                with st.expander("🔍 Researcher Agent — findings"):
                    st.markdown(result["research"])
                with st.expander("✍️ Writer Agent — draft"):
                    st.markdown(result["draft"])
                with st.expander("🔬 Analyzer Agent — reflection"):
                    st.markdown(result["analysis"])

            st.session_state.history.append(result)
        except Exception as e:  # noqa: BLE001 - show runtime errors to the user
            status.update(label="❌ Pipeline failed", state="error")
            st.error(f"Something went wrong while running the agents:\n\n{e}")
