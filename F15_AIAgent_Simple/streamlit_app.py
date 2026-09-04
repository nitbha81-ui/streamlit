"""
Streamlit chat UI for the Simple AI Agent (RAG + Pinecone + MCP tools).

Run with:
    streamlit run streamlit_app.py

It reuses the agent-building logic from AIagent_simple.py so the same
Claude + Pinecone + tools stack powers both the CLI and the web UI.
"""

import uuid

import streamlit as st

from AIagent_simple import build_agent, ask_agent

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Simple AI Agent",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Simple AI Agent")
st.caption("Claude + Pinecone RAG + Tools (Calculator · DateTime · Knowledge Retriever)")


# ------------------------------------------------------------------
# Build the agent once and cache it across reruns
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing agent (Claude + Pinecone)...")
def get_agent():
    """Build the agent a single time per Streamlit session/server."""
    return build_agent()


# A stable thread id so the agent keeps conversation memory for this session.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "This agent can:\n"
        "- Answer facts from a **Pinecone** knowledge base (RAG)\n"
        "- Do math with the **Calculator** tool\n"
        "- Tell the current **date & time**\n"
    )
    st.markdown("**Try asking:**")
    st.markdown(
        "- What is the capital of India?\n"
        "- Calculate sqrt(144)\n"
        "- What time is it?\n"
    )
    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
        st.rerun()


# ------------------------------------------------------------------
# Load the agent (shows an error nicely if keys are missing)
# ------------------------------------------------------------------
try:
    agent = get_agent()
except Exception as e:  # noqa: BLE001
    st.error(f"Failed to initialize the agent: {e}")
    st.info("Check that ANTHROPIC_API_KEY and PINECONE_API_KEY are set in your .env file.")
    st.stop()


# ------------------------------------------------------------------
# Render existing chat history
# ------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ------------------------------------------------------------------
# Handle new user input
# ------------------------------------------------------------------
if prompt := st.chat_input("Ask me anything..."):
    # Show and store the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the agent's answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_agent(agent, prompt, thread_id=st.session_state.thread_id)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
