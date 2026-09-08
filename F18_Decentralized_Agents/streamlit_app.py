"""Streamlit UI for bhaleraostudios (decentralized multi-agent system)."""

import time
import streamlit as st

from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent
from agents.analyzer_agent import AnalyzerAgent
from core.message_bus import bus

st.set_page_config(
    page_title="bhaleraostudios",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling: clean, minimal chat look inspired by Claude / ChatGPT / DeepSeek.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      /* Sidebar background - soft neutral tone */
      section[data-testid="stSidebar"] {
          background-color: #f7f7f8;
          border-right: 1px solid #ececf1;
      }
      section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

      /* Brand mark in the sidebar */
      .brand {
          font-size: 1.15rem;
          font-weight: 700;
          color: #1f2937;
          padding: 0.2rem 0.4rem 0.9rem 0.4rem;
          letter-spacing: -0.01em;
      }
      .brand span { color: #6d5efc; }

      /* "New chat" button styled like a primary pill */
      section[data-testid="stSidebar"] div.stButton > button {
          width: 100%;
          border-radius: 10px;
          border: 1px solid #e2e2ea;
          background: #ffffff;
          color: #1f2937;
          font-weight: 600;
          padding: 0.55rem 0.75rem;
          transition: all 0.15s ease-in-out;
      }
      section[data-testid="stSidebar"] div.stButton > button:hover {
          border-color: #6d5efc;
          color: #6d5efc;
      }

      .history-label {
          font-size: 0.72rem;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: #9096a2;
          margin: 1.1rem 0.4rem 0.4rem 0.4rem;
      }
      .history-item {
          font-size: 0.9rem;
          color: #3f4451;
          padding: 0.45rem 0.55rem;
          border-radius: 8px;
          margin-bottom: 0.15rem;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
      }
      .history-item:hover { background: #ececf1; }

      /* Center and constrain the main chat column for readability */
      .main .block-container {
          max-width: 820px;
          padding-top: 2.2rem;
      }

      .app-title {
          font-size: 1.9rem;
          font-weight: 700;
          color: #1f2937;
          letter-spacing: -0.02em;
      }
      .app-subtitle { color: #9096a2; font-size: 0.9rem; margin-top: -0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Agents (created once, cached across reruns).
# ---------------------------------------------------------------------------
@st.cache_resource
def init_agents():
    researcher = ResearcherAgent()
    writer = WriterAgent()
    analyzer = AnalyzerAgent()
    return researcher, writer, analyzer


researcher, writer, analyzer = init_agents()

# ---------------------------------------------------------------------------
# Session state.
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    # Titles of previous chats (most recent first).
    st.session_state.history = []


def start_new_chat():
    """Archive the current conversation title and clear the chat."""
    if st.session_state.messages:
        first_user = next(
            (m["content"] for m in st.session_state.messages if m["role"] == "user"),
            None,
        )
        if first_user:
            st.session_state.history.insert(0, first_user[:40])
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar: minimal, chat-app style (no agent status).
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">bhalerao<span>studios</span></div>', unsafe_allow_html=True)

    st.button("➕  New chat", on_click=start_new_chat)

    if st.session_state.history:
        st.markdown('<div class="history-label">Recent</div>', unsafe_allow_html=True)
        for title in st.session_state.history[:15]:
            st.markdown(f'<div class="history-item">💬 {title}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main chat area.
# ---------------------------------------------------------------------------
st.markdown('<div class="app-title">bhaleraostudios</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Ask anything.</div>', unsafe_allow_html=True)
st.write("")

# Render existing conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input.
if prompt := st.chat_input("Message bhaleraostudios..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Fresh run each turn.
            bus.messages.clear()
            researcher.publish(
                "query.request",
                {"type": "research_request", "query": prompt},
            )
            time.sleep(1)

            # bus.get_messages returns items shaped as {"topic": ..., "message": {...}},
            # so the actual payload lives under the "message" key.
            final_answer = None
            for record in bus.get_messages("answer.final"):
                payload = record.get("message", {})
                if payload.get("type") == "final_answer":
                    final_answer = payload

            if final_answer:
                answer = final_answer["answer"]
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            else:
                error_msg = "⚠️ Sorry, I couldn't generate an answer. Please try again."
                st.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
