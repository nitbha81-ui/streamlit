"""Streamlit UI for Decentralized Multi-Agent System."""

import streamlit as st
import time
from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent
from agents.analyzer_agent import AnalyzerAgent
from core.message_bus import bus
import threading

st.set_page_config(
    page_title="Decentralized Multi-Agent System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Decentralized Multi-Agent System")
st.caption("Researcher → Writer → Analyzer (No Central Supervisor)")

# Initialize agents
@st.cache_resource
def init_agents():
    researcher = ResearcherAgent()
    writer = WriterAgent()
    analyzer = AnalyzerAgent()
    return researcher, writer, analyzer

researcher, writer, analyzer = init_agents()

# Sidebar
with st.sidebar:
    st.header("📊 Agent Status")
    st.success("✅ Researcher Agent (Active)")
    st.success("✅ Writer Agent (Active)")
    st.success("✅ Analyzer Agent (Active)")
    
    st.divider()
    st.caption("🔬 This system uses **decentralized** architecture.")
    st.caption("No central supervisor — agents communicate via message bus.")

# Main chat area
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.agent_messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "agent_info" in message:
            st.caption(f"🤖 {message['agent_info']}")

# Agent activity log
with st.expander("📋 Agent Activity Log", expanded=False):
    for log in st.session_state.agent_messages[-10:]:
        st.text(log)

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show thinking
    with st.chat_message("assistant"):
        with st.spinner("🔄 Agents are working..."):
            # Clear previous messages
            bus.messages.clear()
            
            # Run the system
            st.session_state.agent_messages.append(f"🔍 Query: {prompt}")
            
            # Researcher publishes query
            researcher.publish("query.request", {
                "type": "research_request",
                "query": prompt
            })
            
            time.sleep(1)
            
            # Collect final answer
            # bus.get_messages returns items shaped as {"topic": ..., "message": {...}},
            # so the actual payload lives under the "message" key.
            final_answer = None
            for record in bus.get_messages("answer.final"):
                payload = record.get("message", {})
                if payload.get("type") == "final_answer":
                    final_answer = payload
                    st.session_state.agent_messages.append(f"✅ Final answer ready")
            
            if final_answer:
                answer = final_answer["answer"]
                st.session_state.agent_messages.append(f"📝 Answer: {answer[:100]}...")
                
                # Display answer
                st.markdown(answer)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "agent_info": "Researcher + Writer + Analyzer"
                })
            else:
                error_msg = "⚠️ No final answer received from agents."
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "agent_info": "System Error"
                })