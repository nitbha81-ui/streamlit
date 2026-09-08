# Decentralized Multi-Agent System

## Architecture: Decentralized
- No central supervisor
- Agents communicate via message bus
- Each agent has its own tools and memory
- Shared knowledge via Pinecone

## Agents
1. **Researcher Agent** – Finds facts (Pinecone + Web Search)
2. **Writer Agent** – Synthesizes information
3. **Analyzer Agent** – Verifies and refines answers

## Setup
1. Copy `.env.example` to `.env` and add your API keys
2. `pip install -r requirements.txt`
3. `streamlit run streamlit_app.py`