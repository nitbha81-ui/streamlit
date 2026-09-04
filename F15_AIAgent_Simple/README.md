# 🤖 Simple AI Agent (RAG + Pinecone + MCP Tools)

A minimal AI agent built with **LangChain v1** that combines:

- **Claude (Anthropic)** as the LLM (`claude-haiku-4-5`)
- **Pinecone** as the vector database for Retrieval-Augmented Generation (RAG)
- Three tools the agent can call:
  1. **KnowledgeRetriever** – answers facts from a Pinecone-backed knowledge base
  2. **Calculator** – safely evaluates math expressions (e.g. `sqrt(144)`, `25 * 4 + 10`)
  3. **DateTime** – returns the current date and time

It ships with two front ends that share the same agent logic:

- A **command-line demo + interactive chat** (`AIagent_simple.py`)
- A **Streamlit web chat UI** (`streamlit_app.py`)

---

## 📁 Project structure

```
F15_AIAgent_Simple/
├── AIagent_simple.py    # Agent logic + CLI demo (importable helpers)
├── streamlit_app.py     # Streamlit chat UI (reuses AIagent_simple)
├── requirements.txt     # Python dependencies
├── .env                 # API keys & Pinecone config (not committed)
└── README.md
```

---

## ✅ Requirements

- Python **3.11**
- An **Anthropic API key** (for Claude)
- A **Pinecone API key** (serverless index)

The embedding model (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions)
runs **locally** and needs no API key.

---

## ⚙️ Setup

### 1. Create / activate a virtual environment

```powershell
# From the project root
python -m venv venv311
.\venv311\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

> The first install downloads PyTorch and the sentence-transformers model,
> so it may take a few minutes.

### 3. Configure your `.env`

Create a `.env` file in the project root:

```env
# Anthropic API Key (for Claude Haiku 4.5)
ANTHROPIC_API_KEY=your-anthropic-key-here

# Pinecone
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX=simple-agent-kb
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

| Variable            | Required | Default            | Description                              |
| ------------------- | -------- | ------------------ | ---------------------------------------- |
| `ANTHROPIC_API_KEY` | ✅       | –                  | Claude API key                           |
| `PINECONE_API_KEY`  | ✅       | –                  | Pinecone API key                         |
| `PINECONE_INDEX`    | ❌       | `simple-agent-kb`  | Index name (auto-created if missing)     |
| `PINECONE_CLOUD`    | ❌       | `aws`              | Serverless cloud provider                |
| `PINECONE_REGION`   | ❌       | `us-east-1`        | Serverless region                        |

> ⚠️ **Keep your keys secret.** Never commit real API keys to version control.
> Rotate any key that has been shared or exposed.

---

## ▶️ Running

### Command-line demo

Runs a set of sample queries, then drops into an interactive prompt:

```powershell
python AIagent_simple.py
```

Type `exit`, `quit`, or `bye` to leave the interactive loop.

### Streamlit web UI

```powershell
streamlit run streamlit_app.py
```

This opens a chat interface in your browser. The agent keeps conversation
memory per session, and the sidebar has a **Clear conversation** button.

---

## 🧠 How it works

1. **LLM** – `build_llm()` creates the Claude chat model from `ANTHROPIC_API_KEY`.
2. **Vector DB** – `build_retriever()` connects to Pinecone, creates the index
   if it doesn't exist, embeds a small sample knowledge base locally, and
   returns a retriever (top-`k = 2`). Documents use stable IDs (`kb-0`, `kb-1`, …)
   so re-runs don't create duplicates.
3. **Tools** – `make_tools(retriever)` bundles the KnowledgeRetriever,
   Calculator, and DateTime tools.
4. **Agent** – `build_agent()` wires the model, tools, system prompt, and an
   in-memory checkpointer (`MemorySaver`) together via LangChain's
   `create_agent`.
5. **Query** – `ask_agent(agent, query, thread_id)` sends a message and returns
   the final answer text.

### The knowledge base

The sample facts live in `KNOWLEDGE_BASE` inside `AIagent_simple.py`:

```python
KNOWLEDGE_BASE = [
    "The capital of France is Paris.",
    "The capital of India is New Delhi.",
    "The capital of Japan is Tokyo.",
    "India has 28 states.",
    "The population of India is over 1.4 billion.",
]
```

Add your own facts to this list and re-run to expand what the agent knows.

---

## 🔒 Security note on the Calculator

The Calculator tool does **not** use Python's `eval()`. Instead it parses the
expression into an AST and evaluates only a whitelist of math operators and
functions (`sqrt`, `sin`, `log`, `factorial`, etc.). This prevents arbitrary
code execution from tool input.

---

## 🧩 Reusing the agent in your own code

Because the CLI demo is guarded by `if __name__ == "__main__":`, you can import
the helpers without triggering the interactive loop:

```python
from AIagent_simple import build_agent, ask_agent

agent = build_agent()
print(ask_agent(agent, "What is the capital of Japan?"))
```

---

## 🛠️ Troubleshooting

| Symptom                                        | Fix                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY not found in .env`          | Add the key to `.env` in the project root.                          |
| `PINECONE_API_KEY not found in .env`           | Add the key to `.env`.                                              |
| Pinecone auth / region errors                  | Verify the key, and that `PINECONE_CLOUD` / `PINECONE_REGION` match your account. |
| First run is slow                              | Normal – the local embedding model is downloading.                  |
| `langchain-community` deprecation warning      | Harmless. The community `HuggingFaceEmbeddings` still works.        |
