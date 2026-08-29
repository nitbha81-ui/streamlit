# 🤖 Multi-Agent Research System

A multi-agent AI system that answers questions by chaining three specialized
agents — **Researcher → Writer → Analyzer** — under a **Supervisor**
orchestrator. It combines Claude (Anthropic) for reasoning, Pinecone for
long-term memory, and DuckDuckGo for live web search, exposed through both a
command-line interface and a Streamlit web app.

---

## Architecture

The system uses a **Layered (Hybrid) architecture**:

| Layer | Responsibility | Components |
|-------|----------------|------------|
| **Reactive** | Fast, direct responses | Web search, Pinecone retrieval |
| **Deliberative** | Planning & orchestration | Supervisor agent |
| **BDI** | Goal-driven behavior | Researcher, Writer, Analyzer |

### The agents

1. **Researcher** — gathers information from the Pinecone knowledge base and the
   web.
2. **Writer** — synthesizes the research into a clear, structured draft.
3. **Analyzer** — reflects on the draft, fact-checks it, and returns an improved
   final answer.
4. **Supervisor** — orchestrates the pipeline: Research → Write → Analyze.

### Design patterns

| Pattern | Where it is used |
|---------|------------------|
| **ReAct** (Reasoning + Acting) | Every agent follows Thought → Action → Observation |
| **Tool Use** | Agents call Pinecone, Web Search, Summarize, FactCheck |
| **Reflection** | The Analyzer critiques and refines the draft |
| **CoALA** | Reasoning + Memory (Pinecone) + Tool Use combined |

---

## Project structure

```
F13_Multiagent/
├── data/                   # Source documents (PDF, DOCX, TXT/MD) for the knowledge base
├── multiagent_backend.py   # Agents, tools, ingestion, and the run_multi_agent() pipeline
├── streamlit_app.py        # Streamlit web UI
├── requirements.txt        # Python dependencies
├── .env                    # API keys and Pinecone config (not committed)
└── README.md
```

---

## Prerequisites

- Python 3.11
- An [Anthropic API key](https://console.anthropic.com/)
- A [Pinecone API key](https://www.pinecone.io/)

---

## Setup

### 1. Create and activate the virtual environment

```powershell
# Windows (PowerShell)
python -m venv venv311
.\venv311\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=memory-vector-store
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

> **Security note:** Never commit real API keys. Add `.env` to your `.gitignore`.
> If a key has already been shared or committed, rotate it in the provider
> dashboard.

---

## Usage

### Web app (Streamlit)

```powershell
streamlit run streamlit_app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).
Ask a question in the chat box; the app shows the final answer plus an optional
agent-by-agent breakdown (Researcher findings, Writer draft, Analyzer
reflection).

### Command line

```powershell
python multiagent_backend.py
```

Type a question at the `🔍 You:` prompt. Type `exit`, `quit`, or `bye` to leave.

---

## How it works

`run_multi_agent(query)` runs the full pipeline and returns a dictionary:

```python
{
    "query":        "...",  # the original question
    "research":     "...",  # Researcher agent output
    "draft":        "...",  # Writer agent output
    "analysis":     "...",  # Analyzer agent reflection (VERDICT/FEEDBACK/...)
    "final_answer": "...",  # improved answer if the Analyzer suggested one
}
```

The Streamlit app imports this function (the CLI loop is guarded behind
`if __name__ == "__main__"`, so importing the module does not start the prompt)
and renders each stage.

---

## Knowledge base (document ingestion)

The Researcher agent answers from your own documents. Place files in the
`data/` folder and the backend will load, chunk, embed, and index them into
Pinecone on first run.

Supported formats:

| Format | Loader |
|--------|--------|
| `.pdf` | `PyPDFLoader` (via `pypdf`) |
| `.docx` | `Docx2txtLoader` (via `docx2txt`) |
| `.txt`, `.md` | `TextLoader` |

Details:

- Documents are split with `RecursiveCharacterTextSplitter`
  (chunk size 1000, overlap 150).
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  (384 dimensions, cosine similarity).
- Each chunk keeps a `source` metadata field so retrieved results cite the
  originating filename.
- **Ingestion only runs when the Pinecone index is empty.** Restarting the app
  will not re-embed or duplicate documents. To re-ingest after changing files,
  clear the index (see below) or use a fresh `PINECONE_INDEX` name.
- If `data/` is missing or empty, a small fallback knowledge base is used.

### Re-indexing after changing documents

Because ingestion is skipped when vectors already exist, delete the existing
vectors (or point `PINECONE_INDEX` at a new name) before restarting so the new
documents are picked up.

---

## Tech stack

- **LLM:** Claude Haiku 4.5 via `langchain-anthropic`
- **Orchestration:** LangChain ReAct agents (`AgentExecutor`)
- **Memory / vector store:** Pinecone (`langchain-pinecone`)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Document loading:** `pypdf` (PDF), `docx2txt` (DOCX), LangChain loaders + text splitter
- **Web search:** DuckDuckGo (`duckduckgo-search`)
- **UI:** Streamlit

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ANTHROPIC_API_KEY not found` | Ensure `.env` exists and contains the key |
| `PINECONE_API_KEY not found` | Add your Pinecone key to `.env` |
| App hangs on start | First run builds the Pinecone index and downloads the embedding model; give it a moment |
| Import errors | Confirm `pip install -r requirements.txt` completed in the active venv |
