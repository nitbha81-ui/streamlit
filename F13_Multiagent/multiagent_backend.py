"""
================================================================================
MULTI-AGENT SYSTEM WITH EXPLICIT ARCHITECTURE & DESIGN PATTERNS
================================================================================

ARCHITECTURE: LAYERED (HYBRID) ARCHITECTURE
--------------------------------------------
This system combines multiple layers:
- Reactive Layer: Quick responses (web search, retrieval)
- Deliberative Layer: Planning and reasoning (Supervisor Agent)
- BDI Layer: Goal-driven behavior (Researcher → Writer → Analyzer)

DESIGN PATTERNS USED:
---------------------
1. ReAct (Reasoning + Acting) Pattern
   - Each agent follows: Thought → Action → Observation → Thought → ...
   - Enables adaptive reasoning in dynamic environments

2. Tool Use Pattern
   - Agents extend LLM capabilities with external tools
   - Tools: Pinecone Search, Web Search, Summarize, FactCheck

3. Reflection Pattern
   - Analyzer Agent critiques and refines the output
   - Enables self-correction and iterative improvement

4. CoALA (Cognitive Architectures for LLM Agents)
   - Combines reasoning + memory (Pinecone) + tool use
   - Dynamically adapts to novel tasks

================================================================================
"""

import os
import sys
import json
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- Make stdout/stderr UTF-8 safe ---
# On Windows the console often defaults to cp1252, which cannot encode the emoji
# used in the status messages below. Without this, the very first print() raises
# UnicodeEncodeError and the whole module fails to import (which surfaces as a
# misleading "backend could not start" message in the UI).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- LangChain Core ---
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import Tool
try:
    # LangChain >= 1.0 moved the classic ReAct agent APIs here.
    from langchain_classic.agents import AgentExecutor, create_react_agent
except ImportError:
    # Fallback for LangChain < 1.0.
    from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

# --- Pinecone ---
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# --- Web Search ---
from langchain_community.tools import DuckDuckGoSearchRun

# --- Load Environment ---
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "multi-agent-index")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in .env")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")

print("="*70)
print("🤖 MULTI-AGENT SYSTEM INITIALIZING...")
print("="*70)
print("\n📐 ARCHITECTURE: Layered (Hybrid)")
print("   - Reactive Layer: Web Search, Retrieval")
print("   - Deliberative Layer: Supervisor Orchestration")
print("   - BDI Layer: Goal-driven Agents (Researcher → Writer → Analyzer)")
print("\n📐 DESIGN PATTERNS:")
print("   - ReAct (Reasoning + Acting): Thought → Action → Observation")
print("   - Tool Use: Agents extend LLM with external capabilities")
print("   - Reflection: Analyzer Agent self-corrects output")
print("   - CoALA: Reasoning + Memory (Pinecone) + Tool Use")
print("="*70)

# ==========================================
# LAYER 1: REACTIVE LAYER (Fast Responses)
# ==========================================

"""
ARCHITECTURE: REACTIVE LAYER
----------------------------
- Purpose: Quick, immediate responses
- Examples: Web search, database retrieval
- Pattern: Tool Use (extending LLM capabilities)
"""

def retrieve_from_pinecone(query: str) -> str:
    """TOOL USE PATTERN: Agent uses external tool (Pinecone)"""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    result = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "knowledge base")
        result.append(f"[Source {i} — {source}] {doc.page_content}")
    return "\n\n".join(result)

def web_search(query: str) -> str:
    """TOOL USE PATTERN: Agent uses external tool (Web Search)"""
    try:
        search = DuckDuckGoSearchRun()
        result = search.invoke(query)
        return result[:1000]
    except Exception as e:
        return f"Web search error: {str(e)}"

def get_current_time(query: str = "") -> str:
    """TOOL USE PATTERN: Agent uses external tool (DateTime API)"""
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

# ==========================================
# LAYER 2: DELIBERATIVE LAYER (Planning & Reasoning)
# ==========================================

"""
ARCHITECTURE: DELIBERATIVE LAYER
--------------------------------
- Purpose: Planning, reasoning, and orchestration
- Examples: Supervisor Agent (manager)
- Pattern: ReAct (Reasoning + Acting loop)
"""

# --- Setup LLM (Claude) ---
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    anthropic_api_key=ANTHROPIC_API_KEY,
    temperature=0.3,
    max_tokens=1024
)

# --- Setup Pinecone (Memory) ---
"""
CoALA ARCHITECTURE: Memory Component
------------------------------------
- Purpose: Long-term storage and retrieval
- Implementation: Pinecone vector database
"""
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = pc.list_indexes()
index_names = [idx.name for idx in existing_indexes.indexes] if existing_indexes.indexes else []

if PINECONE_INDEX not in index_names:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1")
        )
    )

# ==========================================
# KNOWLEDGE BASE: Load documents from data/
# ==========================================

"""
CoALA ARCHITECTURE: Memory Ingestion
------------------------------------
- Reads every supported file in the data/ folder (PDF, DOCX, TXT/MD)
- Splits them into overlapping chunks
- Embeds and stores them in Pinecone for semantic retrieval

Ingestion only runs when the index is empty, so restarting the app does not
re-embed or duplicate the same documents.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Fallback facts used only if the data/ folder is missing or empty.
FALLBACK_KNOWLEDGE = [
    "Ashwagandha coffee is a beverage that combines coffee with Ashwagandha root powder.",
    "Ashwagandha is an adaptogenic herb used in Ayurvedic medicine for centuries.",
    "Turmeric coffee contains curcumin, which has anti-inflammatory properties.",
]


def _load_single_file(path: str) -> List[Document]:
    """Load one file into LangChain Documents based on its extension."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            return PyPDFLoader(path).load()
        if ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(path).load()
        if ext in (".txt", ".md"):
            from langchain_community.document_loaders import TextLoader
            return TextLoader(path, encoding="utf-8").load()
    except Exception as e:
        print(f"⚠️  Could not load {os.path.basename(path)}: {e}")
    return []


def load_documents() -> List[Document]:
    """Read and chunk every supported document in the data/ folder."""
    raw_docs: List[Document] = []

    if os.path.isdir(DATA_DIR):
        for name in sorted(os.listdir(DATA_DIR)):
            file_path = os.path.join(DATA_DIR, name)
            if os.path.isfile(file_path):
                loaded = _load_single_file(file_path)
                for doc in loaded:
                    # Track which source file each chunk came from.
                    doc.metadata["source"] = name
                raw_docs.extend(loaded)
                if loaded:
                    print(f"   📄 Loaded {name} ({len(loaded)} section(s))")

    if not raw_docs:
        print("⚠️  No documents found in data/. Using fallback knowledge base.")
        return [Document(page_content=text, metadata={"source": "fallback"})
                for text in FALLBACK_KNOWLEDGE]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"✂️  Split into {len(chunks)} chunks.")
    return chunks


def _index_is_empty(index_name: str) -> bool:
    """Return True if the Pinecone index has no vectors yet."""
    try:
        stats = pc.Index(index_name).describe_index_stats()
        return (stats.get("total_vector_count", 0) or 0) == 0
    except Exception as e:
        print(f"⚠️  Could not read index stats ({e}); assuming empty.")
        return True


# Connect to the existing index; ingest documents only if it is empty.
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings,
)

if _index_is_empty(PINECONE_INDEX):
    print("📥 Index is empty — ingesting documents from data/ ...")
    docs_to_index = load_documents()
    vectorstore.add_documents(docs_to_index)
    print(f"✅ Indexed {len(docs_to_index)} chunks into Pinecone.")
else:
    print("✅ Existing vectors found — skipping ingestion.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Pinecone vector store ready!")

# ==========================================
# AGENT 1: RESEARCHER AGENT (BDI Layer)
# ==========================================

"""
BDI ARCHITECTURE: BELIEF-DESIRE-INTENTION
-----------------------------------------
- BELIEF: Knowledge stored in Pinecone (facts, documents)
- DESIRE: Goal to find accurate information
- INTENTION: Plan to search Pinecone and Web

DESIGN PATTERN: REACT (Reasoning + Acting)
-------------------------------------------
Each agent follows: Thought → Action → Observation → Thought → Final Answer
"""

researcher_tools = [
    Tool(
        name="KnowledgeBaseSearch",
        func=retrieve_from_pinecone,
        description="Search the knowledge base for information about coffee, herbs, and health benefits."
    ),
    Tool(
        name="WebSearch",
        func=web_search,
        description="Search the web for current information."
    ),
]

researcher_prompt = PromptTemplate.from_template("""
You are a RESEARCHER AGENT.

ARCHITECTURE: BDI (Belief-Desire-Intention)
- BELIEF: Facts stored in knowledge base and web
- DESIRE: Find accurate, complete information
- INTENTION: Search tools systematically

You have access to the following tools:

{tools}

Use this EXACT format:

Question: the input question you must answer
Thought: think about what information you need
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation cycle can repeat)
Thought: I now know the final answer
Final Answer: a thorough summary of the findings

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

researcher_agent = create_react_agent(llm, researcher_tools, researcher_prompt)
researcher_executor = AgentExecutor(
    agent=researcher_agent,
    tools=researcher_tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=3
)

# ==========================================
# AGENT 2: WRITER AGENT (BDI Layer)
# ==========================================

"""
BDI ARCHITECTURE: BELIEF-DESIRE-INTENTION
-----------------------------------------
- BELIEF: Research findings from Researcher Agent
- DESIRE: Create clear, engaging, well-organized answer
- INTENTION: Synthesize information into coherent response

PATTERN: REACT (Reasoning + Acting)
-----------------------------------
Step 1: THOUGHT - How should I structure this answer?
Step 2: ACTION - Use Summarize tool if needed
Step 3: OBSERVATION - Review the summary
Step 4: FINAL ANSWER - Write the well-structured response
"""

# The Writer does not need external tools — it synthesizes the research the
# Researcher gathered into a clean answer. So it is a direct LLM chain rather
# than a ReAct agent (which would waste iterations on unnecessary tool calls).
writer_prompt = PromptTemplate.from_template("""
You are a WRITER AGENT.

ARCHITECTURE: BDI (Belief-Desire-Intention)
- BELIEF: Research findings provided below
- DESIRE: Write a clear, engaging, well-structured answer
- INTENTION: Organize information logically

Using ONLY the research findings below, write a clear, well-structured answer to
the user's question. Do not invent facts that are not supported by the research.

Research findings:
{research}

Question: {input}

Answer:
""")

writer_chain = writer_prompt | llm

# ==========================================
# AGENT 3: ANALYZER AGENT (Reflection Layer)
# ==========================================

"""
ARCHITECTURE: COALA (Cognitive Architectures for LLM Agents)
------------------------------------------------------------
- Purpose: Self-correction and quality verification
- Pattern: REFLECTION (Agent critiques its own output)

DESIGN PATTERN: REFLECTION
--------------------------
Step 1: Review the draft answer
Step 2: Check for factual accuracy
Step 3: Verify all claims are supported
Step 4: Provide feedback
Step 5: Offer an improved version if needed
"""

# The Analyzer performs REFLECTION: it reviews the draft against the research
# and returns an improved version. This is a reasoning task, not a tool-calling
# task, so it is a direct LLM chain.
analyzer_prompt = PromptTemplate.from_template("""
You are an ANALYZER AGENT.

ARCHITECTURE: COALA (Cognitive Architectures for LLM Agents)
- Purpose: Self-correction and quality verification
- PATTERN: REFLECTION (review the draft, verify accuracy, improve it)

Review the draft answer below against the research findings. Check that every
claim is supported and that the answer fully addresses the question. Then reply
in EXACTLY this structure:

VERDICT: PASS or NEEDS_REVISION or FAIL
FEEDBACK: your concise feedback
IMPROVED_ANSWER: the final, improved answer for the user

Original Question: {question}
Research Findings: {research}
Draft Answer: {draft}
""")

analyzer_chain = analyzer_prompt | llm

# ==========================================
# SUPERVISOR AGENT (Orchestrator)
# ==========================================

"""
ARCHITECTURE: LAYERED (HYBRID)
------------------------------
- Layer 1 (Reactive): Web Search, Pinecone Retrieval
- Layer 2 (Deliberative): Supervisor orchestrates the workflow
- Layer 3 (BDI): Goal-driven agents (Researcher → Writer → Analyzer)

DESIGN PATTERN: PLANNING (Orchestration)
----------------------------------------
- Creates a structured plan before acting
- Defines the sequence: Research → Write → Analyze
- Reduces logical errors
"""

def run_multi_agent(query: str) -> Dict[str, Any]:
    """SUPERVISOR AGENT: Orchestrates the multi-agent workflow"""
    
    print("\n" + "="*70)
    print(f"📝 USER QUERY: {query}")
    print("="*70)
    
    # --- Stage 1: Research (BDI Agent) ---
    print("\n🔍 RESEARCHER AGENT working...")
    print("   PATTERN: ReAct (Thought → Action → Observation)")
    research_result = researcher_executor.invoke({"input": query})
    research = research_result.get("output", "No research found.")
    print("✅ Research complete.")
    
    # --- Stage 2: Write (BDI Agent) ---
    print("\n✍️ WRITER AGENT working...")
    print("   PATTERN: LLM synthesis")
    writer_result = writer_chain.invoke({
        "input": query,
        "research": research
    })
    draft = getattr(writer_result, "content", str(writer_result)).strip()
    print("✅ Draft complete.")
    
    # --- Stage 3: Analyze (Reflection Pattern) ---
    print("\n🔬 ANALYZER AGENT working...")
    print("   PATTERN: Reflection (Self-Correction)")
    analyzer_result = analyzer_chain.invoke({
        "question": query,
        "research": research,
        "draft": draft
    })
    analysis = getattr(analyzer_result, "content", str(analyzer_result)).strip()
    print("✅ Analysis complete.")
    
    # --- Extract final answer (everything after IMPROVED_ANSWER:) ---
    final_answer = draft
    if "IMPROVED_ANSWER:" in analysis:
        improved = analysis.split("IMPROVED_ANSWER:", 1)[1].strip()
        if improved:
            final_answer = improved
    
    return {
        "query": query,
        "research": research,
        "draft": draft,
        "analysis": analysis,
        "final_answer": final_answer
    }

# ==========================================
# INTERACTIVE LOOP
# ==========================================

print("\n" + "="*70)
print("🚀 MULTI-AGENT SYSTEM IS READY!")
print("="*70)
print("\n📐 ARCHITECTURE SUMMARY:")
print("   - Layered (Hybrid): Reactive + Deliberative + BDI")
print("📐 DESIGN PATTERNS SUMMARY:")
print("   - Planning: Supervisor orchestrates workflow")
print("   - ReAct: Each agent uses Thought → Action → Observation")
print("   - Tool Use: Agents use external tools (Pinecone, Web Search, Summarize, FactCheck)")
print("   - Reflection: Analyzer Agent self-corrects")
print("   - CoALA: Reasoning + Memory + Tool Use")
print("\nType 'exit' to quit.\n")

def print_result(result: Dict[str, Any]):
    """Pretty print the result."""
    print("\n" + "="*70)
    print("📊 FINAL RESULT")
    print("="*70)
    print(f"\n📝 Query: {result['query']}")
    print("\n✅ FINAL ANSWER:")
    print("-"*50)
    print(result['final_answer'])


def interactive_loop():
    """Run the CLI interactive loop. Only invoked when this file is run directly."""
    while True:
        try:
            user_input = input("\n🔍 You: ").strip()

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            result = run_multi_agent(user_input)
            print_result(result)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


# Only run the interactive loop when executed as a script,
# so that Streamlit (and other importers) can reuse run_multi_agent()
# without triggering the blocking input() loop.
if __name__ == "__main__":
    interactive_loop()