"""
SIMPLE AI AGENT WITH RAG + 2 MCP TOOLS (Calculator & DateTime)
LLM: Claude (Anthropic)
VECTOR DB: Pinecone

This module builds a LangChain agent that has access to three tools:
  1. KnowledgeRetriever - RAG over a Pinecone-backed knowledge base
  2. Calculator        - safe math evaluation
  3. DateTime          - current date and time

It can be run directly (CLI demo + interactive loop) or imported by other
programs (e.g. streamlit_app.py) via the exported helpers:
  - build_agent()  -> creates and returns the agent
  - ask_agent()    -> sends a query to a given agent and returns the answer
"""

import os
import ast
import math
import operator
from datetime import datetime
from dotenv import load_dotenv

# LangChain Core
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

from langchain_core.documents import Document

# Agent (LangChain v1's create_agent)
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# Claude LLM
from langchain_anthropic import ChatAnthropic

# Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Embeddings (local, no API key)
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ==========================================
# CONFIG
# ==========================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384  # all-MiniLM-L6-v2 dimension

# Sample knowledge base
KNOWLEDGE_BASE = [
    "The capital of France is Paris.",
    "The capital of India is New Delhi.",
    "The capital of Japan is Tokyo.",
    "India has 28 states.",
    "The population of India is over 1.4 billion.",
]

SYSTEM_PROMPT = """
You are a helpful AI assistant with access to these tools:
1. KnowledgeRetriever - Search the knowledge base for facts
2. Calculator - Do math
3. DateTime - Get current time

Always use the right tool for the right question.
If you don't know something, use KnowledgeRetriever first.
Be concise and helpful.
"""


# ==========================================
# STEP 1: SETUP LLM (Claude)
# ==========================================

def build_llm() -> ChatAnthropic:
    """Create and return the Claude chat model."""
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env")

    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=anthropic_api_key,
        temperature=0.2,
        max_tokens=1024,
    )


# ==========================================
# STEP 2: SETUP PINECONE (Vector DB)
# ==========================================

def build_retriever(k: int = 2):
    """Create the Pinecone index (if needed), load docs, and return a retriever."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in .env")

    index_name = os.getenv("PINECONE_INDEX", "simple-agent-kb")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if index exists, create if not
    existing = pc.list_indexes()
    index_names = [idx["name"] for idx in existing] if existing else []

    if index_name not in index_names:
        pc.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    # Embeddings (local model, no API key required)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Convert knowledge base to LangChain Documents.
    # A stable id per document keeps re-runs idempotent (no duplicates).
    documents = [Document(page_content=text) for text in KNOWLEDGE_BASE]
    ids = [f"kb-{i}" for i in range(len(documents))]

    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
        ids=ids,
    )

    return vectorstore.as_retriever(search_kwargs={"k": k})


# ==========================================
# STEP 3: MCP TOOLS (KnowledgeRetriever + Calculator + DateTime)
# ==========================================

# Safe operators/functions allowed inside the Calculator tool.
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
}

_SAFE_NAMES = {"pi": math.pi, "e": math.e}


def _safe_eval(node):
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):  # numbers
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("Operator not allowed.")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("Operator not allowed.")
        return op(_safe_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
            raise ValueError("Function not allowed.")
        args = [_safe_eval(arg) for arg in node.args]
        return _SAFE_FUNCTIONS[node.func.id](*args)
    raise ValueError("Unsupported expression.")


def make_tools(retriever):
    """Build the list of agent tools bound to the given retriever."""

    def retrieve_knowledge(query: str) -> str:
        """Search the knowledge base for information."""
        docs = retriever.invoke(query)
        if not docs:
            return "No information found."
        return "\n\n".join(doc.page_content for doc in docs)

    def calculator(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            return f"Result: {result}"
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError) as e:
            return f"Error: Invalid expression ({e})."

    def get_current_time(query: str = "") -> str:
        """Get the current date and time."""
        now = datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

    return [
        Tool(
            name="KnowledgeRetriever",
            func=retrieve_knowledge,
            description="Search the knowledge base for facts about countries, capitals, population, etc.",
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description="Perform mathematical calculations. Example: '25 * 4' or 'sqrt(144)'",
        ),
        Tool(
            name="DateTime",
            func=get_current_time,
            description="Get the current date and time.",
        ),
    ]


# ==========================================
# STEP 4: BUILD THE AGENT
# ==========================================

def build_agent():
    """Assemble the LLM, retriever, tools and return a ready-to-use agent."""
    llm = build_llm()
    retriever = build_retriever()
    tools = make_tools(retriever)
    memory = MemorySaver()

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )


def ask_agent(agent, query: str, thread_id: str = "user-session-1") -> str:
    """Send a query to the agent and return the final answer text."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )
        # Extract the final AI answer
        for msg in reversed(result["messages"]):
            if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", None):
                content = msg.content
                # content may be a string or a list of content blocks
                if isinstance(content, list):
                    parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict)
                    ]
                    return "".join(parts).strip() or str(content)
                return content
        return "No response from agent."
    except Exception as e:  # noqa: BLE001 - surface any runtime error to the caller
        return f"Error: {str(e)}"


# ==========================================
# STEP 5: CLI DEMO (only when run directly)
# ==========================================

def main():
    print("=" * 60)
    print("🤖 SIMPLE AI AGENT WITH RAG + PINECONE + MCP TOOLS")
    print("=" * 60)

    print("\n✅ Initializing Claude Haiku 4.5...")
    print("📚 Setting up Pinecone vector database...")
    agent = build_agent()
    print("✅ Agent created successfully!")

    print("\n" + "=" * 60)
    print("🚀 AGENT IS READY! TESTING...")
    print("=" * 60)

    test_queries = [
        "What is the capital of India?",
        "Calculate 25 * 4 + 10",
        "What time is it?",
        "What is the capital of Japan?",
        "Calculate square root of 144",
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print(f"🤖 Answer: {ask_agent(agent, query)}")

    print("\n" + "=" * 60)
    print("💬 Type 'exit' to quit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            if not user_input:
                continue
            print(f"\n🤖 {ask_agent(agent, user_input)}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
