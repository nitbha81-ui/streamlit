"""Shared Pinecone vector store for all agents."""

import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

try:
    # Preferred: dedicated package (langchain_community version is deprecated)
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - fallback for older installs
    from langchain_community.embeddings import HuggingFaceEmbeddings

from configs.config import config

class SharedVectorStore:
    """Singleton vector store shared by all agents."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        print("📚 Initializing shared vector store...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = self.pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes.indexes] if existing_indexes.indexes else []
        
        if config.PINECONE_INDEX not in index_names:
            print(f"📦 Creating index: {config.PINECONE_INDEX}")
            self.pc.create_index(
                name=config.PINECONE_INDEX,
                dimension=config.EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=config.PINECONE_CLOUD,
                    region=config.PINECONE_REGION
                )
            )
            time.sleep(3)
        
        self.index = self.pc.Index(config.PINECONE_INDEX)
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
        print("✅ Shared vector store ready")
    
    def search(self, query: str) -> str:
        """Search the vector store."""
        docs = self.retriever.invoke(query)
        if not docs:
            return "No information found in the knowledge base."
        return "\n\n".join([f"[Source {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

vector_store = SharedVectorStore()