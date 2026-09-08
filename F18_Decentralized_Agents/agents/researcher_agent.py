"""Researcher Agent: Finds facts from Pinecone and Web."""

from typing import Any, Dict
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from configs.config import config
from configs.prompts import RESEARCHER_PROMPT
from core.vector_store import vector_store
from agents.base_agent import BaseAgent

class ResearcherAgent(BaseAgent):
    """Agent that finds factual information."""
    
    def __init__(self):
        super().__init__("Researcher", "Fact Finder")
        
        self.llm = ChatAnthropic(
            model=config.MODEL,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            temperature=config.TEMPERATURE,
            max_tokens=1024
        )
        
        self.web_search = DuckDuckGoSearchRun()
        
        # Subscribe to query requests
        self.subscribe("query.request")
    
    def handle_message(self, message: Dict[str, Any]):
        """Handle incoming query requests."""
        if message.get("type") == "research_request":
            query = message.get("query", "")
            print(f"🔍 Researcher Agent received: {query}")
            result = self.run(query)
            # Publish findings
            self.publish("research.complete", {
                "type": "research_findings",
                "query": query,
                "findings": result,
                "for": message.get("sender")
            })
    
    def run(self, query: str) -> str:
        """Research the given query."""
        print(f"🔍 Researching: {query}")
        
        # Try Pinecone first
        pinecone_result = vector_store.search(query)
        if "No information found" not in pinecone_result:
            pinecone_found = True
        else:
            pinecone_found = False
        
        # If not found, try web search
        web_result = ""
        if not pinecone_found:
            try:
                print("🌐 Web search triggered...")
                web_result = self.web_search.invoke(query)
                web_result = web_result[:1500]
            except Exception as e:
                web_result = f"Web search error: {str(e)}"
        
        # Combine results
        combined = ""
        if pinecone_found:
            combined += "📚 **From Pinecone Database:**\n\n" + pinecone_result + "\n\n"
        if web_result:
            combined += "🌐 **From Web Search:**\n\n" + web_result
        
        if not combined:
            combined = "No information found for this query."
        
        return combined