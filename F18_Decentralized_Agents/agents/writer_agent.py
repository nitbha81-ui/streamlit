"""Writer Agent: Synthesizes research into answers."""

from typing import Any, Dict
from langchain_anthropic import ChatAnthropic
from configs.config import config
from configs.prompts import WRITER_PROMPT
from agents.base_agent import BaseAgent

class WriterAgent(BaseAgent):
    """Agent that synthesizes information."""
    
    def __init__(self):
        super().__init__("Writer", "Synthesizer")
        
        self.llm = ChatAnthropic(
            model=config.MODEL,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            temperature=0.4,
            max_tokens=1024
        )
        
        self.subscribe("research.complete")
    
    def handle_message(self, message: Dict[str, Any]):
        """Handle research findings."""
        if message.get("type") == "research_findings":
            query = message.get("query", "")
            findings = message.get("findings", "")
            print(f"✍️ Writer Agent processing findings for: {query}")
            
            result = self.run({"query": query, "findings": findings})
            
            # Publish draft
            self.publish("draft.ready", {
                "type": "draft_ready",
                "query": query,
                "draft": result,
                "for": "Analyzer"
            })
    
    def run(self, input_data: dict) -> str:
        """Write an answer based on research findings."""
        query = input_data.get("query", "")
        findings = input_data.get("findings", "")
        
        prompt = f"{WRITER_PROMPT}\n\n"
        prompt += f"Query: {query}\n\n"
        prompt += f"Research Findings:\n{findings}\n\n"
        prompt += "Write a clear, engaging answer based on the above research:"
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"