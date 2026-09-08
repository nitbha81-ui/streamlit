"""Analyzer Agent: Verifies and refines answers."""

from typing import Any, Dict
from langchain_anthropic import ChatAnthropic
from configs.config import config
from configs.prompts import ANALYZER_PROMPT
from agents.base_agent import BaseAgent

class AnalyzerAgent(BaseAgent):
    """Agent that verifies and refines answers."""
    
    def __init__(self):
        super().__init__("Analyzer", "Verifier")
        
        self.llm = ChatAnthropic(
            model=config.MODEL,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            temperature=0.2,
            max_tokens=1024
        )
        
        self.subscribe("draft.ready")
    
    def handle_message(self, message: Dict[str, Any]):
        """Handle draft ready messages."""
        if message.get("type") == "draft_ready":
            query = message.get("query", "")
            draft = message.get("draft", "")
            print(f"🔬 Analyzer Agent verifying draft for: {query}")
            
            result = self.run({"query": query, "draft": draft})
            
            # Publish final result
            self.publish("answer.final", {
                "type": "final_answer",
                "query": query,
                "answer": result,
                "verified_by": self.name
            })
    
    def run(self, input_data: dict) -> str:
        """Analyze and refine the draft."""
        query = input_data.get("query", "")
        draft = input_data.get("draft", "")
        
        prompt = f"{ANALYZER_PROMPT}\n\n"
        prompt += f"Query: {query}\n\n"
        prompt += f"Draft Answer to Analyze:\n{draft}\n\n"
        prompt += "Provide your analysis and an improved version if needed:"
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Analysis error: {str(e)}"