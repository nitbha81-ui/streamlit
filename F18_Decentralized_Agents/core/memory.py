"""Memory management for agents."""

from typing import Dict, Any

class AgentMemory:
    """Simple memory for each agent."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
    
    def store(self, key: str, value: Any):
        self.data[key] = value
    
    def retrieve(self, key: str) -> Any:
        return self.data.get(key, None)
    
    def clear(self):
        self.data.clear()

# Each agent gets its own memory
def create_agent_memory():
    return AgentMemory()