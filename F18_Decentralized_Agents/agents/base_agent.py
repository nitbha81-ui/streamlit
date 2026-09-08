"""Base agent class for all agents in the decentralized system."""

from typing import Any, Dict, Optional
from core.message_bus import bus
from core.memory import create_agent_memory

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory = create_agent_memory()
        self.subscribed_topics = []
        
        print(f"✅ Agent '{name}' ({role}) created")
    
    def subscribe(self, topic: str):
        """Subscribe to a topic on the message bus."""
        bus.subscribe(topic, self.handle_message)
        self.subscribed_topics.append(topic)
        print(f"📬 Agent '{self.name}' subscribed to '{topic}'")
    
    def publish(self, topic: str, message: Dict[str, Any]):
        """Publish a message to the bus."""
        message["sender"] = self.name
        bus.publish(topic, message)
        print(f"📤 Agent '{self.name}' published to '{topic}'")
    
    def handle_message(self, message: Dict[str, Any]):
        """Handle incoming messages. Override in subclasses."""
        pass
    
    def run(self, input_data: Any) -> Any:
        """Run the agent. Override in subclasses."""
        return input_data