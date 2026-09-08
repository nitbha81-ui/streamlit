"""
Simple message bus for decentralized agent communication.
Agents publish messages and subscribe to topics.
"""

from typing import Dict, List, Callable, Any

class MessageBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.messages: List[Dict] = []
    
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
    
    def publish(self, topic: str, message: Dict[str, Any]):
        """Publish a message to a topic."""
        self.messages.append({"topic": topic, "message": message})
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                callback(message)
    
    def get_messages(self, topic: str = None) -> List[Dict]:
        """Get all messages, optionally filtered by topic."""
        if topic:
            return [m for m in self.messages if m["topic"] == topic]
        return self.messages

# Global message bus
bus = MessageBus()