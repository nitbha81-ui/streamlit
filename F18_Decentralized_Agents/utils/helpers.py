"""
Helper functions for the decentralized multi-agent system.
"""

import json
import time
from datetime import datetime

def format_timestamp():
    """Return current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_agent_message(agent_name: str, action: str, details: str = ""):
    """Log an agent action."""
    timestamp = format_timestamp()
    print(f"[{timestamp}] 🤖 {agent_name}: {action} {details}")

def safe_json_loads(data: str) -> dict:
    """Safely parse JSON data."""
    try:
        return json.loads(data)
    except:
        return {}

def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max length."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text