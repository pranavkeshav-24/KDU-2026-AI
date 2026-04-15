"""API-level history management endpoints utilities."""

from typing import List

from langchain_core.messages import BaseMessage
from memory.store import get_session_history


def get_history(thread_id: str) -> List[dict]:
    """Retrieve full conversation history for a thread."""
    history_obj = get_session_history(thread_id)
    return [
        {
            "type": msg.type,
            "content": msg.content
        } for msg in history_obj.messages
    ]


def clear_history(thread_id: str) -> None:
    """Clear conversation history for a thread from Redis."""
    history_obj = get_session_history(thread_id)
    history_obj.clear()
