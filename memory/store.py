"""Session history with a safe in-memory fallback when Redis is unavailable."""

from typing import Dict

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from config import settings

_in_memory_store: Dict[str, InMemoryChatMessageHistory] = {}


def _get_in_memory_history(session_id: str) -> InMemoryChatMessageHistory:
    """Provide a stable local history store for development and offline use."""
    if session_id not in _in_memory_store:
        _in_memory_store[session_id] = InMemoryChatMessageHistory()
    return _in_memory_store[session_id]


def get_session_history(session_id: str):
    """
    Prefer Redis-backed history, but fall back to in-memory storage when Redis
    is not configured or cannot be reached.
    """
    if not settings.REDIS_URL:
        return _get_in_memory_history(session_id)

    try:
        history = RedisChatMessageHistory(
            session_id=session_id,
            url=settings.REDIS_URL,
            ttl=86400,
        )
        # Force an initial read so connection issues are caught here.
        history.messages
        return history
    except Exception:
        return _get_in_memory_history(session_id)
