import pytest
from unittest.mock import patch, MagicMock

from config import settings
from memory.history import get_history, clear_history
from memory.store import get_session_history


@pytest.fixture
def mock_redis(monkeypatch):
    """Mocks RedisChatMessageHistory so we don't need a live DB container to test locally"""
    class MockMessage:
        def __init__(self, type, content):
            self.type = type
            self.content = content

    class MockHistory:
        def __init__(self, session_id, url, ttl):
            self.messages = []
            
        def add_user_message(self, text):
            self.messages.append(MockMessage("human", text))
            
        def add_ai_message(self, text):
            self.messages.append(MockMessage("ai", text))
            
        def clear(self):
            self.messages = []

    history_store = {}

    def get_mock_history(session_id: str, url: str, ttl: int):
        if session_id not in history_store:
            history_store[session_id] = MockHistory(session_id, url, ttl)
        return history_store[session_id]

    monkeypatch.setattr("memory.store.RedisChatMessageHistory", get_mock_history)
    return history_store


def test_session_history_memory(mock_redis):
    # Act
    history = get_session_history("user123:session_abc")
    history.add_user_message("Hello AI")
    history.add_ai_message("Hello User")

    # Assert fetch works
    hist_list = get_history("user123:session_abc")
    assert len(hist_list) == 2
    assert hist_list[0]["type"] == "human"
    assert hist_list[1]["type"] == "ai"

    # Assert clear works
    clear_history("user123:session_abc")
    assert len(get_history("user123:session_abc")) == 0
