from fastapi.testclient import TestClient

from config import settings
from main import app


def test_ui_root_and_browser_redirect(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")

    with TestClient(app, raise_server_exceptions=False) as client:
        root_response = client.get("/")
        assert root_response.status_code == 200
        assert "text/html" in root_response.headers["content-type"]
        assert "Nova" in root_response.text

        redirect_response = client.get("/chat", follow_redirects=False)
        assert redirect_response.status_code == 307
        assert redirect_response.headers["location"] == "/"


def test_chat_fallback_keeps_history_available(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    thread_id = "route-test-thread"

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/chat",
            json={"message": "Hello there", "thread_id": thread_id},
        )
        assert response.status_code == 200
        assert response.json()["intent"] == "assistant_unconfigured"

        history_response = client.get(f"/history/{thread_id}")
        assert history_response.status_code == 200
        assert len(history_response.json()["messages"]) == 2

        delete_response = client.delete(f"/history/{thread_id}")
        assert delete_response.status_code == 200
