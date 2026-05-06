import asyncio

from app.core.config import settings
from app.schemas.action import WidgetActionRequest
from app.services.chatkit_server import ChatKitServer
from app.services.llm_client import OpenAILLMClient
from app.services.session_store import SessionStore
from app.services.token_service import TokenService
from app.services.tool_registry import ToolRegistry


def _server() -> tuple[ChatKitServer, SessionStore, TokenService]:
    session_store = SessionStore()
    token_service = TokenService("test-secret", settings.session_ttl_seconds)
    server = ChatKitServer(
        session_store,
        token_service,
        OpenAILLMClient(None, "test-model"),
        ToolRegistry(),
    )
    return server, session_store, token_service


def test_client_secret_cannot_access_another_thread() -> None:
    server, session_store, token_service = _server()
    alice = session_store.create_session("alice")
    bob = session_store.create_session("bob")
    alice_secret = token_service.issue_client_secret(
        thread_id=alice.thread_id,
        user_id=alice.user_id,
        session_id=alice.session_id,
    )
    session_store.bind_client_secret(alice_secret, alice.session_id)

    try:
        server.authorize_thread(alice_secret, bob.thread_id)
    except Exception as exc:
        assert getattr(exc, "status_code") == 403
    else:
        raise AssertionError("Expected cross-thread access to be forbidden.")


def test_widget_action_idempotency() -> None:
    asyncio.run(_assert_widget_action_idempotency())


async def _assert_widget_action_idempotency() -> None:
    server, session_store, token_service = _server()
    session = session_store.create_session("alice")
    secret = token_service.issue_client_secret(
        thread_id=session.thread_id,
        user_id=session.user_id,
        session_id=session.session_id,
    )
    session_store.bind_client_secret(secret, session.session_id)
    request = WidgetActionRequest(
        thread_id=session.thread_id,
        widget_id="wdg_123456789",
        action_type="book_now",
        payload={},
        idempotency_key="wdg_123456789:book_now",
    )

    first = await server.handle_widget_action(secret, request)
    second = await server.handle_widget_action(secret, request)

    assert first.status == "confirmed"
    assert second.status == "duplicate"
