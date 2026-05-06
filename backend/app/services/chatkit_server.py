from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import HTTPException, status

from app.schemas.action import WidgetActionRequest, WidgetActionResponse
from app.schemas.chat import ChatRequest
from app.schemas.widget import WidgetDefinition
from app.services.llm_client import OpenAILLMClient
from app.services.session_store import SessionRecord, SessionStore
from app.services.token_service import TokenError, TokenService
from app.services.tool_registry import ToolRegistry


class ChatKitServer:
    """Self-managed ChatKit-style protocol wrapper.

    The browser only sees normalized SSE events:
    text_delta, widget, done, and error.
    """

    def __init__(
        self,
        session_store: SessionStore,
        token_service: TokenService,
        llm_client: OpenAILLMClient,
        tool_registry: ToolRegistry,
    ) -> None:
        self._sessions = session_store
        self._tokens = token_service
        self._llm = llm_client
        self._tools = tool_registry

    def authorize_thread(self, client_secret: str, thread_id: str) -> SessionRecord:
        try:
            claims = self._tokens.verify(client_secret)
        except TokenError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

        session = self._sessions.get_by_client_secret(client_secret)
        thread_owner = self._sessions.get_by_thread(thread_id)
        if not session or not thread_owner:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unknown session.")
        if claims["thread_id"] != thread_id or session.thread_id != thread_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="client_secret is not authorized for this thread.",
            )
        if claims["user_id"] != thread_owner.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Thread owner does not match token subject.",
            )
        return session

    async def stream_turn(self, client_secret: str, request: ChatRequest) -> AsyncIterator[str]:
        session = self.authorize_thread(client_secret, request.thread_id)
        self._sessions.append_message(session.thread_id, "user", request.message)

        if session.mode == "human":
            yield self._event("text_delta", {"delta": "A human agent is handling this thread."})
            yield self._event("done", {"thread_id": session.thread_id})
            return

        assistant_parts: list[str] = []
        try:
            async for delta in self._llm.stream_reply(messages=session.messages):
                assistant_parts.append(delta)
                yield self._event("text_delta", {"delta": delta})

            self._sessions.append_message(session.thread_id, "assistant", "".join(assistant_parts))
            for widget in self._tools.widgets_for_turn(
                thread_id=session.thread_id,
                user_message=request.message,
            ):
                if self._sessions.mark_widget_seen(session.thread_id, widget.widget_id):
                    if widget.type == "handoff_notice":
                        self._sessions.set_mode(session.thread_id, "handoff_pending")
                    yield self._event("widget", self._widget_payload(widget))
            yield self._event("done", {"thread_id": session.thread_id})
        except Exception as exc:
            yield self._event("error", {"message": str(exc)})

    async def handle_widget_action(
        self,
        client_secret: str,
        request: WidgetActionRequest,
    ) -> WidgetActionResponse:
        session = self.authorize_thread(client_secret, request.thread_id)
        first_submission = self._sessions.process_idempotency_key(
            session.thread_id,
            request.idempotency_key,
        )
        if not first_submission:
            return WidgetActionResponse(
                widget_id=request.widget_id,
                status="duplicate",
                message="This action was already processed.",
            )

        if request.action_type == "book_now":
            self._sessions.append_message(
                session.thread_id,
                "tool",
                f"Booking confirmed for widget {request.widget_id}.",
            )
            return WidgetActionResponse(
                widget_id=request.widget_id,
                status="confirmed",
                message="Booking request confirmed.",
            )

        if request.action_type == "acknowledge_handoff":
            self._sessions.set_mode(session.thread_id, "human")
            return WidgetActionResponse(
                widget_id=request.widget_id,
                status="confirmed",
                message="Human handoff is now active.",
            )

        return WidgetActionResponse(
            widget_id=request.widget_id,
            status="error",
            message=f"Unsupported action type: {request.action_type}",
        )

    @staticmethod
    def _event(name: str, payload: dict) -> str:
        data = json.dumps(payload, separators=(",", ":"))
        return f"event: {name}\ndata: {data}\n\n"

    @staticmethod
    def _widget_payload(widget: WidgetDefinition) -> dict:
        if hasattr(widget, "model_dump"):
            return widget.model_dump()
        return widget.dict()
