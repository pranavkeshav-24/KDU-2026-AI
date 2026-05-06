from __future__ import annotations

import hashlib
import html
from typing import Iterable

from app.schemas.widget import WidgetDefinition


class ToolRegistry:
    """Server-owned tool registry for server-driven UI widgets."""

    def widgets_for_turn(self, *, thread_id: str, user_message: str) -> Iterable[WidgetDefinition]:
        normalized = user_message.lower()
        if any(term in normalized for term in ("flight", "book", "travel", "trip")):
            widget_id = self._stable_id("flight", thread_id, user_message)
            yield WidgetDefinition(
                widget_id=widget_id,
                type="flight_card",
                props={
                    "airline": "KDU Air",
                    "route": self._route_for(normalized),
                    "departure": "09:20",
                    "arrival": "12:45",
                    "price": "$428",
                    "fare": "Flex economy",
                    "summary": self._safe_text("Recommended itinerary based on your request."),
                },
                action_endpoint="/api/action",
                idempotency_key=f"{widget_id}:book_now",
            )
        if "handoff" in normalized or "human" in normalized:
            widget_id = self._stable_id("handoff", thread_id, user_message)
            yield WidgetDefinition(
                widget_id=widget_id,
                type="handoff_notice",
                props={
                    "title": "Human handoff requested",
                    "body": "AI responses are paused while a support agent reviews the thread.",
                },
                action_endpoint="/api/action",
                idempotency_key=f"{widget_id}:acknowledge",
            )

    @staticmethod
    def _stable_id(prefix: str, thread_id: str, user_message: str) -> str:
        digest = hashlib.sha256(f"{thread_id}:{prefix}:{user_message}".encode("utf-8")).hexdigest()
        return f"wdg_{digest[:24]}"

    @staticmethod
    def _safe_text(value: str) -> str:
        return html.escape(value, quote=True)

    @staticmethod
    def _route_for(message: str) -> str:
        if "tokyo" in message:
            return "San Francisco -> Tokyo"
        if "london" in message:
            return "New York -> London"
        if "paris" in message:
            return "New York -> Paris"
        return "New York -> San Francisco"

