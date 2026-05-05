from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..core import EventType
from ..infrastructure import EventLog


ReplayCallable = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class ReplayDiff:
    agent_name: str
    call_id: str
    changed: bool
    original: Any
    replayed: Any


class ReplayEngine:
    """Replays logged agent inputs against registered current-version callables."""

    def __init__(self, event_log: EventLog, registry: dict[str, ReplayCallable]) -> None:
        self.event_log = event_log
        self.registry = registry

    async def replay_session(self, session_id: str) -> list[ReplayDiff]:
        events = self.event_log.read_session(session_id)
        originals: dict[str, Any] = {}
        calls: list[dict[str, Any]] = []
        for event in events:
            event_type = event.get("event_type")
            payload = event.get("payload", {})
            if event_type == EventType.AGENT_CALLED.value:
                calls.append(payload)
            if event_type == EventType.AGENT_RETURNED.value:
                record = payload.get("record", {})
                call_id = record.get("call_id")
                if call_id:
                    originals[call_id] = payload.get("result")

        diffs: list[ReplayDiff] = []
        for call in calls:
            agent_name = call["agent_name"]
            call_id = call["call_id"]
            if agent_name not in self.registry:
                continue
            replayed = self.registry[agent_name](call.get("payload", {}))
            if inspect.isawaitable(replayed):
                replayed = await replayed
            original = originals.get(call_id)
            diffs.append(
                ReplayDiff(
                    agent_name=agent_name,
                    call_id=call_id,
                    changed=original != replayed,
                    original=original,
                    replayed=replayed,
                )
            )
        return diffs
