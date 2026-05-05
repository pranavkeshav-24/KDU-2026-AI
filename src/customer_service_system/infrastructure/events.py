from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from ..core import EventType, to_jsonable, utc_now


@dataclass(frozen=True)
class Event:
    event_id: str
    content_hash: str
    event_type: EventType
    session_id: str
    timestamp_utc: str
    payload: dict[str, Any]
    duration_ms: int | None = None


class EventLog:
    """Synchronous append-only JSONL event log."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def append(
        self,
        event_type: EventType,
        session_id: str,
        payload: dict[str, Any],
        duration_ms: int | None = None,
    ) -> Event:
        jsonable_payload = to_jsonable(payload)
        timestamp = utc_now()
        stable_material = {
            "event_type": event_type.value,
            "session_id": session_id,
            "payload": jsonable_payload,
            "duration_ms": duration_ms,
        }
        content_hash = sha256(
            json.dumps(stable_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        event = Event(
            event_id=str(uuid.uuid4()),
            content_hash=content_hash,
            event_type=event_type,
            session_id=session_id,
            timestamp_utc=timestamp,
            payload=jsonable_payload,
            duration_ms=duration_ms,
        )
        line = json.dumps(to_jsonable(event), sort_keys=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return event

    def read_session(self, session_id: str) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        events: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("session_id") == session_id:
                    events.append(event)
        return events
