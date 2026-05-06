from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
import uuid

from app.core.config import settings
from app.services.token_service import TokenService


SessionMode = str


@dataclass
class SessionRecord:
    session_id: str
    thread_id: str
    user_id: str
    expires_at: int
    mode: SessionMode = "ai"
    client_secret_fingerprint: str | None = None
    seen_widget_ids: set[str] = field(default_factory=set)
    processed_idempotency_keys: set[str] = field(default_factory=set)
    messages: list[dict[str, str]] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions_by_id: dict[str, SessionRecord] = {}
        self._sessions_by_thread: dict[str, str] = {}
        self._sessions_by_secret: dict[str, str] = {}

    def create_session(self, user_id: str) -> SessionRecord:
        now = int(time.time())
        record = SessionRecord(
            session_id=f"sess_{uuid.uuid4().hex}",
            thread_id=f"thread_{uuid.uuid4().hex}",
            user_id=user_id,
            expires_at=now + settings.session_ttl_seconds,
        )
        with self._lock:
            self._sessions_by_id[record.session_id] = record
            self._sessions_by_thread[record.thread_id] = record.session_id
        return record

    def bind_client_secret(self, client_secret: str, session_id: str) -> None:
        fingerprint = TokenService.fingerprint(client_secret)
        with self._lock:
            session = self._sessions_by_id[session_id]
            session.client_secret_fingerprint = fingerprint
            self._sessions_by_secret[fingerprint] = session_id

    def get_by_client_secret(self, client_secret: str) -> SessionRecord | None:
        fingerprint = TokenService.fingerprint(client_secret)
        with self._lock:
            session_id = self._sessions_by_secret.get(fingerprint)
            return self._sessions_by_id.get(session_id) if session_id else None

    def get_by_thread(self, thread_id: str) -> SessionRecord | None:
        with self._lock:
            session_id = self._sessions_by_thread.get(thread_id)
            return self._sessions_by_id.get(session_id) if session_id else None

    def append_message(self, thread_id: str, role: str, content: str) -> None:
        with self._lock:
            session = self.get_by_thread(thread_id)
            if session:
                session.messages.append({"role": role, "content": content})

    def mark_widget_seen(self, thread_id: str, widget_id: str) -> bool:
        with self._lock:
            session = self.get_by_thread(thread_id)
            if not session or widget_id in session.seen_widget_ids:
                return False
            session.seen_widget_ids.add(widget_id)
            return True

    def process_idempotency_key(self, thread_id: str, key: str) -> bool:
        with self._lock:
            session = self.get_by_thread(thread_id)
            if not session or key in session.processed_idempotency_keys:
                return False
            session.processed_idempotency_keys.add(key)
            return True

    def set_mode(self, thread_id: str, mode: SessionMode) -> None:
        with self._lock:
            session = self.get_by_thread(thread_id)
            if session:
                session.mode = mode


store = SessionStore()

