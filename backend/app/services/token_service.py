from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


class TokenError(ValueError):
    """Raised when a client_secret is invalid or expired."""


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


class TokenService:
    def __init__(self, secret: str, ttl_seconds: int) -> None:
        self._secret = secret.encode("utf-8")
        self._ttl_seconds = ttl_seconds

    def issue_client_secret(self, *, thread_id: str, user_id: str, session_id: str) -> str:
        now = int(time.time())
        payload = {
            "typ": "chatkit_client_secret",
            "sid": session_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "iat": now,
            "exp": now + self._ttl_seconds,
        }
        encoded_payload = _b64encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        signature = self._sign(encoded_payload)
        return f"ck_{encoded_payload}.{signature}"

    def verify(self, client_secret: str) -> dict[str, Any]:
        if not client_secret.startswith("ck_") or "." not in client_secret:
            raise TokenError("Malformed client_secret.")
        encoded_payload, signature = client_secret.removeprefix("ck_").split(".", 1)
        expected = self._sign(encoded_payload)
        if not hmac.compare_digest(signature, expected):
            raise TokenError("Invalid client_secret signature.")
        payload = json.loads(_b64decode(encoded_payload))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise TokenError("Expired client_secret.")
        return payload

    @staticmethod
    def fingerprint(client_secret: str) -> str:
        return hashlib.sha256(client_secret.encode("utf-8")).hexdigest()

    def _sign(self, encoded_payload: str) -> str:
        signature = hmac.new(
            self._secret,
            encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        return _b64encode(signature)

