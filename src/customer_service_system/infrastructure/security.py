from __future__ import annotations

import re
from typing import Any


SENSITIVE_KEY_PATTERN = re.compile(r"(api_)?key|secret|token|password|credential|url", re.IGNORECASE)


class SensitivePayloadError(ValueError):
    pass


def validate_no_sensitive_fields(payload: Any, path: str = "$") -> None:
    """Reject payloads that attempt to cross an agent boundary with secrets."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            next_path = f"{path}.{key_text}"
            if SENSITIVE_KEY_PATTERN.search(key_text):
                raise SensitivePayloadError(f"sensitive field is not allowed at {next_path}")
            validate_no_sensitive_fields(value, next_path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            validate_no_sensitive_fields(value, f"{path}[{index}]")

