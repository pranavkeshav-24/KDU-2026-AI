from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from observability.logger import StructuredLogger


logger = StructuredLogger("context")


@dataclass
class ContextPayload:
    task_id: str
    user_intent: str
    employee_id: str | None = None
    routing_number: str | None = None
    account_number: str | None = None
    required_action: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value not in (None, {}, [])}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "ContextPayload":
        return cls(**json.loads(payload))


def build_context_payload(task_id: str, user_intent: str, **kwargs) -> ContextPayload:
    payload = ContextPayload(task_id=task_id, user_intent=user_intent, **kwargs)
    logger.log("context_payload_created", task_id=task_id, fields_included=sorted(payload.to_dict().keys()))
    return payload

