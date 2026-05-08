from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GuardrailAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"


@dataclass(frozen=True)
class CustomerRecord:
    user_id: str
    name: str
    email: str
    ssn: str

    @property
    def ssn_last4(self) -> str:
        return self.ssn[-4:]

    def as_sensitive_context(self) -> str:
        return f"Name: {self.name}\nEmail: {self.email}\nSSN: {self.ssn}"


@dataclass
class GuardrailDecision:
    action: GuardrailAction
    reason: str
    text: str
    triggered_rules: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def triggered(self) -> bool:
        return self.action != GuardrailAction.ALLOW


@dataclass
class CloudSafetyResult:
    provider: str
    blocked: bool
    categories: dict[str, int | str]
    reason: str
    latency_ms: float
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanEvent:
    name: str
    run_type: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatTurnResult:
    user_input: str
    raw_model_output: str
    final_output: str
    blocked: bool
    guardrail_events: list[GuardrailDecision]
    cloud_events: list[CloudSafetyResult]
    spans: list[SpanEvent]
    total_latency_ms: float

