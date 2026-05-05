from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Intent(str, Enum):
    BILLING = "BILLING"
    TECHNICAL = "TECHNICAL"
    ACCOUNT = "ACCOUNT"
    OTHER = "OTHER"


class AgentOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    TIMED_OUT = "TIMED_OUT"


class EventType(str, Enum):
    UTTERANCE_READY = "UTTERANCE_READY"
    INTERRUPTION_DETECTED = "INTERRUPTION_DETECTED"
    RESPONSE_TRUNCATED = "RESPONSE_TRUNCATED"
    PLAYBACK_TRUNCATED = "PLAYBACK_TRUNCATED"
    PLAYBACK_COMPLETE = "PLAYBACK_COMPLETE"
    AGENT_CALLED = "AGENT_CALLED"
    AGENT_RETURNED = "AGENT_RETURNED"
    HANDOFF_COMPLETED = "HANDOFF_COMPLETED"
    RESPONSE_SPOKEN = "RESPONSE_SPOKEN"
    QUEUE_REJECTED = "QUEUE_REJECTED"
    CIRCUIT_OPENED = "CIRCUIT_OPENED"
    CIRCUIT_HALF_OPEN = "CIRCUIT_HALF_OPEN"
    CIRCUIT_CLOSED = "CIRCUIT_CLOSED"


class WorkerName(str, Enum):
    DB = "DB Agent"
    VECTOR = "Vector Agent"


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Transcript:
    text: str
    confidence: float


@dataclass(frozen=True)
class HandoffPayload:
    session_id: str
    conversation_history: list[Message]
    classified_intent: Intent
    entity_context: dict[str, Any]
    triage_confidence: float
    timestamp_utc: str
    call_reason_summary: str


@dataclass(frozen=True)
class TriageResult:
    intent: Intent
    entity_context: dict[str, Any]
    confidence: float
    call_reason_summary: str


@dataclass(frozen=True)
class WorkerError:
    code: str
    message: str
    retryable: bool = True


@dataclass(frozen=True)
class DBResult:
    balance: float | None = None
    currency: str = "USD"
    due_date: str | None = None
    status: str = "UNKNOWN"
    records: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorPassage:
    title: str
    passage: str
    score: float


@dataclass(frozen=True)
class VectorResult:
    passages: list[VectorPassage]


@dataclass(frozen=True)
class WorkerEnvelope:
    worker: WorkerName
    outcome: AgentOutcome
    result: DBResult | VectorResult | None = None
    error: WorkerError | None = None
    latency_ms: int = 0


@dataclass(frozen=True)
class ConsensusPayload:
    session_id: str
    db: WorkerEnvelope
    vector: WorkerEnvelope
    confidence: float
    answer_facts: dict[str, Any]
    support_passages: list[VectorPassage]
    user_guidance: str


@dataclass(frozen=True)
class AgentUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass(frozen=True)
class AgentCallRecord:
    agent_name: str
    call_id: str
    session_id: str
    input_payload_hash: str
    tool_calls_made: list[dict[str, Any]]
    token_usage: AgentUsage
    latency_ms: int
    outcome: AgentOutcome
    error: WorkerError | None = None


@dataclass
class SessionState:
    session_id: str
    conversation_history: list[Message] = field(default_factory=list)
    active_response_text: str | None = None
    spoken_response_text: str = ""

    def append(self, role: str, content: str) -> None:
        self.conversation_history.append(Message(role=role, content=content))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value

