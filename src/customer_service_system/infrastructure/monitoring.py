from __future__ import annotations

import time
import uuid
from hashlib import sha256
from typing import Any, Awaitable, Callable, TypeVar

from ..core import AgentCallRecord, AgentOutcome, AgentUsage, EventType, WorkerError, to_jsonable
from .events import EventLog
from .security import validate_no_sensitive_fields
from .token_pruning import estimate_tokens

T = TypeVar("T")


def payload_hash(payload: Any) -> str:
    import json

    return sha256(
        json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


async def monitored_agent_call(
    *,
    agent_name: str,
    session_id: str,
    payload: Any,
    event_log: EventLog,
    call: Callable[[], Awaitable[T]],
) -> T:
    validate_no_sensitive_fields(to_jsonable(payload))
    call_id = str(uuid.uuid4())
    event_log.append(
        EventType.AGENT_CALLED,
        session_id,
        {
            "agent_name": agent_name,
            "call_id": call_id,
            "input_payload_hash": payload_hash(payload),
            "payload": payload,
        },
    )
    started = time.perf_counter()
    try:
        result = await call()
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        record = AgentCallRecord(
            agent_name=agent_name,
            call_id=call_id,
            session_id=session_id,
            input_payload_hash=payload_hash(payload),
            tool_calls_made=[],
            token_usage=AgentUsage(prompt_tokens=estimate_tokens(str(payload))),
            latency_ms=latency_ms,
            outcome=AgentOutcome.FAILURE,
            error=WorkerError(code=type(exc).__name__, message=str(exc)),
        )
        event_log.append(
            EventType.AGENT_RETURNED,
            session_id,
            {"record": record, "result": None},
            duration_ms=latency_ms,
        )
        raise
    latency_ms = int((time.perf_counter() - started) * 1000)
    record = AgentCallRecord(
        agent_name=agent_name,
        call_id=call_id,
        session_id=session_id,
        input_payload_hash=payload_hash(payload),
        tool_calls_made=[],
        token_usage=AgentUsage(
            prompt_tokens=estimate_tokens(str(payload)),
            completion_tokens=estimate_tokens(str(result)),
            estimated_cost_usd=0.0,
        ),
        latency_ms=latency_ms,
        outcome=AgentOutcome.SUCCESS,
    )
    event_log.append(
        EventType.AGENT_RETURNED,
        session_id,
        {"record": record, "result": result},
        duration_ms=latency_ms,
    )
    return result
