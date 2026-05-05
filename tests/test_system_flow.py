from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from customer_service_system.agents import ConsensusAgent, DBAgent, VectorAgent
from customer_service_system.core import DBResult, EventType, SystemConfig, VectorResult, WorkerError
from customer_service_system.infrastructure import ConcurrencyQueue, EventLog
from customer_service_system.orchestration import Coordinator, Orchestrator


def event_log_path(name: str) -> Path:
    return Path("logs") / f"test-{name}-{uuid.uuid4().hex}.jsonl"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_happy_path_billing_turn():
    log_path = event_log_path("happy")
    orchestrator = Orchestrator(SystemConfig(event_log_path=str(log_path)))

    response = await orchestrator.process_text_turn("session-1", "What is my current balance?")

    assert "Your current balance is USD 142.00" in response
    assert orchestrator.get_history("session-1")[-1].role == "assistant"
    event_types = [event["event_type"] for event in read_jsonl(log_path)]
    assert EventType.HANDOFF_COMPLETED.value in event_types
    assert EventType.RESPONSE_SPOKEN.value in event_types


class SlowDBAgent(DBAgent):
    async def query(self, payload):
        await asyncio.sleep(0.05)
        return await super().query(payload)


class SlowVectorAgent(VectorAgent):
    async def search(self, payload):
        await asyncio.sleep(0.05)
        return await super().search(payload)


@pytest.mark.asyncio
async def test_coordinator_dispatches_workers_in_parallel():
    from customer_service_system.core import HandoffPayload, Intent

    log = EventLog(str(event_log_path("parallel")))
    coordinator = Coordinator(
        db_agent=SlowDBAgent(ConcurrencyQueue(2, 10)),
        vector_agent=SlowVectorAgent(),
        consensus_agent=ConsensusAgent(),
        event_log=log,
        per_task_timeout_seconds=1,
    )
    handoff = HandoffPayload(
        session_id="parallel",
        conversation_history=[],
        classified_intent=Intent.BILLING,
        entity_context={},
        triage_confidence=0.9,
        timestamp_utc="2026-05-05T00:00:00.000+00:00",
        call_reason_summary="balance inquiry",
    )

    started = asyncio.get_running_loop().time()
    result = await coordinator.run(handoff)
    elapsed = asyncio.get_running_loop().time() - started

    assert result.confidence > 0.8
    assert elapsed < 0.095


class FailingVectorAgent(VectorAgent):
    async def search(self, payload) -> VectorResult:
        raise RuntimeError("vector store is down")


@pytest.mark.asyncio
async def test_partial_worker_failure_still_reaches_consensus():
    from customer_service_system.core import HandoffPayload, Intent

    coordinator = Coordinator(
        db_agent=DBAgent(ConcurrencyQueue(2, 10)),
        vector_agent=FailingVectorAgent(),
        consensus_agent=ConsensusAgent(),
        event_log=EventLog(str(event_log_path("partial"))),
        per_task_timeout_seconds=1,
    )
    handoff = HandoffPayload(
        session_id="partial",
        conversation_history=[],
        classified_intent=Intent.BILLING,
        entity_context={},
        triage_confidence=0.9,
        timestamp_utc="2026-05-05T00:00:00.000+00:00",
        call_reason_summary="balance inquiry",
    )

    result = await coordinator.run(handoff)

    assert result.answer_facts["balance"] == 142.00
    assert result.vector.error is not None
    assert result.confidence < 0.95


@pytest.mark.asyncio
async def test_interruption_truncates_playback():
    log_path = event_log_path("interrupt")
    orchestrator = Orchestrator(SystemConfig(event_log_path=str(log_path)))
    orchestrator.tts.chunk_delay_seconds = 0.01

    task = asyncio.create_task(orchestrator.process_text_turn("interrupt", "What is my current balance?"))
    for _ in range(100):
        if orchestrator.audio.playback_lock.held:
            break
        await asyncio.sleep(0.002)

    orchestrator.handle_interruption("interrupt")
    spoken = await task

    assert spoken
    assert "due on 2026-06-01" not in spoken
    event_types = [event["event_type"] for event in read_jsonl(log_path)]
    assert EventType.INTERRUPTION_DETECTED.value in event_types
    assert EventType.PLAYBACK_TRUNCATED.value in event_types


@pytest.mark.asyncio
async def test_both_worker_failures_return_graceful_fallback():
    from customer_service_system.core import HandoffPayload, Intent

    class FailingDBAgent(DBAgent):
        async def query(self, payload) -> DBResult:
            raise RuntimeError("db is down")

    coordinator = Coordinator(
        db_agent=FailingDBAgent(ConcurrencyQueue(2, 10)),
        vector_agent=FailingVectorAgent(),
        consensus_agent=ConsensusAgent(),
        event_log=EventLog(str(event_log_path("failures"))),
        per_task_timeout_seconds=1,
    )
    handoff = HandoffPayload(
        session_id="failures",
        conversation_history=[],
        classified_intent=Intent.BILLING,
        entity_context={},
        triage_confidence=0.9,
        timestamp_utc="2026-05-05T00:00:00.000+00:00",
        call_reason_summary="balance inquiry",
    )

    result = await coordinator.run(handoff)

    assert result.answer_facts == {}
    assert "human agent" in result.user_guidance
    assert isinstance(result.db.error, WorkerError)
