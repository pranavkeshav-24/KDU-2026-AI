from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from customer_service_system.core import SystemConfig
from customer_service_system.infrastructure import EventLog
from customer_service_system.orchestration import Orchestrator
from customer_service_system.replay import ReplayEngine


@pytest.mark.asyncio
async def test_replay_reexecutes_logged_agent_payloads():
    log_path = Path("logs") / f"test-replay-{uuid.uuid4().hex}.jsonl"
    orchestrator = Orchestrator(SystemConfig(event_log_path=str(log_path)))
    await orchestrator.process_text_turn("replay", "What is my current balance?")

    engine = ReplayEngine(
        EventLog(str(log_path)),
        {
            "Triage Agent": lambda payload: {
                "intent": "BILLING",
                "entity_context": {},
                "confidence": 0.97,
                "call_reason_summary": f"Customer said: {payload['transcript'][:180]}",
            }
        },
    )

    diffs = await engine.replay_session("replay")

    assert diffs
    assert diffs[0].agent_name == "Triage Agent"
    assert diffs[0].changed is False
