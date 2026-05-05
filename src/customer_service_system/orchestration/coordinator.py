from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable

from ..agents import ConsensusAgent, DBAgent, VectorAgent
from ..core import (
    AgentOutcome,
    ConsensusPayload,
    DBResult,
    EventType,
    HandoffPayload,
    VectorResult,
    WorkerEnvelope,
    WorkerError,
    WorkerName,
)
from ..infrastructure import EventLog, QueueFullError, monitored_agent_call, validate_no_sensitive_fields


@dataclass
class Coordinator:
    db_agent: DBAgent
    vector_agent: VectorAgent
    consensus_agent: ConsensusAgent
    event_log: EventLog
    per_task_timeout_seconds: float = 4.0

    async def run(self, handoff: HandoffPayload) -> ConsensusPayload:
        db_payload = {
            "session_id": handoff.session_id,
            "query_intent": handoff.classified_intent.value,
            "filters": handoff.entity_context,
        }
        vector_payload = {
            "session_id": handoff.session_id,
            "query": handoff.call_reason_summary,
            "max_results": 3,
        }
        validate_no_sensitive_fields(db_payload)
        validate_no_sensitive_fields(vector_payload)

        db_task = asyncio.create_task(self._run_worker(WorkerName.DB, handoff.session_id, db_payload))
        vector_task = asyncio.create_task(self._run_worker(WorkerName.VECTOR, handoff.session_id, vector_payload))
        db_envelope, vector_envelope = await asyncio.gather(db_task, vector_task)

        try:
            return await monitored_agent_call(
                agent_name="Consensus Agent",
                session_id=handoff.session_id,
                payload={"db": db_envelope, "vector": vector_envelope},
                event_log=self.event_log,
                call=lambda: self.consensus_agent.reconcile(
                    session_id=handoff.session_id,
                    db_result=db_envelope.result if isinstance(db_envelope.result, DBResult) else None,
                    db_error=db_envelope.error,
                    vector_result=vector_envelope.result if isinstance(vector_envelope.result, VectorResult) else None,
                    vector_error=vector_envelope.error,
                ),
            )
        except Exception as exc:
            self.event_log.append(
                EventType.AGENT_RETURNED,
                handoff.session_id,
                {"agent_name": "Consensus Agent", "fallback": True, "error": str(exc)},
            )
            return await self.consensus_agent.reconcile(
                handoff.session_id,
                None,
                WorkerError(code="CONSENSUS_FAILURE", message=str(exc), retryable=False),
                None,
                WorkerError(code="CONSENSUS_FAILURE", message=str(exc), retryable=False),
            )

    async def _run_worker(
        self,
        worker: WorkerName,
        session_id: str,
        payload: dict[str, Any],
    ) -> WorkerEnvelope:
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._call_worker(worker, session_id, payload),
                timeout=self.per_task_timeout_seconds,
            )
            return WorkerEnvelope(
                worker=worker,
                outcome=AgentOutcome.SUCCESS,
                result=result,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except asyncio.TimeoutError:
            return WorkerEnvelope(
                worker=worker,
                outcome=AgentOutcome.TIMED_OUT,
                error=WorkerError(code="TIMED_OUT", message=f"{worker.value} exceeded timeout"),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except QueueFullError as exc:
            self.event_log.append(EventType.QUEUE_REJECTED, session_id, {"worker": worker.value, "error": str(exc)})
            return WorkerEnvelope(
                worker=worker,
                outcome=AgentOutcome.FAILURE,
                error=WorkerError(code="QUEUE_FULL", message=str(exc)),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            return WorkerEnvelope(
                worker=worker,
                outcome=AgentOutcome.FAILURE,
                error=WorkerError(code=type(exc).__name__, message=str(exc)),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

    async def _call_worker(
        self,
        worker: WorkerName,
        session_id: str,
        payload: dict[str, Any],
    ) -> DBResult | VectorResult:
        if worker is WorkerName.DB:
            call: Awaitable[DBResult] = monitored_agent_call(
                agent_name=worker.value,
                session_id=session_id,
                payload=payload,
                event_log=self.event_log,
                call=lambda: self.db_agent.query(payload),
            )
            return await call
        call_vector: Awaitable[VectorResult] = monitored_agent_call(
            agent_name=worker.value,
            session_id=session_id,
            payload=payload,
            event_log=self.event_log,
            call=lambda: self.vector_agent.search(payload),
        )
        return await call_vector
