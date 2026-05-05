from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from ..core import (
    ConsensusPayload,
    DBResult,
    HandoffPayload,
    Intent,
    TriageResult,
    VectorPassage,
    VectorResult,
    WorkerError,
)
from ..infrastructure import CircuitBreaker, ConcurrencyQueue, QueueFullError


class TriageAgent:
    async def classify(self, transcript: str, session_id: str) -> TriageResult:
        text = transcript.lower()
        if any(term in text for term in ("bill", "billing", "balance", "invoice", "payment", "charge")):
            intent = Intent.BILLING
            confidence = 0.97
        elif any(term in text for term in ("login", "password", "account", "profile")):
            intent = Intent.ACCOUNT
            confidence = 0.9
        elif any(term in text for term in ("broken", "error", "technical", "bug", "not working")):
            intent = Intent.TECHNICAL
            confidence = 0.88
        else:
            intent = Intent.OTHER
            confidence = 0.68

        entities: dict[str, Any] = {}
        account_match = re.search(r"\baccount\s+([A-Za-z0-9-]+)\b", transcript, re.IGNORECASE)
        if account_match:
            entities["account_id"] = account_match.group(1)
        amount_match = re.search(r"\$?(\d+(?:\.\d{2})?)", transcript)
        if amount_match:
            entities["amount"] = float(amount_match.group(1))

        return TriageResult(
            intent=intent,
            entity_context=entities,
            confidence=confidence,
            call_reason_summary=f"Customer said: {transcript[:180]}",
        )


@dataclass
class DBAgent:
    queue: ConcurrencyQueue
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    account_records: dict[str, DBResult] = field(
        default_factory=lambda: {
            "default": DBResult(balance=142.00, currency="USD", due_date="2026-06-01", status="CURRENT"),
            "past-due": DBResult(balance=293.55, currency="USD", due_date="2026-04-15", status="PAST_DUE"),
        }
    )

    async def query(self, payload: dict[str, Any]) -> DBResult:
        self.circuit_breaker.before_call()
        try:
            async with self.queue.slot():
                await asyncio.sleep(0)
                account_id = payload.get("filters", {}).get("account_id") or "default"
                result = self.account_records.get(account_id, self.account_records["default"])
                self.circuit_breaker.record_success()
                return result
        except QueueFullError:
            raise
        except Exception:
            self.circuit_breaker.record_failure()
            raise


@dataclass
class VectorAgent:
    passages: list[VectorPassage] = field(
        default_factory=lambda: [
            VectorPassage(
                title="Balance inquiry FAQ",
                passage="Customers can ask for their current balance, payment due date, and billing status.",
                score=0.0,
            ),
            VectorPassage(
                title="Payment grace period policy",
                passage="Accounts in good standing have a short grace period after the due date before late fees apply.",
                score=0.0,
            ),
            VectorPassage(
                title="Disputed charge procedure",
                passage="If a customer disputes a charge, collect the date, amount, and merchant before escalating.",
                score=0.0,
            ),
        ]
    )

    async def search(self, payload: dict[str, Any]) -> VectorResult:
        await asyncio.sleep(0)
        query = payload.get("query", "")
        max_results = int(payload.get("max_results", 3))
        query_terms = {term.lower() for term in re.findall(r"\w+", query)}
        ranked: list[VectorPassage] = []
        for passage in self.passages:
            passage_terms = {term.lower() for term in re.findall(r"\w+", passage.title + " " + passage.passage)}
            overlap = len(query_terms & passage_terms)
            score = overlap / max(1, len(query_terms))
            ranked.append(VectorPassage(passage.title, passage.passage, score))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return VectorResult(passages=ranked[:max_results])


class ConsensusAgent:
    async def reconcile(
        self,
        session_id: str,
        db_result: DBResult | None,
        db_error: WorkerError | None,
        vector_result: VectorResult | None,
        vector_error: WorkerError | None,
    ) -> ConsensusPayload:
        facts: dict[str, Any] = {}
        passages = vector_result.passages if vector_result else []
        confidence = 0.95

        if db_result:
            facts.update(
                {
                    "balance": db_result.balance,
                    "currency": db_result.currency,
                    "due_date": db_result.due_date,
                    "status": db_result.status,
                }
            )
        else:
            confidence -= 0.35
        if vector_error:
            confidence -= 0.15
        if db_error and vector_error:
            confidence = 0.1

        if db_error and vector_error:
            guidance = "I was unable to retrieve the information needed. A human agent will follow up."
        elif db_error:
            guidance = "Use knowledge-base guidance only and invite the customer to retry account-specific lookup."
        elif vector_error:
            guidance = "Use account records only; knowledge-base support was unavailable."
        else:
            guidance = "Use account records and supporting policy context."

        from ..core import AgentOutcome, WorkerEnvelope, WorkerName

        return ConsensusPayload(
            session_id=session_id,
            db=WorkerEnvelope(
                worker=WorkerName.DB,
                outcome=AgentOutcome.SUCCESS if db_result else AgentOutcome.FAILURE,
                result=db_result,
                error=db_error,
            ),
            vector=WorkerEnvelope(
                worker=WorkerName.VECTOR,
                outcome=AgentOutcome.SUCCESS if vector_result else AgentOutcome.FAILURE,
                result=vector_result,
                error=vector_error,
            ),
            confidence=max(0.0, min(1.0, confidence)),
            answer_facts=facts,
            support_passages=passages,
            user_guidance=guidance,
        )


class BillingAgent:
    async def answer(self, handoff: HandoffPayload, consensus: ConsensusPayload) -> str:
        if not consensus.answer_facts:
            return consensus.user_guidance

        balance = consensus.answer_facts.get("balance")
        currency = consensus.answer_facts.get("currency", "USD")
        due_date = consensus.answer_facts.get("due_date")
        status = consensus.answer_facts.get("status", "UNKNOWN")

        if handoff.classified_intent is not Intent.BILLING:
            return (
                "I can help with that, and I also checked your billing context. "
                f"Your account status is {status.lower()}."
            )

        response = f"Your current balance is {currency} {balance:.2f}"
        if due_date:
            response += f", due on {due_date}"
        response += f". Your billing status is {status.lower().replace('_', ' ')}."
        if consensus.confidence < 0.8:
            response += " I have partial information right now, so a human agent may verify this if needed."
        return response
