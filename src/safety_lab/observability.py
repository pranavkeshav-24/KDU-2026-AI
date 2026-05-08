from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator
from uuid import uuid4

from safety_lab.models import SpanEvent


class TraceRecorder:
    def __init__(
        self,
        *,
        project_name: str,
        enabled_langsmith: bool = False,
        sample_policy: str = "guardrail_triggered",
        sample_rate: float = 1.0,
        redact_pii: bool = True,
        trace_path: Path = Path("reports/langsmith-local-traces.jsonl"),
    ) -> None:
        self.project_name = project_name
        self.enabled_langsmith = enabled_langsmith
        self.sample_policy = sample_policy
        self.sample_rate = sample_rate
        self.redact_pii = redact_pii
        self.trace_path = trace_path
        self.spans: list[SpanEvent] = []
        self._langsmith_client = self._build_langsmith_client() if enabled_langsmith else None

    @contextmanager
    def span(
        self,
        name: str,
        *,
        run_type: str = "chain",
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        start = perf_counter()
        outputs: dict[str, Any] = {}
        run_id = str(uuid4())
        if self._langsmith_client:
            self._create_langsmith_run(run_id, name, run_type, inputs or {}, metadata or {})
        try:
            yield outputs
        except Exception as exc:
            outputs["error"] = str(exc)
            raise
        finally:
            latency_ms = (perf_counter() - start) * 1000
            span = SpanEvent(
                name=name,
                run_type=run_type,
                inputs=self._safe_payload(inputs or {}),
                outputs=self._safe_payload(outputs),
                latency_ms=latency_ms,
                metadata=metadata or {},
            )
            self.spans.append(span)
            if self._langsmith_client:
                self._update_langsmith_run(run_id, outputs, latency_ms)

    def flush(self, *, guardrail_triggered: bool, failed: bool, total_latency_ms: float) -> None:
        if not self._should_sample(
            guardrail_triggered=guardrail_triggered,
            failed=failed,
            total_latency_ms=total_latency_ms,
        ):
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            for span in self.spans:
                handle.write(json.dumps(asdict(span), sort_keys=True) + "\n")

    def _should_sample(self, *, guardrail_triggered: bool, failed: bool, total_latency_ms: float) -> bool:
        if random.random() > self.sample_rate:
            return False
        if self.sample_policy == "all":
            return True
        if self.sample_policy == "failed":
            return failed
        if self.sample_policy == "high_latency":
            return total_latency_ms >= 1000
        return guardrail_triggered

    def _safe_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.redact_pii:
            return payload
        text = json.dumps(payload, default=str)
        text = text.replace("123-45-6789", "[SSN_REDACTED]")
        return json.loads(text)

    @staticmethod
    def _build_langsmith_client() -> Any | None:
        try:
            from langsmith import Client

            return Client()
        except Exception:
            return None

    def _create_langsmith_run(
        self,
        run_id: str,
        name: str,
        run_type: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        try:
            self._langsmith_client.create_run(
                id=run_id,
                name=name,
                run_type=run_type,
                inputs=self._safe_payload(inputs),
                project_name=self.project_name,
                extra={"metadata": metadata},
            )
        except Exception:
            self._langsmith_client = None

    def _update_langsmith_run(self, run_id: str, outputs: dict[str, Any], latency_ms: float) -> None:
        try:
            self._langsmith_client.update_run(
                run_id,
                outputs=self._safe_payload(outputs),
                extra={"metadata": {"latency_ms": latency_ms}},
            )
        except Exception:
            self._langsmith_client = None
