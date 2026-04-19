# observability/logger.py
import json
import threading
from datetime import datetime
from pathlib import Path
from config.loader import config_loader


class ObsLogger:
    """
    Structured JSON observability logger.

    Emits one JSONL line per request with full cost, routing, and quality metadata.
    Compatible with jq, CloudWatch Log Insights, and Splunk queries.

    AWS Future Replacement: Amazon CloudWatch Logs
    ──────────────────────────────────────────────
    Replace JSONL file writes with CloudWatch Logs SDK:
    - boto3 logs.put_log_events() with the same JSON structure
    - CloudWatch Log Insights for ad-hoc queries (cost by tier, latency P95, etc.)
    - CloudWatch Metric Filters to extract custom metrics from log fields
    - CloudWatch Alarms on extracted metrics (error rate, cost_usd sum, etc.)
    - Log retention policies (e.g., 90 days) configured per log group

    Additionally:
    - AWS X-Ray for distributed tracing across Lambda invocations
    - CloudWatch Container Insights if migrating to ECS/Fargate

    Migration path:
      1. Create a CloudWatch Log Group: /fixit-ai/requests
      2. Replace file write with logs.put_log_events()
      3. Create metric filter for cost_usd → CloudWatch metric
      4. Set up dashboard with cost, latency, and fallback_rate widgets
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._file_lock = threading.Lock()
        self._output_path = None

    def _get_output_path(self) -> Path:
        if self._output_path is None:
            config = config_loader.get()
            self._output_path = Path(config.logging["output_path"])
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
        return self._output_path

    def log_request(self, response) -> dict:
        """
        Log a full request trace as a structured JSON line.
        response: QueryResponse dataclass from router/engine.py
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query_id": response.query_id,
            "session_id": getattr(response, "session_id", None),
            "category": response.category,
            "complexity": response.complexity,
            "tier_used": response.tier_used,
            "model_used": response.model_used,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "total_tokens": response.tokens_in + response.tokens_out,
            "cost_usd": response.cost_usd,
            "latency_ms": response.latency_ms,
            "routing_reason": response.routing_reason,
            "fallback_activated": response.fallback_activated,
            "budget_utilization_pct": getattr(response, "budget_utilization_pct", None),
            "prompt_version": getattr(response, "prompt_version", "v1"),
            "eval_score": None,  # Set by eval framework
            "error": None,
        }

        self._write(log_entry)
        return log_entry

    def log_error(self, query_id: str, error: str, context: dict = None) -> dict:
        """Log an error event."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query_id": query_id,
            "event_type": "error",
            "error": error,
            "context": context or {},
        }
        self._write(log_entry)
        return log_entry

    def _write(self, entry: dict):
        """Thread-safe write to JSONL file."""
        try:
            output_path = self._get_output_path()
            with self._file_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Never let logging errors crash the main request

    def get_recent_logs(self, n: int = 20) -> list:
        """Read the last N log entries from the JSONL file."""
        try:
            output_path = self._get_output_path()
            if not output_path.exists():
                return []
            with self._file_lock:
                lines = output_path.read_text(encoding="utf-8").strip().split("\n")
            lines = [l for l in lines if l.strip()]
            recent = lines[-n:]
            return [json.loads(line) for line in recent]
        except Exception:
            return []

    def get_cost_summary(self) -> dict:
        """Aggregate cost metrics from log file."""
        logs = self.get_recent_logs(n=10000)
        if not logs:
            return {"total_requests": 0, "total_cost_usd": 0.0, "tier_breakdown": {}}

        tier_breakdown: dict = {}
        total_cost = 0.0
        total_latency = 0.0
        fallback_count = 0

        for entry in logs:
            tier = entry.get("tier_used", "unknown")
            cost = entry.get("cost_usd", 0.0)
            latency = entry.get("latency_ms", 0.0)

            tier_breakdown.setdefault(tier, {"requests": 0, "cost_usd": 0.0})
            tier_breakdown[tier]["requests"] += 1
            tier_breakdown[tier]["cost_usd"] = round(
                tier_breakdown[tier]["cost_usd"] + cost, 6
            )
            total_cost += cost
            total_latency += latency
            if entry.get("fallback_activated"):
                fallback_count += 1

        n = len(logs)
        return {
            "total_requests": n,
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_per_request_usd": round(total_cost / n, 6) if n else 0,
            "avg_latency_ms": round(total_latency / n, 2) if n else 0,
            "fallback_activations": fallback_count,
            "fallback_rate_pct": round((fallback_count / n) * 100, 2) if n else 0,
            "tier_breakdown": tier_breakdown,
        }
