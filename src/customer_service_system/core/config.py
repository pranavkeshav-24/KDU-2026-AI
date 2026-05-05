from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


@dataclass(frozen=True)
class SystemConfig:
    """Runtime knobs for cost, latency, and capacity protection."""

    vad_trailing_silence_ms: int = 600
    agent_timeout_seconds: float = 4.0
    triage_history_token_budget: int = 2_000
    billing_history_token_budget: int = 3_000
    max_db_concurrent: int = 10
    max_db_queue_depth: int = 50
    llm_tokens_per_minute: int = 120_000
    llm_throttle_ratio: float = 0.8
    event_log_path: str = "logs/events.jsonl"

    @classmethod
    def from_env(cls) -> "SystemConfig":
        return cls(
            agent_timeout_seconds=_float_env("AGENT_TIMEOUT_SECONDS", 4.0),
            max_db_concurrent=_int_env("MAX_DB_CONCURRENT", 10),
            max_db_queue_depth=_int_env("MAX_DB_QUEUE_DEPTH", 50),
            llm_tokens_per_minute=_int_env("LLM_TOKENS_PER_MINUTE", 120_000),
        )

