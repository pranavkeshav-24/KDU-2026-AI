from __future__ import annotations

from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    use_openai_llm: bool = env_bool("USE_OPENAI_LLM", False)

    enable_local_guardrails: bool = env_bool("ENABLE_LOCAL_GUARDRAILS", True)
    enable_cloud_safety: bool = env_bool("ENABLE_CLOUD_SAFETY", False)
    cloud_safety_provider: str = os.getenv("CLOUD_SAFETY_PROVIDER", "aws")
    cloud_safety_threshold: int = int(os.getenv("CLOUD_SAFETY_THRESHOLD", "4"))

    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    bedrock_guardrail_id: str | None = os.getenv("BEDROCK_GUARDRAIL_ID") or None
    bedrock_guardrail_version: str = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")

    langsmith_tracing: bool = env_bool("LANGSMITH_TRACING", False)
    langsmith_api_key: str | None = os.getenv("LANGSMITH_API_KEY")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "kdu-production-guardrails-lab")
    trace_sample_policy: str = os.getenv("TRACE_SAMPLE_POLICY", "guardrail_triggered")
    trace_sample_rate: float = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))
    trace_redact_pii: bool = env_bool("TRACE_REDACT_PII", True)


settings = Settings()
