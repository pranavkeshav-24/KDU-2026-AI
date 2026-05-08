from __future__ import annotations

from pathlib import Path

from safety_lab.chatbot import CustomerServiceBot
from safety_lab.cloud_safety import SafetyThresholds, build_cloud_safety_client
from safety_lab.config import Settings, settings
from safety_lab.guardrails import LocalGuardrailEngine
from safety_lab.llm import MockVulnerableLLM, OpenAIChatLLM
from safety_lab.observability import TraceRecorder


def build_bot(config: Settings = settings, *, force_base: bool = False) -> CustomerServiceBot:
    llm = (
        OpenAIChatLLM(config.openai_api_key, config.openai_model)
        if config.use_openai_llm and config.openai_api_key
        else MockVulnerableLLM()
    )
    guardrails = None if force_base or not config.enable_local_guardrails else LocalGuardrailEngine()
    thresholds = SafetyThresholds.uniform(config.cloud_safety_threshold)
    cloud_safety = None
    if config.enable_cloud_safety and not force_base:
        cloud_safety = build_cloud_safety_client(
            config.cloud_safety_provider,
            thresholds,
            aws_guardrail_id=config.bedrock_guardrail_id,
            aws_guardrail_version=config.bedrock_guardrail_version,
            aws_region=config.aws_region,
        )
    tracer = TraceRecorder(
        project_name=config.langsmith_project,
        enabled_langsmith=config.langsmith_tracing,
        sample_policy=config.trace_sample_policy,
        sample_rate=config.trace_sample_rate,
        redact_pii=config.trace_redact_pii,
        trace_path=Path("reports/langsmith-local-traces.jsonl"),
    )
    return CustomerServiceBot(llm=llm, guardrails=guardrails, cloud_safety=cloud_safety, tracer=tracer)
