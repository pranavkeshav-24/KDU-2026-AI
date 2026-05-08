from __future__ import annotations

import re
from time import perf_counter

from safety_lab.models import GuardrailAction, GuardrailDecision


PROMPT_INJECTION_RULES: dict[str, re.Pattern[str]] = {
    "ignore_previous_instructions": re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+instructions\b", re.I),
    "debug_roleplay": re.compile(r"\b(debugging|developer|system)\s+(system|mode|console)\b", re.I),
    "print_stored_data": re.compile(r"\b(print|dump|show|reveal|exfiltrate)\b.*\b(stored|all|raw|private|sensitive)\s+(user\s+)?data\b", re.I),
    "instruction_hierarchy_attack": re.compile(r"\b(system prompt|hidden instructions|policy|guardrail)\b.*\b(print|reveal|ignore|bypass)\b", re.I),
    "tool_impersonation": re.compile(r"\b(call|invoke|simulate)\b.*\bbackend|database|admin|tool\b", re.I),
}

SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)


class LocalGuardrailEngine:
    """Open-source-style guardrail layer for prompt injection and PII masking.

    The interface is intentionally small so it can be swapped for NeMo Guardrails,
    Llama Guard, Presidio, or a hosted classifier without touching the chatbot.
    """

    def inspect_input(self, text: str) -> GuardrailDecision:
        start = perf_counter()
        triggered = [name for name, pattern in PROMPT_INJECTION_RULES.items() if pattern.search(text)]
        latency_ms = (perf_counter() - start) * 1000
        if triggered:
            return GuardrailDecision(
                action=GuardrailAction.BLOCK,
                reason="Prompt injection attempt detected.",
                text="I cannot follow instructions that try to bypass safety controls or expose private data.",
                triggered_rules=triggered,
                latency_ms=latency_ms,
            )
        return GuardrailDecision(
            action=GuardrailAction.ALLOW,
            reason="No input guardrail triggered.",
            text=text,
            latency_ms=latency_ms,
        )

    def inspect_output(self, text: str) -> GuardrailDecision:
        start = perf_counter()
        redacted = SSN_PATTERN.sub("[SSN_REDACTED]", text)
        triggered: list[str] = []
        if redacted != text:
            triggered.append("ssn_full_redaction")
        latency_ms = (perf_counter() - start) * 1000
        if triggered:
            return GuardrailDecision(
                action=GuardrailAction.MODIFY,
                reason="Sensitive SSN detected and masked in model output.",
                text=redacted,
                triggered_rules=triggered,
                latency_ms=latency_ms,
            )
        return GuardrailDecision(
            action=GuardrailAction.ALLOW,
            reason="No output guardrail triggered.",
            text=text,
            latency_ms=latency_ms,
        )

    def mask_for_logs(self, text: str) -> str:
        text = SSN_PATTERN.sub("[SSN_REDACTED]", text)
        return EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)


def asks_for_ssn_last4(text: str) -> bool:
    normalized = text.lower()
    return "ssn" in normalized and ("last 4" in normalized or "last four" in normalized)

