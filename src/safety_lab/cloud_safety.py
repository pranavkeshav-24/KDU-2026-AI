from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from safety_lab.models import CloudSafetyResult


CLOUD_CATEGORIES = ("Hate", "Insults", "Sexual", "Violence")


@dataclass(frozen=True)
class SafetyThresholds:
    hate: int = 4
    insults: int = 4
    sexual: int = 4
    violence: int = 4

    @classmethod
    def uniform(cls, severity: int) -> "SafetyThresholds":
        return cls(hate=severity, insults=severity, sexual=severity, violence=severity)

    def for_category(self, category: str) -> int:
        return getattr(self, category.lower())


class CloudSafetyClient(Protocol):
    provider: str

    def analyze(self, text: str, *, source: str) -> CloudSafetyResult:
        ...


class LocalCloudSafetyClient:
    provider = "local-threshold-simulator"

    def __init__(self, thresholds: SafetyThresholds) -> None:
        self.thresholds = thresholds

    def analyze(self, text: str, *, source: str) -> CloudSafetyResult:
        start = perf_counter()
        categories = {
            "Hate": self._score(text, [r"\bhate\b.*\b(group|race|religion)\b", r"\bdehumanize\b"]),
            "Insults": self._score(text, [r"\bidiot\b", r"\bstupid\b", r"\bworthless\b"]),
            "Sexual": self._score(text, [r"\bexplicit sexual\b", r"\bnude\b", r"\bsex act\b"]),
            "Violence": self._score(text, [r"\bkill\b", r"\bsevere bleeding\b", r"\bgore\b", r"\bweapon\b"]),
        }
        blocked_categories = [
            category
            for category, score in categories.items()
            if score >= self.thresholds.for_category(category)
        ]
        latency_ms = (perf_counter() - start) * 1000
        return CloudSafetyResult(
            provider=self.provider,
            blocked=bool(blocked_categories),
            categories=categories,
            reason=(
                f"{source} blocked by categories: {', '.join(blocked_categories)}"
                if blocked_categories
                else f"{source} allowed by configured thresholds."
            ),
            latency_ms=latency_ms,
            raw_response={"source": source, "thresholds": self.thresholds.__dict__},
        )

    @staticmethod
    def _score(text: str, patterns: list[str]) -> int:
        for pattern in patterns:
            if re.search(pattern, text, re.I):
                if "severe bleeding" in pattern:
                    return 2
                return 6
        return 0


class BedrockGuardrailsClient:
    provider = "aws-bedrock-guardrails"

    def __init__(self, guardrail_id: str, guardrail_version: str, region: str) -> None:
        import boto3

        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def analyze(self, text: str, *, source: str) -> CloudSafetyResult:
        start = perf_counter()
        response = self.client.apply_guardrail(
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.guardrail_version,
            source="INPUT" if source == "input" else "OUTPUT",
            content=[{"text": {"text": text, "qualifiers": ["guard_content"]}}],
        )
        latency_ms = (perf_counter() - start) * 1000
        action = response.get("action")
        assessments = response.get("assessments", [])
        categories = self._extract_categories(assessments)
        return CloudSafetyResult(
            provider=self.provider,
            blocked=action == "GUARDRAIL_INTERVENED",
            categories=categories,
            reason=f"Bedrock action={action}",
            latency_ms=latency_ms,
            raw_response=response,
        )

    @staticmethod
    def _extract_categories(assessments: list[dict]) -> dict[str, str]:
        categories: dict[str, str] = {}
        for assessment in assessments:
            filters = assessment.get("contentPolicy", {}).get("filters", [])
            for item in filters:
                filter_type = item.get("type")
                confidence = item.get("confidence")
                action = item.get("action")
                if filter_type:
                    categories[filter_type] = f"{confidence}:{action}"
        return categories


def build_cloud_safety_client(
    provider: str,
    thresholds: SafetyThresholds,
    *,
    aws_guardrail_id: str | None = None,
    aws_guardrail_version: str = "DRAFT",
    aws_region: str = "us-east-1",
) -> CloudSafetyClient:
    provider = provider.lower()
    if provider in {"aws", "bedrock"}:
        if not aws_guardrail_id:
            raise ValueError("BEDROCK_GUARDRAIL_ID is required for AWS cloud safety.")
        return BedrockGuardrailsClient(aws_guardrail_id, aws_guardrail_version, aws_region)
    if provider == "local":
        return LocalCloudSafetyClient(thresholds)
    raise ValueError("Unsupported cloud safety provider. Use 'aws' for Bedrock Guardrails or 'local' for tests.")
