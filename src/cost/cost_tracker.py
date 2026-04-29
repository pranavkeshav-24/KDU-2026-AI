from __future__ import annotations

from dataclasses import dataclass

from src.cost.pricing import PRICING


@dataclass
class UsageRecord:
    file_id: str | None
    operation: str
    provider: str
    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cached_input_tokens + self.output_tokens


class CostTracker:
    def calculate_cost(
        self,
        model: str,
        input_tokens: int = 0,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        pricing = PRICING.get(model, {})
        return round(
            (input_tokens / 1_000_000) * pricing.get("input_per_1m", 0.0)
            + (cached_input_tokens / 1_000_000) * pricing.get("cached_input_per_1m", 0.0)
            + (output_tokens / 1_000_000) * pricing.get("output_per_1m", 0.0),
            8,
        )

    def units(self, model: str) -> tuple[float, float, float]:
        pricing = PRICING.get(model, {})
        return (
            pricing.get("input_per_1m", 0.0),
            pricing.get("cached_input_per_1m", 0.0),
            pricing.get("output_per_1m", 0.0),
        )

