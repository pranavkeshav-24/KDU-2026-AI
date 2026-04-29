from __future__ import annotations

from src.config import AppConfig


class OpenRouterClient:
    def __init__(self, config: AppConfig):
        self.config = config

    @property
    def available(self) -> bool:
        return bool(self.config.openrouter_api_key)

    def explain_unavailable(self) -> str:
        return "OpenRouter fallback is not configured. Set OPENROUTER_API_KEY to enable it."

