from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


class OpenAILLMClient:
    """Small streaming adapter that normalizes provider output to text deltas."""

    def __init__(self, api_key: str | None, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None
        if api_key:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=api_key)
            except ImportError:
                self._client = None

    async def stream_reply(self, *, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if not self._client:
            async for chunk in self._mock_reply(messages[-1]["content"]):
                yield chunk
            return

        prompt = self._to_responses_input(messages)
        try:
            async with self._client.responses.stream(
                model=self._model,
                input=prompt,
                instructions=(
                    "You are a concise travel booking assistant. Offer practical itinerary "
                    "help and let the server emit booking widgets when relevant."
                ),
            ) as stream:
                async for event in stream:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        yield event.delta
        except Exception as exc:
            yield f"\n\nI could not reach the configured LLM provider: {exc}"

    @staticmethod
    def _to_responses_input(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [{"role": item["role"], "content": item["content"]} for item in messages]

    @staticmethod
    async def _mock_reply(message: str) -> AsyncIterator[str]:
        response = (
            "I found a practical travel option and prepared the next action for you. "
            "Review the itinerary card below, then book only if the timing and fare work."
        )
        if "human" in message.lower() or "handoff" in message.lower():
            response = "I have paused AI handling and requested a human support handoff."
        for word in response.split(" "):
            yield word + " "
            await asyncio.sleep(0.02)

