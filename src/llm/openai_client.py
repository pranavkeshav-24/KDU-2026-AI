from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.llm.openrouter_client import OpenRouterClient
from src.utils.text_utils import estimate_tokens, sentence_summary, top_terms


class OpenAIClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self._client: Any | None = None
        self.provider = "local"
        self.fallback = OpenRouterClient(config)
        if config.openai_api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=config.openai_api_key)
                self.provider = "openai"
            except Exception:
                self._client = None
        elif config.openrouter_api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url)
                self.provider = "openrouter"
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def summarize_json(self, text: str, file_id: str | None = None) -> tuple[dict[str, Any], dict[str, int]]:
        sample = text[:24_000]
        if not self.available:
            terms = top_terms(sample, limit=8)
            summary = sentence_summary(sample, self.config.summary_word_limit)
            key_points = [f"Mentions {term}." for term in terms[:5]] or ["Readable text was extracted locally."]
            return (
                {
                    "summary": summary,
                    "key_points": key_points,
                    "topic_tags": terms[:10],
                    "accessibility_notes": "Generated locally because no OpenAI API key is configured.",
                },
                {"input_tokens": 0, "output_tokens": 0},
            )

        from src.llm.prompts import SUMMARY_SYSTEM_PROMPT

        input_tokens = estimate_tokens(SUMMARY_SYSTEM_PROMPT + sample)
        try:
            response = self._client.chat.completions.create(
                model=self._chat_model(),
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": sample},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            usage = getattr(response, "usage", None)
            parsed = json.loads(content)
            return parsed, {
                "input_tokens": getattr(usage, "input_tokens", None)
                or getattr(usage, "prompt_tokens", None)
                or input_tokens,
                "output_tokens": getattr(usage, "output_tokens", None)
                or getattr(usage, "completion_tokens", None)
                or estimate_tokens(content),
            }
        except Exception as exc:
            summary = sentence_summary(sample, self.config.summary_word_limit)
            return (
                {
                    "summary": summary,
                    "key_points": ["Summary API failed; local fallback was used."],
                    "topic_tags": top_terms(sample, limit=8),
                    "accessibility_notes": f"{self.provider} summary failed: {exc}",
                },
                {"input_tokens": input_tokens, "output_tokens": 0},
            )

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], dict[str, int], str]:
        if not texts:
            return [], {"input_tokens": 0}, "local"
        input_tokens = sum(estimate_tokens(text) for text in texts)
        if not self.available:
            from src.retrieval.embedder import local_embedding

            return [local_embedding(text) for text in texts], {"input_tokens": 0}, "local"
        if self.provider == "openrouter":
            from src.retrieval.embedder import local_embedding

            return [local_embedding(text) for text in texts], {"input_tokens": 0}, "local"
        try:
            response = self._client.embeddings.create(model=self.config.embedding_model, input=texts)
            embeddings = [item.embedding for item in response.data]
            usage = getattr(response, "usage", None)
            return embeddings, {
                "input_tokens": getattr(usage, "prompt_tokens", None)
                or getattr(usage, "total_tokens", None)
                or input_tokens
            }, "openai"
        except Exception:
            from src.retrieval.embedder import local_embedding

            return [local_embedding(text) for text in texts], {"input_tokens": 0}, "local"

    def vision_image_json(self, image_path: Path, prompt: str) -> tuple[dict[str, Any], dict[str, int], str]:
        if not self.available:
            return (
                {
                    "extracted_text": "",
                    "alt_text": f"Image file {image_path.name}. Vision analysis requires an OpenAI API key.",
                    "detailed_description": "No remote vision call was made.",
                    "objects_or_entities": [],
                    "warnings": ["Vision unavailable; configure OPENAI_API_KEY for OCR and descriptions."],
                },
                {"input_tokens": 0, "output_tokens": 0},
                "local",
            )
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_url = f"data:{mime};base64,{encoded}"
        input_tokens = estimate_tokens(prompt) + 1000
        try:
            response = self._client.chat.completions.create(
                model=self._vision_model(),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": self.config.vision_detail}},
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            usage = getattr(response, "usage", None)
            return json.loads(content), {
                "input_tokens": getattr(usage, "prompt_tokens", None) or input_tokens,
                "output_tokens": getattr(usage, "completion_tokens", None) or estimate_tokens(content),
            }, self.provider
        except Exception as exc:
            message = str(exc)
            return (
                {
                    "extracted_text": "",
                    "alt_text": "Vision analysis failed; see warnings for provider details.",
                    "detailed_description": f"{self.provider} vision failed: {message}",
                    "objects_or_entities": [],
                    "warnings": [message],
                },
                {"input_tokens": input_tokens, "output_tokens": 0},
                self.provider,
            )

    def _chat_model(self) -> str:
        if self.provider == "openrouter":
            return self.config.openrouter_model
        return self.config.llm_model

    def _vision_model(self) -> str:
        if self.provider == "openrouter":
            return self.config.openrouter_model
        return self.config.vision_model
