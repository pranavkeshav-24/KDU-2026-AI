from __future__ import annotations

from typing import List

import nltk
from transformers import AutoTokenizer

from config import CHUNK_MAX_TOKENS, SUMMARIZER_MODEL


class Preprocessor:
    def __init__(self, tokenizer_name: str = SUMMARIZER_MODEL, max_tokens: int = CHUNK_MAX_TOKENS) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self._ensure_nltk_resources()

    @staticmethod
    def _ensure_nltk_resources() -> None:
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

    def sentence_split(self, text: str) -> List[str]:
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]

    def chunk(self, text: str) -> List[str]:
        sentences = self.sentence_split(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            token_count = len(self.tokenizer.tokenize(sentence))

            # Keep extremely long sentences instead of dropping them.
            if token_count > self.max_tokens:
                if current:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0
                chunks.append(sentence)
                continue

            if current_len + token_count > self.max_tokens and current:
                chunks.append(" ".join(current))
                current, current_len = [sentence], token_count
            else:
                current.append(sentence)
                current_len += token_count

        if current:
            chunks.append(" ".join(current))

        return chunks
