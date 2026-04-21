from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import MERGED_REDUCTION_TOKEN_THRESHOLD, SUMMARIZER_MODEL


class BartSummarizer:
    """
    Abstractive summarizer powered by BART.

    Stage behavior:
    1) Summarize each chunk independently.
    2) Merge partial summaries.
    3) If merged text is still long, run one additional reduction pass.
    """

    def __init__(self, model_name: str = SUMMARIZER_MODEL) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def _summarize_once(self, text: str, min_len: int, max_len: int) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                min_length=min_len,
                max_length=max_len,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def summarize_chunks(self, chunks: List[str]) -> tuple[List[str], str]:
        partial_summaries: List[str] = []

        for chunk in chunks:
            summary = self._summarize_once(chunk, min_len=45, max_len=180)
            partial_summaries.append(summary)

        merged = " ".join(partial_summaries).strip()

        merged_token_len = len(self.tokenizer.tokenize(merged))
        if merged_token_len > MERGED_REDUCTION_TOKEN_THRESHOLD:
            merged = self._summarize_once(merged, min_len=80, max_len=260)

        return partial_summaries, merged
