from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import LENGTH_CONFIG, REFINER_MODEL


class Refiner:
    def __init__(self, model_name: str = REFINER_MODEL) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def refine(self, summary: str, length: str) -> str:
        cfg = LENGTH_CONFIG[length]
        input_len = len(self.tokenizer.tokenize(summary))

        min_len = min(cfg["min_length"], max(10, input_len - 5))
        max_len = max(cfg["max_length"], min_len + 10)

        inputs = self.tokenizer(
            summary,
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
