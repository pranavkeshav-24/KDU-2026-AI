from __future__ import annotations

from transformers import pipeline

from config import QA_CONFIDENCE_THRESHOLD, QA_MODEL


class QAModel:
    def __init__(self, model_name: str = QA_MODEL) -> None:
        self.pipe = pipeline("question-answering", model=model_name, tokenizer=model_name)

    def answer(self, question: str, context: str) -> dict:
        output = self.pipe(
            question=question,
            context=context,
        )

        answer_text = output.get("answer", "").strip()
        score = float(output.get("score", 0.0))

        if not answer_text:
            answer_text = "Not enough information in the summary."
            score = min(score, 0.2)

        return {
            "answer": answer_text,
            "score": score,
            "low_confidence": score < QA_CONFIDENCE_THRESHOLD,
        }
