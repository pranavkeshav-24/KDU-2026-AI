from __future__ import annotations

import re
from collections import Counter


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"(?<![.!?:;])\n(?=[a-z0-9])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.33))


def top_terms(text: str, limit: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "not",
        "you",
        "your",
        "content",
        "page",
    }
    counts = Counter(word for word in words if word not in stop)
    return [word for word, _count in counts.most_common(limit)]


def sentence_summary(text: str, word_limit: int = 150) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return "No readable content was extracted."
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    selected: list[str] = []
    words = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if words + len(sentence_words) > word_limit and selected:
            break
        selected.append(sentence)
        words += len(sentence_words)
    return " ".join(selected).strip()

