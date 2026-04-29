from __future__ import annotations

import json
import math

from src.storage.db import Database


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(length))
    left_norm = math.sqrt(sum(value * value for value in left[:length])) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right[:length])) or 1.0
    return dot / (left_norm * right_norm)


class VectorStore:
    def __init__(self, db: Database):
        self.db = db

    def upsert(self, vectors: list[tuple[str, str, str, list[float], str]]) -> None:
        self.db.store_vectors(vectors)

    def search(self, query_embedding: list[float], top_k: int, file_id: str | None = None) -> list[tuple[str, float]]:
        rows = self.db.list_vectors(file_id=file_id)
        scored: list[tuple[str, float]] = []
        for row in rows:
            embedding = json.loads(row["embedding_json"])
            scored.append((row["chunk_id"], cosine_similarity(query_embedding, embedding)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

