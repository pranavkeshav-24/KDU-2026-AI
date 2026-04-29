from __future__ import annotations

from src.storage.db import Database


class KeywordIndex:
    def __init__(self, db: Database):
        self.db = db

    def search(self, query: str, top_k: int, file_id: str | None = None) -> list[tuple[str, float]]:
        return self.db.keyword_search(query, top_k=top_k, file_id=file_id)

