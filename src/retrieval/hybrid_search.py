from __future__ import annotations

from collections import defaultdict

from src.config import AppConfig, config
from src.cost.cost_tracker import UsageRecord
from src.retrieval.embedder import Embedder
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.vector_store import VectorStore
from src.storage.db import Database
from src.storage.models import SearchResult


class HybridSearch:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        app_config: AppConfig = config,
    ):
        self.db = db
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.config = app_config

    def search(
        self,
        query: str,
        mode: str = "Hybrid",
        top_k: int | None = None,
        file_id: str | None = None,
    ) -> list[SearchResult]:
        top_k = top_k or self.config.top_k_search_results
        semantic_results: list[tuple[str, float]] = []
        keyword_results: list[tuple[str, float]] = []
        query_tokens = 0
        query_cost = 0.0

        if mode in {"Hybrid", "Semantic"}:
            query_embedding, usage, provider = self.embedder.embed_query(query)
            query_tokens = usage.get("input_tokens", 0)
            model = self.config.embedding_model if provider == "openai" else "local-hash-embedding"
            if provider == "openai":
                query_cost = self.db.log_usage(
                    UsageRecord(
                        file_id=None,
                        operation="query_embedding",
                        provider=provider,
                        model=model,
                        input_tokens=query_tokens,
                    )
                )
            semantic_results = self.vector_store.search(query_embedding, top_k=top_k * 3, file_id=file_id)

        if mode in {"Hybrid", "Keyword"}:
            keyword_results = self.keyword_index.search(query, top_k=top_k * 3, file_id=file_id)

        fused = self._rrf(semantic_results, keyword_results, top_k)
        rows = self.db.fetch_chunks([chunk_id for chunk_id, _score, _match in fused])
        results: list[SearchResult] = []
        for chunk_id, score, match_type in fused:
            row = rows.get(chunk_id)
            if row:
                results.append(self.db.row_to_search_result(row, score, match_type))

        self.db.log_query(
            query=query,
            semantic_count=len(semantic_results),
            keyword_count=len(keyword_results),
            final_count=len(results),
            query_embedding_tokens=query_tokens,
            calculated_cost_usd=query_cost,
        )
        return results

    def _rrf(
        self,
        semantic_results: list[tuple[str, float]],
        keyword_results: list[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[str, float, str]]:
        scores: dict[str, float] = defaultdict(float)
        matches: dict[str, set[str]] = defaultdict(set)
        for rank, (chunk_id, _score) in enumerate(semantic_results, start=1):
            scores[chunk_id] += self.config.semantic_weight * (1 / (self.config.hybrid_search_rrf_k + rank))
            matches[chunk_id].add("Semantic")
        for rank, (chunk_id, _score) in enumerate(keyword_results, start=1):
            scores[chunk_id] += self.config.keyword_weight * (1 / (self.config.hybrid_search_rrf_k + rank))
            matches[chunk_id].add("Keyword")
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(chunk_id, score, " + ".join(sorted(matches[chunk_id]))) for chunk_id, score in ranked]

