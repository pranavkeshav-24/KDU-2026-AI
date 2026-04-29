from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.cost.cost_tracker import CostTracker, UsageRecord
from src.storage.models import Chunk, PageContent, SearchResult
from src.utils.file_utils import new_id


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size_bytes INTEGER,
                    upload_path TEXT,
                    status TEXT,
                    processing_mode TEXT,
                    created_at TEXT,
                    processed_at TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS pages (
                    page_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    page_number INTEGER,
                    text_char_count INTEGER,
                    text_area_ratio REAL,
                    image_count INTEGER,
                    image_area_ratio REAL,
                    page_type TEXT,
                    extraction_method TEXT,
                    vision_required INTEGER DEFAULT 0,
                    vision_enriched INTEGER DEFAULT 0,
                    warning_message TEXT,
                    FOREIGN KEY(file_id) REFERENCES files(file_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS visual_regions (
                    region_id TEXT PRIMARY KEY,
                    page_id TEXT NOT NULL,
                    region_type TEXT,
                    bbox_json TEXT,
                    image_path TEXT,
                    is_repeated INTEGER DEFAULT 0,
                    is_decorative INTEGER DEFAULT 0,
                    vision_status TEXT,
                    description TEXT,
                    FOREIGN KEY(page_id) REFERENCES pages(page_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS outputs (
                    output_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL UNIQUE,
                    full_text TEXT,
                    summary TEXT,
                    key_points_json TEXT,
                    topic_tags_json TEXT,
                    accessibility_notes TEXT,
                    created_at TEXT,
                    FOREIGN KEY(file_id) REFERENCES files(file_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    chunk_index INTEGER,
                    chunk_text TEXT,
                    page_start INTEGER,
                    page_end INTEGER,
                    timestamp_start REAL,
                    timestamp_end REAL,
                    source_type TEXT,
                    vision_enriched INTEGER DEFAULT 0,
                    vector_id TEXT,
                    created_at TEXT,
                    FOREIGN KEY(file_id) REFERENCES files(file_id) ON DELETE CASCADE
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS keyword_index USING fts5(
                    chunk_id UNINDEXED,
                    file_id UNINDEXED,
                    chunk_text,
                    tokens
                );

                CREATE TABLE IF NOT EXISTS vectors (
                    vector_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    model TEXT,
                    created_at TEXT,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS query_logs (
                    query_id TEXT PRIMARY KEY,
                    query_text TEXT,
                    semantic_result_count INTEGER,
                    keyword_result_count INTEGER,
                    final_result_count INTEGER,
                    query_embedding_tokens INTEGER,
                    calculated_cost_usd REAL,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS api_usage (
                    usage_id TEXT PRIMARY KEY,
                    file_id TEXT,
                    operation TEXT,
                    provider TEXT,
                    model TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    cached_input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    unit_input_cost REAL,
                    unit_cached_input_cost REAL,
                    unit_output_cost REAL,
                    calculated_cost_usd REAL,
                    created_at TEXT
                );
                """
            )

    def create_file(
        self,
        file_id: str,
        file_name: str,
        file_type: str,
        file_size_bytes: int,
        upload_path: str,
        processing_mode: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO files (
                    file_id, file_name, file_type, file_size_bytes, upload_path,
                    status, processing_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    file_name,
                    file_type,
                    file_size_bytes,
                    upload_path,
                    "uploaded",
                    processing_mode,
                    utc_now(),
                ),
            )

    def update_file_status(self, file_id: str, status: str, error_message: str | None = None) -> None:
        processed_at = utc_now() if status in {"completed", "failed"} else None
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE files
                SET status = ?, error_message = COALESCE(?, error_message),
                    processed_at = COALESCE(?, processed_at)
                WHERE file_id = ?
                """,
                (status, error_message, processed_at, file_id),
            )

    def get_file(self, file_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()

    def list_files(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM files ORDER BY created_at DESC").fetchall()

    def delete_file(self, file_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))

    def file_stats(self, file_id: str) -> dict[str, int]:
        with self.connect() as conn:
            pages = conn.execute("SELECT COUNT(*) AS count FROM pages WHERE file_id = ?", (file_id,)).fetchone()
            chunks = conn.execute("SELECT COUNT(*) AS count FROM chunks WHERE file_id = ?", (file_id,)).fetchone()
            vision = conn.execute(
                "SELECT COUNT(*) AS count FROM pages WHERE file_id = ? AND vision_enriched = 1",
                (file_id,),
            ).fetchone()
        return {
            "pages": int(pages["count"]),
            "chunks": int(chunks["count"]),
            "vision_pages": int(vision["count"]),
        }

    def list_pages(self, file_id: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM pages WHERE file_id = ? ORDER BY page_number",
                (file_id,),
            ).fetchall()

    def replace_pages(self, file_id: str, pages: Iterable[PageContent]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM pages WHERE file_id = ?", (file_id,))
            for page in pages:
                conn.execute(
                    """
                    INSERT INTO pages (
                        page_id, file_id, page_number, text_char_count, text_area_ratio,
                        image_count, image_area_ratio, page_type, extraction_method,
                        vision_required, vision_enriched, warning_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{file_id}_p{page.page_number}",
                        file_id,
                        page.page_number,
                        page.text_char_count,
                        page.text_area_ratio,
                        page.image_count,
                        page.image_area_ratio,
                        page.page_type,
                        page.extraction_method,
                        int(page.vision_required),
                        int(page.vision_enriched),
                        "\n".join(page.warnings),
                    ),
                )

    def upsert_output(
        self,
        file_id: str,
        full_text: str,
        summary: str,
        key_points: list[str],
        topic_tags: list[str],
        accessibility_notes: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO outputs (
                    output_id, file_id, full_text, summary, key_points_json,
                    topic_tags_json, accessibility_notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    full_text = excluded.full_text,
                    summary = excluded.summary,
                    key_points_json = excluded.key_points_json,
                    topic_tags_json = excluded.topic_tags_json,
                    accessibility_notes = excluded.accessibility_notes,
                    created_at = excluded.created_at
                """,
                (
                    new_id("out"),
                    file_id,
                    full_text,
                    summary,
                    json.dumps(key_points),
                    json.dumps(topic_tags),
                    accessibility_notes,
                    utc_now(),
                ),
            )

    def get_output(self, file_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM outputs WHERE file_id = ?", (file_id,)).fetchone()

    def replace_chunks(self, file_id: str, chunks: list[Chunk]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM keyword_index WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM vectors WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, file_id, chunk_index, chunk_text, page_start,
                        page_end, timestamp_start, timestamp_end, source_type,
                        vision_enriched, vector_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.file_id,
                        chunk.chunk_index,
                        chunk.chunk_text,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.timestamp_start,
                        chunk.timestamp_end,
                        chunk.source_type,
                        int(chunk.vision_enriched),
                        chunk.vector_id,
                        utc_now(),
                    ),
                )
                conn.execute(
                    "INSERT INTO keyword_index (chunk_id, file_id, chunk_text, tokens) VALUES (?, ?, ?, ?)",
                    (chunk.chunk_id, chunk.file_id, chunk.chunk_text, chunk.chunk_text),
                )

    def store_vectors(self, vectors: list[tuple[str, str, str, list[float], str]]) -> None:
        with self.connect() as conn:
            for vector_id, chunk_id, file_id, embedding, model in vectors:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO vectors (
                        vector_id, chunk_id, file_id, embedding_json, model, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (vector_id, chunk_id, file_id, json.dumps(embedding), model, utc_now()),
                )
                conn.execute("UPDATE chunks SET vector_id = ? WHERE chunk_id = ?", (vector_id, chunk_id))

    def list_vectors(self, file_id: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as conn:
            if file_id:
                return conn.execute("SELECT * FROM vectors WHERE file_id = ?", (file_id,)).fetchall()
            return conn.execute("SELECT * FROM vectors").fetchall()

    def fetch_chunks(self, chunk_ids: list[str]) -> dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*, f.file_name
                FROM chunks c
                JOIN files f ON f.file_id = c.file_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        return {row["chunk_id"]: row for row in rows}

    def keyword_search(
        self,
        query: str,
        top_k: int,
        file_id: str | None = None,
    ) -> list[tuple[str, float]]:
        query = query.strip()
        if not query:
            return []
        sql = """
            SELECT chunk_id, bm25(keyword_index) AS score
            FROM keyword_index
            WHERE keyword_index MATCH ?
        """
        params: list[Any] = [query]
        if file_id:
            sql += " AND file_id = ?"
            params.append(file_id)
        sql += " ORDER BY score LIMIT ?"
        params.append(top_k)
        try:
            with self.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            escaped = " OR ".join(part.replace('"', '""') for part in query.split())
            if not escaped:
                return []
            with self.connect() as conn:
                rows = conn.execute(sql, [escaped, *params[1:]]).fetchall()
        return [(row["chunk_id"], 1.0 / (1.0 + abs(float(row["score"])))) for row in rows]

    def log_usage(self, usage: UsageRecord) -> float:
        tracker = CostTracker()
        cost = tracker.calculate_cost(
            usage.model,
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.output_tokens,
        )
        unit_input, unit_cached, unit_output = tracker.units(usage.model)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO api_usage (
                    usage_id, file_id, operation, provider, model, input_tokens,
                    cached_input_tokens, output_tokens, total_tokens, unit_input_cost,
                    unit_cached_input_cost, unit_output_cost, calculated_cost_usd, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("usage"),
                    usage.file_id,
                    usage.operation,
                    usage.provider,
                    usage.model,
                    usage.input_tokens,
                    usage.cached_input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    unit_input,
                    unit_cached,
                    unit_output,
                    cost,
                    utc_now(),
                ),
            )
        return cost

    def log_query(
        self,
        query: str,
        semantic_count: int,
        keyword_count: int,
        final_count: int,
        query_embedding_tokens: int,
        calculated_cost_usd: float,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO query_logs (
                    query_id, query_text, semantic_result_count, keyword_result_count,
                    final_result_count, query_embedding_tokens, calculated_cost_usd, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("query"),
                    query,
                    semantic_count,
                    keyword_count,
                    final_count,
                    query_embedding_tokens,
                    calculated_cost_usd,
                    utc_now(),
                ),
            )

    def usage_summary(self) -> dict[str, Any]:
        with self.connect() as conn:
            total = conn.execute("SELECT COALESCE(SUM(calculated_cost_usd), 0) AS total FROM api_usage").fetchone()
            by_file = conn.execute(
                """
                SELECT f.file_name, COALESCE(SUM(a.calculated_cost_usd), 0) AS total
                FROM files f
                LEFT JOIN api_usage a ON a.file_id = f.file_id
                GROUP BY f.file_id
                ORDER BY total DESC
                """
            ).fetchall()
            by_operation = conn.execute(
                """
                SELECT operation, COUNT(*) AS calls, COALESCE(SUM(calculated_cost_usd), 0) AS total
                FROM api_usage
                GROUP BY operation
                ORDER BY total DESC
                """
            ).fetchall()
            by_model = conn.execute(
                """
                SELECT model, COUNT(*) AS calls, COALESCE(SUM(calculated_cost_usd), 0) AS total
                FROM api_usage
                GROUP BY model
                ORDER BY total DESC
                """
            ).fetchall()
            local_ops = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM api_usage
                WHERE calculated_cost_usd = 0
                """
            ).fetchone()
            query_count = conn.execute("SELECT COUNT(*) AS count FROM query_logs").fetchone()
            vision = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN vision_required = 1 THEN 1 ELSE 0 END) AS required,
                    SUM(CASE WHEN vision_required = 0 AND image_count > 0 THEN 1 ELSE 0 END) AS avoided
                FROM pages
                """
            ).fetchone()
        return {
            "total": float(total["total"]),
            "by_file": by_file,
            "by_operation": by_operation,
            "by_model": by_model,
            "local_operations": int(local_ops["count"]),
            "query_embeddings": int(query_count["count"]),
            "vision_required": int(vision["required"] or 0),
            "vision_avoided": int(vision["avoided"] or 0),
        }

    def row_to_search_result(self, row: sqlite3.Row, score: float, match_type: str) -> SearchResult:
        return SearchResult(
            chunk_id=row["chunk_id"],
            file_id=row["file_id"],
            file_name=row["file_name"],
            chunk_text=row["chunk_text"],
            score=score,
            match_type=match_type,
            page_start=row["page_start"],
            page_end=row["page_end"],
            source_type=row["source_type"],
            vision_enriched=bool(row["vision_enriched"]),
        )
