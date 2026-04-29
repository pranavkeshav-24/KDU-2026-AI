from __future__ import annotations

import json
from pathlib import Path

from src.config import AppConfig, config
from src.cost.cost_tracker import UsageRecord
from src.llm.openai_client import OpenAIClient
from src.processors.audio_processor import AudioProcessor
from src.processors.image_processor import ImageProcessor
from src.processors.pdf_processor import PDFProcessor
from src.retrieval.chunker import Chunker
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.storage.db import Database
from src.storage.models import UnifiedDocument
from src.vision.vision_client import VisionClient


class ProcessingOrchestrator:
    def __init__(self, db: Database, app_config: AppConfig = config):
        self.db = db
        self.config = app_config
        self.llm_client = OpenAIClient(app_config)
        self.vision_client = VisionClient(self.llm_client)
        self.chunker = Chunker(app_config)
        self.embedder = Embedder(self.llm_client, app_config)
        self.vector_store = VectorStore(db)

    def process_file(self, file_id: str) -> UnifiedDocument:
        record = self.db.get_file(file_id)
        if record is None:
            raise ValueError(f"Unknown file_id: {file_id}")

        self.db.update_file_status(file_id, "processing")
        path = Path(record["upload_path"])
        try:
            document = self._extract(record["file_type"], file_id, record["file_name"], path, record["processing_mode"])
            self.db.replace_pages(file_id, document.pages)
            self._summarize(document)
            chunks = self.chunker.chunk_document(document)
            self.db.replace_chunks(file_id, chunks)
            vectors, usage, provider = self.embedder.embed_chunks(chunks)
            self.vector_store.upsert(vectors)
            if provider == "openai":
                self.db.log_usage(
                    UsageRecord(
                        file_id=file_id,
                        operation="chunk_embeddings",
                        provider=provider,
                        model=self.config.embedding_model,
                        input_tokens=usage.get("input_tokens", 0),
                    )
                )
            else:
                self.db.log_usage(
                    UsageRecord(
                        file_id=file_id,
                        operation="chunk_embeddings",
                        provider=provider,
                        model="local-hash-embedding",
                    )
                )
            self.db.upsert_output(
                file_id=file_id,
                full_text=document.full_text,
                summary=document.summary,
                key_points=document.key_points,
                topic_tags=document.topic_tags,
                accessibility_notes=document.accessibility_notes,
            )
            self._write_processed_json(document)
            self.db.update_file_status(file_id, "completed")
            return document
        except Exception as exc:
            self.db.update_file_status(file_id, "failed", str(exc))
            raise

    def _extract(
        self,
        file_type: str,
        file_id: str,
        file_name: str,
        path: Path,
        processing_mode: str,
    ) -> UnifiedDocument:
        if file_type == "pdf":
            return PDFProcessor(self.db, self.vision_client, self.config).process(file_id, file_name, path, processing_mode)
        if file_type == "image":
            return ImageProcessor(self.db, self.vision_client, self.config).process(file_id, file_name, path, processing_mode)
        if file_type == "audio":
            return AudioProcessor(self.db, self.config).process(file_id, file_name, path, processing_mode)
        raise ValueError(f"Unsupported file type: {file_type}")

    def _summarize(self, document: UnifiedDocument) -> None:
        summary_data, usage = self.llm_client.summarize_json(document.full_text, document.file_id)
        document.summary = str(summary_data.get("summary") or "")
        document.key_points = [str(item) for item in summary_data.get("key_points", [])][:7]
        document.topic_tags = [str(item) for item in summary_data.get("topic_tags", [])][:10]
        document.accessibility_notes = str(summary_data.get("accessibility_notes") or "")
        provider = self.llm_client.provider if self.llm_client.available and usage.get("input_tokens", 0) else "local"
        model = (
            self.config.openrouter_model
            if provider == "openrouter"
            else self.config.llm_model
            if provider == "openai"
            else "local-summary"
        )
        self.db.log_usage(
            UsageRecord(
                file_id=document.file_id,
                operation="summary_key_points_tags",
                provider=provider,
                model=model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
        )

    def _write_processed_json(self, document: UnifiedDocument) -> None:
        payload = {
            "file_id": document.file_id,
            "file_name": document.file_name,
            "file_type": document.file_type,
            "processing_mode": document.processing_mode,
            "full_text": document.full_text,
            "pages": [page.__dict__ for page in document.pages],
            "audio_segments": [segment.__dict__ for segment in document.audio_segments],
            "summary": document.summary,
            "key_points": document.key_points,
            "topic_tags": document.topic_tags,
            "accessibility_notes": document.accessibility_notes,
            "warnings": document.warnings,
        }
        output = self.config.processed_dir / f"{document.file_id}.json"
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
