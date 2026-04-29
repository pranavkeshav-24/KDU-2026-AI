from __future__ import annotations

from src.config import AppConfig, config
from src.storage.models import Chunk, UnifiedDocument
from src.utils.file_utils import new_id
from src.utils.text_utils import estimate_tokens


class Chunker:
    def __init__(self, app_config: AppConfig = config):
        self.config = app_config

    def chunk_document(self, document: UnifiedDocument) -> list[Chunk]:
        if document.pages:
            return self._chunk_pages(document)
        return self._chunk_text(
            document.full_text,
            document.file_id,
            source_type=document.file_type,
            start_page=None,
            end_page=None,
            start_index=0,
        )

    def _chunk_pages(self, document: UnifiedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_pages: list[int] = []
        current_vision = False
        chunk_index = 0

        for page in document.pages:
            page_text = page.text
            if page.visual_description:
                page_text = f"{page_text}\n\nVisual context: {page.visual_description}".strip()
            if not page_text:
                continue
            candidate = "\n\n".join([*current_parts, f"[Page {page.page_number}]\n{page_text}"])
            if estimate_tokens(candidate) > self.config.chunk_size_tokens and current_parts:
                chunks.append(
                    Chunk(
                        chunk_id=new_id("chunk"),
                        file_id=document.file_id,
                        chunk_index=chunk_index,
                        chunk_text="\n\n".join(current_parts),
                        page_start=min(current_pages),
                        page_end=max(current_pages),
                        source_type="pdf_page" if document.file_type == "pdf" else document.file_type,
                        vision_enriched=current_vision,
                    )
                )
                chunk_index += 1
                overlap = self._overlap_text(current_parts[-1])
                current_parts = [overlap] if overlap else []
                current_pages = [current_pages[-1]] if current_pages else []
                current_vision = current_vision and bool(current_parts)

            current_parts.append(f"[Page {page.page_number}]\n{page_text}")
            current_pages.append(page.page_number)
            current_vision = current_vision or page.vision_enriched

        if current_parts:
            chunks.append(
                Chunk(
                    chunk_id=new_id("chunk"),
                    file_id=document.file_id,
                    chunk_index=chunk_index,
                    chunk_text="\n\n".join(current_parts),
                    page_start=min(current_pages),
                    page_end=max(current_pages),
                    source_type="pdf_page" if document.file_type == "pdf" else document.file_type,
                    vision_enriched=current_vision,
                )
            )
        return chunks

    def _chunk_text(
        self,
        text: str,
        file_id: str,
        source_type: str,
        start_page: int | None,
        end_page: int | None,
        start_index: int,
    ) -> list[Chunk]:
        words = text.split()
        if not words:
            return []
        approx_words = max(80, int(self.config.chunk_size_tokens / 1.33))
        overlap_words = max(0, int(self.config.chunk_overlap_tokens / 1.33))
        chunks: list[Chunk] = []
        index = start_index
        cursor = 0
        while cursor < len(words):
            window = words[cursor : cursor + approx_words]
            chunks.append(
                Chunk(
                    chunk_id=new_id("chunk"),
                    file_id=file_id,
                    chunk_index=index,
                    chunk_text=" ".join(window),
                    page_start=start_page,
                    page_end=end_page,
                    source_type=source_type,
                )
            )
            index += 1
            if cursor + approx_words >= len(words):
                break
            cursor += max(1, approx_words - overlap_words)
        return chunks

    def _overlap_text(self, text: str) -> str:
        words = text.split()
        overlap_words = max(0, int(self.config.chunk_overlap_tokens / 1.33))
        if not words or overlap_words == 0:
            return ""
        return " ".join(words[-overlap_words:])

