from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageContent:
    page_number: int
    text: str
    visual_description: str = ""
    extraction_method: str = "unknown"
    vision_enriched: bool = False
    warnings: list[str] = field(default_factory=list)
    page_type: str = "unknown"
    text_char_count: int = 0
    text_area_ratio: float = 0.0
    image_count: int = 0
    image_area_ratio: float = 0.0
    vision_required: bool = False


@dataclass
class AudioSegment:
    text: str
    timestamp_start: float | None = None
    timestamp_end: float | None = None


@dataclass
class UnifiedDocument:
    file_id: str
    file_name: str
    file_type: str
    processing_mode: str
    full_text: str
    pages: list[PageContent] = field(default_factory=list)
    audio_segments: list[AudioSegment] = field(default_factory=list)
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    topic_tags: list[str] = field(default_factory=list)
    accessibility_notes: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    file_id: str
    chunk_index: int
    chunk_text: str
    page_start: int | None = None
    page_end: int | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    source_type: str = "unknown"
    vision_enriched: bool = False
    vector_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk_id: str
    file_id: str
    file_name: str
    chunk_text: str
    score: float
    match_type: str
    page_start: int | None = None
    page_end: int | None = None
    source_type: str = "unknown"
    vision_enriched: bool = False

