from __future__ import annotations

from pathlib import Path

from src.config import AppConfig, config
from src.cost.cost_tracker import UsageRecord
from src.storage.db import Database
from src.storage.models import PageContent, UnifiedDocument
from src.utils.text_utils import clean_text
from src.vision.vision_client import VisionClient


class ImageProcessor:
    def __init__(self, db: Database, vision_client: VisionClient, app_config: AppConfig = config):
        self.db = db
        self.vision_client = vision_client
        self.config = app_config

    def process(self, file_id: str, file_name: str, path: Path, processing_mode: str) -> UnifiedDocument:
        vision_data, usage, provider = self.vision_client.analyze_image(path)
        extracted_text = clean_text(vision_data.get("extracted_text") or "")
        alt_text = clean_text(vision_data.get("alt_text") or "")
        detailed = clean_text(vision_data.get("detailed_description") or "")
        warnings = vision_data.get("warnings") or []
        page_text = clean_text("\n\n".join(part for part in [extracted_text, alt_text, detailed] if part))
        model = (
            self.config.openrouter_model
            if provider == "openrouter"
            else self.config.vision_model
            if provider == "openai"
            else "local-vision-placeholder"
        )
        self.db.log_usage(
            UsageRecord(
                file_id=file_id,
                operation="image_vision",
                provider=provider,
                model=model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
        )
        page = PageContent(
            page_number=1,
            text=page_text,
            visual_description=detailed or alt_text,
            extraction_method="image_vision" if provider in {"openai", "openrouter"} else "image_metadata_placeholder",
            vision_enriched=provider in {"openai", "openrouter"},
            warnings=warnings,
            page_type="image",
            text_char_count=len(extracted_text),
            image_count=1,
            image_area_ratio=1.0,
            vision_required=True,
        )
        return UnifiedDocument(
            file_id=file_id,
            file_name=file_name,
            file_type="image",
            processing_mode=processing_mode,
            full_text=page_text,
            pages=[page],
            warnings=warnings,
        )
