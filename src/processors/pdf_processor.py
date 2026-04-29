from __future__ import annotations

from pathlib import Path

from src.config import AppConfig, config
from src.cost.cost_tracker import UsageRecord
from src.layout.page_classifier import classify_page
from src.layout.page_inventory import PageInventory, inventory_pdf
from src.layout.visual_region_detector import meaningful_regions
from src.storage.db import Database
from src.storage.models import PageContent, UnifiedDocument
from src.utils.text_utils import clean_text
from src.vision.vision_client import VisionClient


class PDFProcessor:
    def __init__(self, db: Database, vision_client: VisionClient, app_config: AppConfig = config):
        self.db = db
        self.vision_client = vision_client
        self.config = app_config

    def process(self, file_id: str, file_name: str, path: Path, processing_mode: str) -> UnifiedDocument:
        try:
            inventories = inventory_pdf(
                path,
                large_image_threshold=self.config.large_image_area_threshold,
                header_footer_margin_ratio=self.config.header_footer_margin_ratio,
            )
        except Exception as exc:
            return UnifiedDocument(
                file_id=file_id,
                file_name=file_name,
                file_type="pdf",
                processing_mode=processing_mode,
                full_text="",
                warnings=[str(exc)],
            )

        pages: list[PageContent] = []
        for inventory in inventories:
            classification = classify_page(inventory, processing_mode, self.config)
            text = clean_text(inventory.text)
            visual_description = ""
            warnings: list[str] = []
            vision_enriched = False

            if classification.vision_required:
                try:
                    image_path = self._render_page(path, file_id, inventory.page_number)
                    vision_data, usage, provider = self.vision_client.analyze_pdf_region(image_path)
                    extracted = vision_data.get("extracted_text") or ""
                    detailed = vision_data.get("detailed_description") or vision_data.get("alt_text") or ""
                    if extracted and len(extracted) > len(text):
                        text = clean_text(extracted)
                    visual_description = clean_text(detailed)
                    warnings.extend(vision_data.get("warnings") or [])
                    vision_enriched = provider != "local" and bool(visual_description or extracted)
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
                            operation="pdf_vision",
                            provider=provider,
                            model=model,
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                        )
                    )
                except Exception as exc:
                    warnings.append(f"Vision enrichment failed: {exc}")

            if classification.extraction_method == "native_text_plus_crop_vision":
                region_count = len(meaningful_regions(inventory, self.config))
                if region_count and not visual_description:
                    warnings.append(f"{region_count} visual region(s) detected; configure vision for detailed descriptions.")

            pages.append(self._page_content(inventory, classification, text, visual_description, warnings, vision_enriched))

        full_text = clean_text("\n\n".join(page.text for page in pages if page.text))
        self.db.log_usage(
            UsageRecord(
                file_id=file_id,
                operation="native_pdf_extraction",
                provider="local",
                model="pymupdf",
            )
        )
        return UnifiedDocument(
            file_id=file_id,
            file_name=file_name,
            file_type="pdf",
            processing_mode=processing_mode,
            full_text=full_text,
            pages=pages,
        )

    def _page_content(
        self,
        inventory: PageInventory,
        classification,
        text: str,
        visual_description: str,
        warnings: list[str],
        vision_enriched: bool,
    ) -> PageContent:
        if not text and visual_description:
            text = visual_description
        return PageContent(
            page_number=inventory.page_number,
            text=text,
            visual_description=visual_description,
            extraction_method=classification.extraction_method,
            vision_enriched=vision_enriched,
            warnings=warnings,
            page_type=classification.page_type,
            text_char_count=inventory.text_char_count,
            text_area_ratio=inventory.text_area_ratio,
            image_count=inventory.image_count,
            image_area_ratio=inventory.image_area_ratio,
            vision_required=classification.vision_required,
        )

    def _render_page(self, path: Path, file_id: str, page_number: int) -> Path:
        import fitz

        output = self.config.page_renders_dir / f"{file_id}_page_{page_number}.png"
        with fitz.open(path) as doc:
            page = doc[page_number - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            pix.save(output)
        return output
