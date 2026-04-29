from __future__ import annotations

from dataclasses import dataclass

from src.config import AppConfig, config
from src.layout.page_inventory import PageInventory


@dataclass
class PageClassification:
    page_type: str
    extraction_method: str
    vision_required: bool
    confidence: float
    reason: str


def classify_page(
    inventory: PageInventory,
    processing_mode: str = "balanced",
    app_config: AppConfig = config,
) -> PageClassification:
    has_sufficient_text = inventory.text_char_count >= app_config.native_text_min_chars
    text_heavy = inventory.text_area_ratio >= app_config.text_area_threshold
    mostly_image = inventory.image_area_ratio >= app_config.scanned_page_image_area_threshold
    has_large_visual = inventory.large_image_count > 0
    poor_ocr = inventory.ocr_quality_signal > 0.25
    tableish = inventory.table_like_text_score > 0.35

    if not has_sufficient_text and mostly_image:
        return PageClassification("scanned", "full_page_vision", True, 0.92, "Sparse text and page is mostly image.")
    if text_heavy and not has_large_visual:
        return PageClassification("text_only", "native_text", False, 0.9, "Text coverage is sufficient and no large visual region was detected.")
    if has_sufficient_text and has_large_visual:
        if processing_mode == "fast":
            return PageClassification("mixed", "native_text", False, 0.72, "Fast mode skips crop-level visual enrichment.")
        return PageClassification("mixed", "native_text_plus_crop_vision", True, 0.82, "Native text plus meaningful visual region.")
    if has_large_visual and (tableish or poor_ocr or processing_mode == "deep"):
        return PageClassification("visual_region", "crop_vision", True, 0.76, "Large visual or table-like region needs enrichment.")
    if not has_sufficient_text:
        return PageClassification("low_text", "native_text", processing_mode != "fast", 0.62, "Native text is sparse.")
    return PageClassification("text_only", "native_text", False, 0.7, "Defaulted to native extraction.")

