from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageRegion:
    bbox: tuple[float, float, float, float]
    area_ratio: float
    is_header_footer: bool = False
    is_repeated: bool = False
    kind: str = "image"


@dataclass
class PageInventory:
    page_number: int
    text: str
    text_char_count: int
    text_block_count: int
    text_area_ratio: float
    image_count: int
    image_area_ratio: float
    large_image_count: int
    table_like_text_score: float
    ocr_quality_signal: float
    image_regions: list[ImageRegion] = field(default_factory=list)


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = rect
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def inventory_pdf(path: Path, large_image_threshold: float = 0.15, header_footer_margin_ratio: float = 0.10) -> list[PageInventory]:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PyMuPDF is required for PDF inventory. Install pymupdf.") from exc

    inventories: list[PageInventory] = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            page_rect = page.rect
            page_area = max(1.0, float(page_rect.width * page_rect.height))
            text_dict: dict[str, Any] = page.get_text("dict")
            text_blocks = [block for block in text_dict.get("blocks", []) if block.get("type") == 0]
            image_blocks = [block for block in text_dict.get("blocks", []) if block.get("type") == 1]

            text_area = sum(_rect_area(tuple(block.get("bbox", (0, 0, 0, 0)))) for block in text_blocks)
            image_area = sum(_rect_area(tuple(block.get("bbox", (0, 0, 0, 0)))) for block in image_blocks)
            image_regions: list[ImageRegion] = []
            for block in image_blocks:
                bbox = tuple(float(value) for value in block.get("bbox", (0, 0, 0, 0)))
                ratio = _rect_area(bbox) / page_area
                y0, y1 = bbox[1], bbox[3]
                margin = page_rect.height * header_footer_margin_ratio
                is_header_footer = y1 <= margin or y0 >= page_rect.height - margin
                image_regions.append(ImageRegion(bbox=bbox, area_ratio=ratio, is_header_footer=is_header_footer))

            text = page.get_text("text") or ""
            words = text.split()
            table_like = _table_like_score(text)
            ocr_quality = _ocr_quality_signal(text, words)
            inventories.append(
                PageInventory(
                    page_number=index,
                    text=text,
                    text_char_count=len(text.strip()),
                    text_block_count=len(text_blocks),
                    text_area_ratio=min(1.0, text_area / page_area),
                    image_count=len(image_blocks),
                    image_area_ratio=min(1.0, image_area / page_area),
                    large_image_count=sum(1 for region in image_regions if region.area_ratio >= large_image_threshold),
                    table_like_text_score=table_like,
                    ocr_quality_signal=ocr_quality,
                    image_regions=image_regions,
                )
            )
    return inventories


def _table_like_score(text: str) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    tabular = 0
    for line in lines:
        if "\t" in line or line.count("  ") >= 2 or sum(ch.isdigit() for ch in line) >= 4:
            tabular += 1
    return tabular / max(1, len(lines))


def _ocr_quality_signal(text: str, words: list[str]) -> float:
    if not text.strip():
        return 0.0
    odd_words = sum(1 for word in words if len(word) > 18 or any(ch in word for ch in "�□"))
    short_lines = sum(1 for line in text.splitlines() if 0 < len(line.strip()) < 3)
    return min(1.0, (odd_words + short_lines) / max(1, len(words)))

