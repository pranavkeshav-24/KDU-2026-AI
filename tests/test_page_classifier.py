from src.layout.page_classifier import classify_page
from src.layout.page_inventory import PageInventory


def inv(**kwargs):
    defaults = {
        "page_number": 1,
        "text": "",
        "text_char_count": 0,
        "text_block_count": 0,
        "text_area_ratio": 0.0,
        "image_count": 0,
        "image_area_ratio": 0.0,
        "large_image_count": 0,
        "table_like_text_score": 0.0,
        "ocr_quality_signal": 0.0,
        "image_regions": [],
    }
    defaults.update(kwargs)
    return PageInventory(**defaults)


def test_text_heavy_page_uses_native_text():
    result = classify_page(inv(text_char_count=1200, text_area_ratio=0.6), "balanced")
    assert result.page_type == "text_only"
    assert not result.vision_required


def test_scanned_page_requires_full_page_vision():
    result = classify_page(inv(text_char_count=10, image_area_ratio=0.9, image_count=1), "balanced")
    assert result.page_type == "scanned"
    assert result.vision_required


def test_fast_mode_skips_mixed_crop_vision():
    result = classify_page(inv(text_char_count=1200, text_area_ratio=0.3, image_area_ratio=0.3, large_image_count=1), "fast")
    assert result.page_type == "mixed"
    assert not result.vision_required

