from __future__ import annotations

from src.config import AppConfig, config
from src.layout.page_inventory import ImageRegion, PageInventory


def meaningful_regions(inventory: PageInventory, app_config: AppConfig = config) -> list[ImageRegion]:
    regions: list[ImageRegion] = []
    for region in inventory.image_regions:
        if region.is_header_footer or region.is_repeated:
            continue
        if region.area_ratio >= app_config.large_image_area_threshold:
            regions.append(region)
    return regions

