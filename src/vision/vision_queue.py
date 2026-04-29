from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(order=True)
class VisionQueueItem:
    priority: int
    file_id: str
    page_number: int | None
    region_id: str | None
    region_type: str
    image_path: Path
    reason_for_vision: str
    estimated_cost: float = 0.0


class VisionQueue:
    def __init__(self):
        self._items: list[VisionQueueItem] = []

    def add(self, item: VisionQueueItem) -> None:
        self._items.append(item)
        self._items.sort()

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def as_list(self) -> list[VisionQueueItem]:
        return list(self._items)

