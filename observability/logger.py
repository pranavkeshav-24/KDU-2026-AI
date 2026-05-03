from __future__ import annotations

import json
import logging
import time
from typing import Any


logging.basicConfig(level=logging.INFO, format="%(message)s")


class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log(self, event: str, **kwargs: Any) -> None:
        entry = {
            "timestamp": time.time(),
            "logger": self.logger.name,
            "event": event,
            **kwargs,
        }
        self.logger.info(json.dumps(entry, sort_keys=True))

