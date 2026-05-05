from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenReservation:
    queued: bool
    window_tokens: int


class AdaptiveTokenThrottle:
    """Client-side rolling-window token throttle."""

    def __init__(self, tokens_per_minute: int, throttle_ratio: float = 0.8) -> None:
        self.tokens_per_minute = tokens_per_minute
        self.limit = int(tokens_per_minute * throttle_ratio)
        self._events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    async def reserve(self, tokens: int) -> TokenReservation:
        queued = False
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0][0] > 60:
                    self._events.popleft()
                current = sum(count for _, count in self._events)
                if current + tokens <= self.limit:
                    self._events.append((now, tokens))
                    return TokenReservation(queued=queued, window_tokens=current + tokens)
            queued = True
            await asyncio.sleep(0.05)

