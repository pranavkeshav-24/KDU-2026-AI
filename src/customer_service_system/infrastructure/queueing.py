from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator


class QueueFullError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueueMetrics:
    queue_depth: int
    in_use: int
    capacity: int
    wait_time_ms: int


class ConcurrencyQueue:
    """Semaphore guard with explicit queue-depth rejection."""

    def __init__(self, max_concurrent: int, max_depth: int) -> None:
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._waiting = 0
        self._in_use = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[QueueMetrics]:
        async with self._lock:
            if self._waiting >= self.max_depth:
                raise QueueFullError("database concurrency queue is full")
            self._waiting += 1
        started = time.perf_counter()
        await self._semaphore.acquire()
        wait_ms = int((time.perf_counter() - started) * 1000)
        async with self._lock:
            self._waiting -= 1
            self._in_use += 1
            metrics = QueueMetrics(
                queue_depth=self._waiting,
                in_use=self._in_use,
                capacity=self.max_concurrent,
                wait_time_ms=wait_ms,
            )
        try:
            yield metrics
        finally:
            async with self._lock:
                self._in_use -= 1
            self._semaphore.release()

