from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .pipeline import AudioCoordinator


@dataclass(frozen=True)
class PlaybackResult:
    spoken_text: str
    unspoken_text: str
    completed: bool


class StreamingTTSRenderer:
    """Streams text chunks while holding the exclusive playback lock."""

    def __init__(self, audio: AudioCoordinator, chunk_delay_seconds: float = 0.0) -> None:
        self.audio = audio
        self.chunk_delay_seconds = chunk_delay_seconds
        self._stop = asyncio.Event()
        self._owner = "tts-renderer"

    def stop(self) -> None:
        self._stop.set()

    async def speak(self, text: str) -> PlaybackResult:
        self._stop.clear()
        await self.audio.playback_lock.acquire(self._owner)
        spoken: list[str] = []
        unspoken: list[str] = []
        words = text.split()
        try:
            for index, word in enumerate(words):
                if self._stop.is_set() or self.audio.interruption_event.is_set():
                    unspoken = words[index:]
                    return PlaybackResult(" ".join(spoken), " ".join(unspoken), completed=False)
                spoken.append(word)
                if self.chunk_delay_seconds:
                    await asyncio.sleep(self.chunk_delay_seconds)
            return PlaybackResult(" ".join(spoken), "", completed=True)
        finally:
            self.audio.playback_lock.release(self._owner)
