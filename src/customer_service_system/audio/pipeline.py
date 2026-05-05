from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from ..core import EventType


@dataclass(frozen=True)
class AudioFrame:
    data: bytes
    monotonic_ts: float
    is_speech: bool


@dataclass(frozen=True)
class AudioEvent:
    event_type: EventType
    frames: list[AudioFrame]
    monotonic_ts: float


class PlaybackLock:
    """Single owner gate for speaker output and interruption detection."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: str | None = None

    @property
    def held(self) -> bool:
        return self._lock.locked()

    @property
    def owner(self) -> str | None:
        return self._owner

    async def acquire(self, owner: str) -> None:
        await self._lock.acquire()
        self._owner = owner

    def release(self, owner: str) -> None:
        if not self._lock.locked():
            return
        if self._owner != owner:
            raise RuntimeError(f"playback lock is owned by {self._owner}, not {owner}")
        self._owner = None
        self._lock.release()


class AudioCoordinator:
    def __init__(self) -> None:
        self.playback_lock = PlaybackLock()
        self.interruption_event = asyncio.Event()

    def signal_interruption(self) -> None:
        self.interruption_event.set()

    def clear_interruption(self) -> None:
        self.interruption_event.clear()


class SimulatedAudioCapture:
    """Deterministic audio source used for tests and the local demo."""

    def __init__(self, utterances: list[str], frame_delay_seconds: float = 0.0) -> None:
        self.utterances = utterances
        self.frame_delay_seconds = frame_delay_seconds

    async def frames(self) -> AsyncIterator[AudioFrame]:
        for utterance in self.utterances:
            for word in utterance.split():
                if self.frame_delay_seconds:
                    await asyncio.sleep(self.frame_delay_seconds)
                yield AudioFrame(
                    data=word.encode("utf-8"),
                    monotonic_ts=time.monotonic(),
                    is_speech=True,
                )
            yield AudioFrame(data=b"", monotonic_ts=time.monotonic(), is_speech=False)


@dataclass
class VoiceActivityDetector:
    trailing_silence_ms: int = 600
    _buffer: list[AudioFrame] = field(default_factory=list)
    _speech_started: bool = False

    async def consume(
        self,
        frames: AsyncIterator[AudioFrame],
        audio: AudioCoordinator,
    ) -> AsyncIterator[AudioEvent]:
        async for frame in frames:
            if frame.is_speech and not self._speech_started:
                self._speech_started = True
                self._buffer = []
                if audio.playback_lock.held:
                    audio.signal_interruption()
                    yield AudioEvent(
                        event_type=EventType.INTERRUPTION_DETECTED,
                        frames=[frame],
                        monotonic_ts=frame.monotonic_ts,
                    )
            if frame.is_speech:
                self._buffer.append(frame)
                continue

            if self._speech_started and self._buffer:
                event = AudioEvent(
                    event_type=EventType.UTTERANCE_READY,
                    frames=list(self._buffer),
                    monotonic_ts=frame.monotonic_ts,
                )
                self._buffer = []
                self._speech_started = False
                if audio.playback_lock.held and not audio.interruption_event.is_set():
                    continue
                yield event
