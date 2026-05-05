from __future__ import annotations

from ..core import Transcript
from .pipeline import AudioFrame


class LocalWhisperTranscriptionService:
    """Drop-in stand-in for Whisper: byte frames become transcript text."""

    async def transcribe(self, frames: list[AudioFrame]) -> Transcript:
        text = " ".join(frame.data.decode("utf-8", errors="ignore") for frame in frames if frame.data)
        return Transcript(text=text.strip(), confidence=0.99 if text.strip() else 0.0)
