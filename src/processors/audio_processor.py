from __future__ import annotations

from pathlib import Path

from src.config import AppConfig, config
from src.cost.cost_tracker import UsageRecord
from src.storage.db import Database
from src.storage.models import AudioSegment, UnifiedDocument
from src.utils.text_utils import clean_text


class AudioProcessor:
    def __init__(self, db: Database, app_config: AppConfig = config):
        self.db = db
        self.config = app_config

    def process(self, file_id: str, file_name: str, path: Path, processing_mode: str) -> UnifiedDocument:
        warnings: list[str] = []
        transcript = ""
        segments: list[AudioSegment] = []
        try:
            from transformers import pipeline

            transcriber = pipeline(
                "automatic-speech-recognition",
                model=self.config.whisper_model,
                return_timestamps=True,
            )
            result = transcriber(str(path))
            transcript = clean_text(result.get("text", ""))
            for chunk in result.get("chunks", []) or []:
                timestamp = chunk.get("timestamp") or (None, None)
                segments.append(
                    AudioSegment(
                        text=clean_text(chunk.get("text", "")),
                        timestamp_start=timestamp[0],
                        timestamp_end=timestamp[1],
                    )
                )
        except Exception as exc:
            warnings.append(f"Local Whisper transcription unavailable: {exc}")
            transcript = (
                "Audio file was uploaded, but local Whisper transcription could not run. "
                "Install transformers, torch, and an audio backend to enable offline transcription."
            )
        self.db.log_usage(
            UsageRecord(
                file_id=file_id,
                operation="audio_transcription",
                provider="local",
                model=self.config.whisper_model,
            )
        )
        return UnifiedDocument(
            file_id=file_id,
            file_name=file_name,
            file_type="audio",
            processing_mode=processing_mode,
            full_text=transcript,
            audio_segments=segments,
            warnings=warnings,
        )

