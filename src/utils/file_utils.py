from __future__ import annotations

import re
import uuid
from pathlib import Path


PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AUDIO_EXTENSIONS = {".mp3", ".wav"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS


def safe_filename(name: str) -> str:
    stem = Path(name).stem or "upload"
    suffix = Path(name).suffix.lower()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return f"{cleaned[:80] or 'upload'}{suffix}"


def file_type_for_name(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def validate_upload(name: str, size_bytes: int, max_size_mb: int) -> tuple[bool, str]:
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return False, "Supported formats are PDF, JPG, PNG, MP3, and WAV."
    if size_bytes > max_size_mb * 1024 * 1024:
        return False, f"File exceeds the configured {max_size_mb} MB limit."
    return True, "File validated."


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

