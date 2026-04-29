from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from src.config import AppConfig
from src.storage.db import Database
from src.utils.file_utils import file_type_for_name, new_id, safe_filename, validate_upload


class UploadService:
    def __init__(self, config: AppConfig, db: Database):
        self.config = config
        self.db = db

    def save_upload(
        self,
        uploaded_file: BinaryIO,
        original_name: str,
        size_bytes: int,
        processing_mode: str,
    ) -> tuple[str, Path]:
        valid, message = validate_upload(original_name, size_bytes, self.config.max_file_size_mb)
        if not valid:
            raise ValueError(message)

        file_id = new_id("file")
        file_type = file_type_for_name(original_name)
        filename = safe_filename(original_name)
        destination = self.config.uploads_dir / f"{file_id}_{filename}"

        data = uploaded_file.read()
        destination.write_bytes(data)
        self.db.create_file(
            file_id=file_id,
            file_name=original_name,
            file_type=file_type,
            file_size_bytes=len(data),
            upload_path=str(destination),
            processing_mode=processing_mode,
        )
        return file_id, destination

