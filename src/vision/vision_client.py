from __future__ import annotations

from pathlib import Path

from src.llm.openai_client import OpenAIClient
from src.llm.prompts import IMAGE_ACCESSIBILITY_PROMPT, PDF_VISION_PROMPT


class VisionClient:
    def __init__(self, llm_client: OpenAIClient):
        self.llm_client = llm_client

    def analyze_image(self, image_path: Path):
        return self.llm_client.vision_image_json(image_path, IMAGE_ACCESSIBILITY_PROMPT)

    def analyze_pdf_region(self, image_path: Path):
        return self.llm_client.vision_image_json(image_path, PDF_VISION_PROMPT)

