""""Settings management for the Multimodal AI Assistant."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM Gateway
    OPENROUTER_API_KEY: str = "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"
    OPENROUTER_SITE_URL: str = "http://127.0.0.1:8001"
    OPENROUTER_APP_NAME: str = "KDU-2026-AI"
    
    # Tools
    OPENWEATHERMAP_API_KEY: str = "0d9bc17bc106e9827acb14050f26f64c"
    
    # Infrastructure
    REDIS_URL: str = "redis://localhost:6379"
    DATABASE_URL: str = "sqlite:///./multimodal_assistant.db"
    
    # Application Behavior
    DEFAULT_STYLE: Literal["expert", "child", "casual", "formal"] = "expert"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # API model mapping.
    # These defaults are pinned to live free OpenRouter model IDs verified on April 12, 2026.
    VISION_MODEL: str = "google/gemma-4-26b-a4b-it:free"
    REASONING_MODEL: str = "google/gemma-4-26b-a4b-it:free"
    FAST_MODEL: str = "google/gemma-4-26b-a4b-it:free"
    STRUCTURED_MODEL: str = "google/gemma-4-26b-a4b-it:free"
    FALLBACK_MODEL: str = "google/gemma-4-31b-it:free"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
