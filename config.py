"""Settings management for the Multimodal AI Assistant."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM Gateway
    OPENROUTER_API_KEY: str
    OPENROUTER_SITE_URL: str
    OPENROUTER_APP_NAME: str
    
    # Tools
    OPENWEATHERMAP_API_KEY: str
    
    # Infrastructure
    REDIS_URL: str
    DATABASE_URL: str
    
    # Application Behavior
    DEFAULT_STYLE: Literal["expert", "child", "casual", "formal"]
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"]

    # API model mapping.
    VISION_MODEL: str
    REASONING_MODEL: str
    FAST_MODEL: str
    STRUCTURED_MODEL: str
    FALLBACK_MODEL: str
    MODEL_FALLBACKS: str = ""
    OPENROUTER_RETRY_ATTEMPTS: int = 2
    OPENROUTER_RETRY_BASE_DELAY_SECONDS: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
