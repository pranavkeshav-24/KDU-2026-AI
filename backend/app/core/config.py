from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    session_secret: str = os.getenv(
        "SESSION_SECRET",
        "dev-only-change-me-to-a-32-byte-random-secret",
    )
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "900"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    default_user_id: str = os.getenv("DEFAULT_USER_ID", "demo-user")
    cors_origins: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        object.__setattr__(self, "cors_origins", _split_csv(origins))


settings = Settings()
