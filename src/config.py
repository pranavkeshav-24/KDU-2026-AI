from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def _read_vscode_env_file_from_settings() -> Path | None:
    settings_path = ROOT_DIR / ".vscode" / "settings.json"
    if not settings_path.exists():
        return None
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None

    env_file = settings.get("python.envFile")
    if not isinstance(env_file, str) or not env_file.strip():
        return None

    resolved = env_file.strip().replace("${workspaceFolder}", str(ROOT_DIR))
    candidate = Path(resolved)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate


def _iter_env_files() -> list[Path]:
    candidates: list[Path] = [ROOT_DIR / ".env", ROOT_DIR / ".env.local"]

    settings_env = _read_vscode_env_file_from_settings()
    if settings_env is not None:
        candidates.append(settings_env)

    explicit_env_file = os.getenv("PYTHON_ENV_FILE") or os.getenv("ENV_FILE")
    if explicit_env_file:
        candidates.append(Path(explicit_env_file))

    virtual_env = os.getenv("VIRTUAL_ENV")
    if virtual_env:
        candidates.append(Path(virtual_env) / ".env")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _parse_env_line(line: str) -> tuple[str, str] | None:
    if not line or line.startswith("#") or "=" not in line:
        return None

    if line.startswith("export "):
        line = line[len("export ") :].strip()
        if "=" not in line:
            return None

    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if value and value[0] in {'"', "'"} and value[-1:] == value[0]:
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()

    return key, value


def _load_env_file() -> None:
    for env_path in _iter_env_files():
        if not env_path.exists() or not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            parsed = _parse_env_line(raw_line.strip())
            if parsed is None:
                continue
            key, value = parsed
            if not os.environ.get(key):
                os.environ[key] = value


_load_env_file()


@dataclass(frozen=True)
class AppConfig:
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vision_model: str = os.getenv("VISION_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    whisper_model: str = os.getenv("WHISPER_MODEL", "openai/whisper-small")
    processing_mode: str = os.getenv("PROCESSING_MODE", "balanced")
    vector_store: str = os.getenv("VECTOR_STORE", "sqlite")
    keyword_search: str = os.getenv("KEYWORD_SEARCH", "sqlite_fts5")
    summary_word_limit: int = int(os.getenv("SUMMARY_WORD_LIMIT", "150"))
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "700"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
    top_k_search_results: int = int(os.getenv("TOP_K_SEARCH_RESULTS", "5"))
    vision_detail: str = os.getenv("VISION_DETAIL", "low")
    native_text_min_chars: int = int(os.getenv("NATIVE_TEXT_MIN_CHARS", "100"))
    text_area_threshold: float = float(os.getenv("TEXT_AREA_THRESHOLD", "0.45"))
    large_image_area_threshold: float = float(os.getenv("LARGE_IMAGE_AREA_THRESHOLD", "0.15"))
    scanned_page_image_area_threshold: float = float(os.getenv("SCANNED_PAGE_IMAGE_AREA_THRESHOLD", "0.60"))
    header_footer_margin_ratio: float = float(os.getenv("HEADER_FOOTER_MARGIN_RATIO", "0.10"))
    hybrid_search_rrf_k: int = int(os.getenv("HYBRID_SEARCH_RRF_K", "60"))
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
    keyword_weight: float = float(os.getenv("KEYWORD_WEIGHT", "0.4"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    app_name: str = "Content Accessibility Suite"

    @property
    def uploads_dir(self) -> Path:
        return DATA_DIR / "uploads"

    @property
    def processed_dir(self) -> Path:
        return DATA_DIR / "processed"

    @property
    def page_renders_dir(self) -> Path:
        return DATA_DIR / "page_renders"

    @property
    def visual_crops_dir(self) -> Path:
        return DATA_DIR / "visual_crops"

    @property
    def db_path(self) -> Path:
        return DATA_DIR / "app.db"


config = AppConfig()


def ensure_data_dirs(app_config: AppConfig = config) -> None:
    for path in (
        DATA_DIR,
        app_config.uploads_dir,
        app_config.processed_dir,
        app_config.page_renders_dir,
        app_config.visual_crops_dir,
        DATA_DIR / "chroma",
        DATA_DIR / "keyword_index",
    ):
        path.mkdir(parents=True, exist_ok=True)
