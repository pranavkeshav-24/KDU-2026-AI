from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def _load_env_file() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] in {"'", '"'} and value[-1:] == value[0]:
            value = value[1:-1]
        if key and not os.environ.get(key):
            os.environ[key] = value


_load_env_file()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

GENERAL_AGENT_MODEL = os.getenv("GENERAL_AGENT_MODEL", "gpt-4o-mini")
REASONING_MODEL = os.getenv("REASONING_MODEL", "o3-mini")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")

MAX_TOKENS_PER_CALL = int(os.getenv("MAX_TOKENS_PER_CALL", "1024"))
SUB_AGENT_MAX_TOKENS = int(os.getenv("SUB_AGENT_MAX_TOKENS", "512"))
PLANNER_MAX_TOKENS = int(os.getenv("PLANNER_MAX_TOKENS", "1024"))
CIRCUIT_BREAKER_COOLDOWN_SECONDS = int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "60"))


def provider_name() -> str:
    if OPENAI_API_KEY:
        return "openai"
    if OPENROUTER_API_KEY:
        return "openrouter"
    return "local"


def get_openai_client():
    if OPENAI_API_KEY:
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=OPENAI_API_KEY)
    if OPENROUTER_API_KEY:
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    raise EnvironmentError("No LLM provider API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")


def selected_model(default_model: str) -> str:
    return OPENROUTER_MODEL if provider_name() == "openrouter" else default_model

