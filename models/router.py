"""Task classification and OpenRouter LLM mapping."""

import httpx

from langchain_openai import ChatOpenAI

from config import settings
from models.schemas import ChatRequest


def classify_task(payload: ChatRequest, has_image: bool = False) -> str:
    """
    Lightweight heuristic pass to classify user intent without an LLM hop.
    """
    if has_image:
        return "vision"
    
    msg_lower = payload.message.lower()
    
    # Simple structured requests
    if any(k in msg_lower for k in ["weather", "temperature", "forecast", "humidity"]):
        return "structured"
    
    # Reserve the fast path for tiny prompts like greetings and short follow-ups.
    if len(payload.message.split()) <= 3:
        return "fast"
    
    # Default intensive
    return "reasoning"


def get_llm(task: str, model_override: str | None = None) -> ChatOpenAI:
    """
    Retrieves the instantiated ChatOpenAI connection initialized against 
    OpenRouter depending on the assigned task category.
    """
    task_model_map = {
        "vision": settings.VISION_MODEL,
        "reasoning": settings.REASONING_MODEL,
        "fast": settings.FAST_MODEL,
        "structured": settings.STRUCTURED_MODEL,
    }
    
    model_id = model_override or task_model_map.get(task, settings.REASONING_MODEL)
    
    # Connect to OpenRouter API compatibility layer
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY.strip(),
        model=model_id,
        default_headers={
            "HTTP-Referer": settings.OPENROUTER_SITE_URL,
            "X-Title": settings.OPENROUTER_APP_NAME,
        },
        timeout=httpx.Timeout(connect=2.0, read=30.0, write=10.0, pool=5.0),
        max_retries=0,
    )


def get_model_candidates(task: str) -> list[str]:
    """Return the primary and fallback models for a task, without duplicates."""
    task_model_map = {
        "vision": settings.VISION_MODEL,
        "reasoning": settings.REASONING_MODEL,
        "fast": settings.FAST_MODEL,
        "structured": settings.STRUCTURED_MODEL,
    }

    candidates = [task_model_map.get(task, settings.REASONING_MODEL), settings.FALLBACK_MODEL]
    deduped = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped
