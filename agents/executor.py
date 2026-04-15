"""Core agent pipeline orchestrating tool bindings, parsing, and execution."""

import asyncio

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import settings
from memory.store import get_session_history
from models.router import classify_task, get_llm, get_model_candidates
from models.schemas import AssistantResponse, ChatRequest
from tools.image_analysis import build_vision_message
from tools.weather import build_weather_tool

base_parser = PydanticOutputParser(pydantic_object=AssistantResponse)

SYSTEM_PROMPT = """
You are a helpful Multimodal AI Assistant.
Always respond following the strict JSON format instructions provided below.

User Context:
Name: {user_name}
Location: {user_city} (Lat: {user_lat}, Lon: {user_lon})
Timezone: {user_timezone}
Language: {user_lang}
Unit Preference: {unit_system}

Persona Directive:
{persona}

Format Instructions:
{format_instructions}
"""


def _save_turn(thread_id: str, user_message: str, assistant_message: str) -> None:
    """Persist local-mode conversations so the UI still behaves consistently."""
    history = get_session_history(thread_id)
    history.add_user_message(user_message)
    history.add_ai_message(assistant_message)


def _build_fallback_response(
    message: str,
    user_profile,
    has_image: bool,
    error: Exception | None = None,
) -> AssistantResponse:
    """Return a structured response when live model calls are unavailable."""
    city = getattr(user_profile, "city", None)
    location_note = f" for {city}" if city else ""
    error_text = str(error) if error else ""

    if "temporarily rate-limited upstream" in error_text:
        content = (
            "The free OpenRouter model is temporarily rate-limited upstream right now. "
            "Please retry in a moment."
        )
        intent = "assistant_rate_limited"
    elif "No endpoints found" in error_text:
        content = (
            "The configured OpenRouter model is no longer available. "
            "The repo needs its default model IDs updated."
        )
        intent = "assistant_model_unavailable"
    elif "Insufficient credits" in error_text:
        content = (
            "This request hit a paid OpenRouter model, but the current account has no credits. "
            "Switch to a free model or add credits to the account."
        )
        intent = "assistant_insufficient_credits"

    elif has_image:
        content = (
            "I received your image, but live vision analysis is unavailable right now. "
            "Add `OPENROUTER_API_KEY` to enable image understanding."
        )
        intent = "vision_unavailable"
    elif any(token in message.lower() for token in ["weather", "temperature", "forecast", "humidity"]):
        content = (
            f"I can help with weather{location_note}, but the live model or weather service "
            "is not configured yet. Add `OPENROUTER_API_KEY` and "
            "`OPENWEATHERMAP_API_KEY` in `.env` to enable it."
        )
        intent = "weather_unavailable"
    elif not settings.OPENROUTER_API_KEY:
        content = (
            "Nova is running, but live AI responses are disabled because "
            "`OPENROUTER_API_KEY` is missing. Add it in `.env` and restart the app."
        )
        intent = "assistant_unconfigured"
    else:
        content = (
            "I couldn't reach the model service for that request, so the UI stayed online "
            "but the answer could not be generated right now. Please try again in a moment."
        )
        intent = "assistant_temporarily_unavailable"

    if error:
        content = f"{content}\n\nDetails: {str(error)}"

    return AssistantResponse(
        intent=intent,
        content=content,
        follow_up="Once the required services are available, try the same prompt again.",
    )


def _is_rate_limited_error(error: Exception) -> bool:
    """Identify provider throttling responses that should be retried."""
    err = str(error).lower()
    return (
        "429" in err
        or "rate-limited upstream" in err
        or "rate limit" in err
        or "too many requests" in err
    )


async def orchestrate_chat(
    message: str,
    thread_id: str,
    user_profile,
    persona_prompt: str,
    image_bytes: bytes = None,
) -> AssistantResponse:
    """Main entry point for evaluating a request over the assistant pipeline."""
    has_image = bool(image_bytes)
    payload = ChatRequest(message=message, thread_id=thread_id)

    if not settings.OPENROUTER_API_KEY:
        fallback = _build_fallback_response(message, user_profile, has_image)
        _save_turn(thread_id, message, fallback.content)
        return fallback

    task_type = classify_task(payload=payload, has_image=has_image)
    last_error = None

    for model_id in get_model_candidates(task_type):
        attempts = max(settings.OPENROUTER_RETRY_ATTEMPTS, 0) + 1

        for attempt in range(attempts):
            try:
                llm = get_llm(task_type, model_override=model_id)

                if has_image:
                    message_payload = build_vision_message(image_bytes=image_bytes, user_text=message)
                    result = await llm.ainvoke([message_payload])
                    parsed = base_parser.parse(result.content)
                    _save_turn(thread_id, message, parsed.content)
                    return parsed

                tools = [build_weather_tool()]

                prompt = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ])

                agent = create_tool_calling_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                chain_with_history = RunnableWithMessageHistory(
                    agent_executor,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="history",
                )

                response = await chain_with_history.ainvoke(
                    {
                        "input": message,
                        "user_name": getattr(user_profile, "name", "Guest"),
                        "user_city": getattr(user_profile, "city", "Unknown"),
                        "user_lat": getattr(user_profile, "lat", 0.0),
                        "user_lon": getattr(user_profile, "lon", 0.0),
                        "user_timezone": getattr(user_profile, "timezone", "UTC"),
                        "user_lang": getattr(user_profile, "language", "en"),
                        "unit_system": getattr(user_profile, "unit_system", "metric"),
                        "persona": persona_prompt,
                        "format_instructions": base_parser.get_format_instructions(),
                    },
                    config={"configurable": {"session_id": thread_id}},
                )

                return base_parser.parse(response["output"])
            except Exception as exc:
                last_error = exc
                should_retry = _is_rate_limited_error(exc) and attempt < attempts - 1
                if should_retry:
                    wait_seconds = settings.OPENROUTER_RETRY_BASE_DELAY_SECONDS * (2**attempt)
                    await asyncio.sleep(wait_seconds)
                    continue
                break

    fallback = _build_fallback_response(message, user_profile, has_image, error=last_error)
    _save_turn(thread_id, message, fallback.content)
    return fallback
