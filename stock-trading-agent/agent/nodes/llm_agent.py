import os
import time

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..state import AgentState, ensure_state_defaults

MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

SYSTEM_PROMPT = (
    "You are a concise assistant inside a stock trading demo. "
    "Give helpful answers for general questions. "
    "Specific price lookups, conversions, portfolio summaries, and trade execution "
    "are handled by other graph nodes, so do not invent transactions."
)

last_call_time = 0.0


def has_real_api_key() -> bool:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return False

    masked_values = {"dummy-key", "sk-or-xxxxxxxxxxxx"}
    return api_key not in masked_values and "xxxxxxxx" not in api_key


def get_llm_with_fallback() -> ChatOpenAI | None:
    if not has_real_api_key():
        return None

    global last_call_time
    elapsed = time.time() - last_call_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    for model in MODELS:
        try:
            llm = ChatOpenAI(
                model=model,
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=0.1,
                max_retries=1,
            )
            last_call_time = time.time()
            return llm
        except Exception:
            continue
    return None

def llm_agent(state: AgentState) -> dict:
    safe_state = ensure_state_defaults(state)
    llm = get_llm_with_fallback()
    if llm is None:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I can help with stock prices, USD conversions, portfolio summaries, "
                        "and trade previews locally. Add a real `OPENROUTER_API_KEY` to `.env` "
                        "if you also want open-ended chat replies."
                    )
                )
            ]
        }

    messages = [SystemMessage(content=SYSTEM_PROMPT), *safe_state["messages"][-10:]]

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I couldn't reach the configured chat model right now. "
                        f"Details: {exc}"
                    )
                )
            ]
        }

    usage = response.response_metadata.get("token_usage", {})
    tokens = usage.get("total_tokens", 0)

    return {
        "messages": [response],
        "total_tokens": safe_state.get("total_tokens", 0) + tokens,
    }
