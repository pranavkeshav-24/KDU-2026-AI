"""
LLM integration via OpenRouter (OpenAI-compatible API).
Primary model and fallback are configured from environment variables.
"""
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import settings


def get_llm(model: str = None, fallback: bool = False) -> ChatOpenAI:
    """
    Create an OpenRouter-backed LLM using the OpenAI-compatible endpoint.

    Args:
        model: Override model string. Defaults to settings.LLM_MODEL.
        fallback: If True, use the fallback model instead of primary.

    Returns:
        ChatOpenAI instance configured for OpenRouter.
    """
    if model is None:
        model = settings.LLM_FALLBACK_MODEL if fallback else settings.LLM_MODEL

    return ChatOpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=settings.OPENROUTER_API_KEY,
        model=model,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        max_retries=0,  # Manual retries are handled in pipeline.query for explicit fallback behavior.
        default_headers={
            'HTTP-Referer': 'https://github.com/KDU-2026-AI',
            'X-Title': 'KDU RAG Teaching Assistant',
        },
    )


def build_rag_chain(llm: ChatOpenAI, prompt: ChatPromptTemplate):
    """
    Build a simple LangChain LCEL chain: prompt | llm | output_parser.

    Args:
        llm: ChatOpenAI (or compatible) LLM instance.
        prompt: ChatPromptTemplate with {context} and {question} placeholders.

    Returns:
        Runnable chain that accepts {'context': str, 'question': str}
        and returns a plain string answer.
    """
    return prompt | llm | StrOutputParser()
