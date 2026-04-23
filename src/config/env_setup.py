import streamlit as st
import os

from dotenv import load_dotenv

def init_environment() -> None:
    """Load the master environment configs."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

def render_sidebar():
    """Renders the settings and configuration side panel."""
    st.sidebar.header("Configuration")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    serper_api_key = st.sidebar.text_input("Serper API Key", type="password", value=os.getenv("SERPER_API_KEY", ""))

    openai_api_key = openai_api_key.strip()
    return openai_api_key, serper_api_key

def check_keys(openai_api_key: str):
    """Validates provided keys."""
    if not openai_api_key:
        st.warning("Please provide an OpenAI API Key to start.")
        st.stop()

    if openai_api_key in {"your_openai_api_key_here", "OPENAI_API_KEY"}:
        st.error("OPENAI_API_KEY looks like a placeholder. Please set your real OpenAI API key in .env.")
        st.stop()

    if not openai_api_key.startswith("sk-"):
        st.error("OPENAI_API_KEY format looks invalid. OpenAI keys usually start with 'sk-'.")
        st.stop()

def compute_cost(prompt_tokens: int, completion_tokens: int, input_cost_per_1m: float, output_cost_per_1m: float) -> float:
    """Calculate the estimated session cost."""
    return (
        (prompt_tokens / 1000000) * input_cost_per_1m
        + (completion_tokens / 1000000) * output_cost_per_1m
    )
