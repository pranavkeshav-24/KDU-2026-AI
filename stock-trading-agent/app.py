import os
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage as HumanMessageType

from agent.graph import build_graph
from agent.state import ensure_state_defaults

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

st.set_page_config(page_title="Stock Trading Agent", layout="wide")

THEMES: dict[str, dict[str, str]] = {
    "harbor_light": {
        "name": "Harbor Light",
        "bg_start": "#ecf3fb",
        "bg_end": "#f8fbff",
        "bg_glow_a": "rgba(14, 165, 166, 0.16)",
        "bg_glow_b": "rgba(37, 99, 235, 0.12)",
        "sidebar_start": "rgba(255,255,255,0.98)",
        "sidebar_end": "rgba(248,252,255,0.96)",
        "card_bg": "rgba(255, 255, 255, 0.88)",
        "hero_start": "rgba(14, 165, 166, 0.16)",
        "hero_end": "rgba(37, 99, 235, 0.12)",
        "hero_border": "rgba(14, 165, 166, 0.24)",
        "text_primary": "#102a43",
        "text_muted": "#486581",
        "accent": "#0ea5a6",
        "accent_strong": "#0f766e",
        "outline": "rgba(16, 42, 67, 0.14)",
        "warning": "#9a3412",
        "warning_bg": "#fff7ed",
        "input_bg": "rgba(255,255,255,0.88)",
    },
    "slate_night": {
        "name": "Slate Night",
        "bg_start": "#0d1424",
        "bg_end": "#0f1b33",
        "bg_glow_a": "rgba(56, 189, 248, 0.20)",
        "bg_glow_b": "rgba(45, 212, 191, 0.16)",
        "sidebar_start": "rgba(12,18,34,0.96)",
        "sidebar_end": "rgba(16,24,44,0.94)",
        "card_bg": "rgba(15, 25, 48, 0.78)",
        "hero_start": "rgba(56, 189, 248, 0.22)",
        "hero_end": "rgba(45, 212, 191, 0.16)",
        "hero_border": "rgba(56, 189, 248, 0.32)",
        "text_primary": "#e6eefc",
        "text_muted": "#a9c0e6",
        "accent": "#22d3ee",
        "accent_strong": "#14b8a6",
        "outline": "rgba(148, 184, 246, 0.24)",
        "warning": "#fdba74",
        "warning_bg": "rgba(124, 45, 18, 0.25)",
        "input_bg": "rgba(15, 25, 48, 0.86)",
    },
    "sandstone": {
        "name": "Sandstone",
        "bg_start": "#f6f0e8",
        "bg_end": "#f9f5ef",
        "bg_glow_a": "rgba(217, 119, 6, 0.15)",
        "bg_glow_b": "rgba(20, 184, 166, 0.10)",
        "sidebar_start": "rgba(255,251,244,0.98)",
        "sidebar_end": "rgba(250,245,236,0.95)",
        "card_bg": "rgba(255, 251, 245, 0.88)",
        "hero_start": "rgba(217, 119, 6, 0.16)",
        "hero_end": "rgba(20, 184, 166, 0.11)",
        "hero_border": "rgba(180, 83, 9, 0.25)",
        "text_primary": "#2d1f13",
        "text_muted": "#6b4c36",
        "accent": "#c2410c",
        "accent_strong": "#0f766e",
        "outline": "rgba(75, 45, 20, 0.18)",
        "warning": "#9a3412",
        "warning_bg": "#fff2df",
        "input_bg": "rgba(255,251,245,0.92)",
    },
}

DEFAULT_THEME_KEY = "harbor_light"
INTERRUPT_KEY = "__interrupt__"


def _initial_theme_key() -> str:
    theme_from_env = os.getenv("APP_THEME", "").strip().lower().replace("-", "_")
    if theme_from_env in THEMES:
        return theme_from_env
    return DEFAULT_THEME_KEY


def _build_theme_css(theme: dict[str, str]) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
  --bg-start: {theme["bg_start"]};
  --bg-end: {theme["bg_end"]};
  --bg-glow-a: {theme["bg_glow_a"]};
  --bg-glow-b: {theme["bg_glow_b"]};
  --sidebar-start: {theme["sidebar_start"]};
  --sidebar-end: {theme["sidebar_end"]};
  --card-bg: {theme["card_bg"]};
  --hero-start: {theme["hero_start"]};
  --hero-end: {theme["hero_end"]};
  --hero-border: {theme["hero_border"]};
  --text-primary: {theme["text_primary"]};
  --text-muted: {theme["text_muted"]};
  --accent: {theme["accent"]};
  --accent-strong: {theme["accent_strong"]};
  --outline: {theme["outline"]};
  --warning: {theme["warning"]};
  --warning-bg: {theme["warning_bg"]};
  --input-bg: {theme["input_bg"]};
}}

html, body, [class*="css"] {{
  font-family: 'Space Grotesk', sans-serif;
  color: var(--text-primary);
}}

.stApp {{
  background:
    radial-gradient(1200px 500px at 90% -10%, var(--bg-glow-a), transparent 45%),
    radial-gradient(900px 400px at 0% 0%, var(--bg-glow-b), transparent 45%),
    linear-gradient(155deg, var(--bg-start), var(--bg-end));
}}

.block-container {{
  padding-top: 1.35rem;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, var(--sidebar-start), var(--sidebar-end));
  border-right: 1px solid var(--outline);
}}

[data-testid="stSidebar"] * {{
  color: var(--text-primary);
}}

.hero-card {{
  background: linear-gradient(115deg, var(--hero-start), var(--hero-end));
  border: 1px solid var(--hero-border);
  border-radius: 18px;
  padding: 1rem 1.2rem;
  margin-bottom: 1rem;
}}

.hero-card h1 {{
  margin: 0;
  color: var(--text-primary);
  font-size: 1.3rem;
  font-weight: 700;
}}

.hero-card p {{
  margin: .35rem 0 0;
  color: var(--text-muted);
}}

.section-title {{
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--text-muted);
  margin-bottom: 0.45rem;
}}

.metric-card {{
  background: var(--card-bg);
  border: 1px solid var(--outline);
  border-radius: 14px;
  padding: .68rem .86rem;
  margin-bottom: .58rem;
  backdrop-filter: blur(6px);
}}

.metric-label {{
  color: var(--text-muted);
  font-size: .76rem;
  margin-bottom: .18rem;
}}

.metric-value {{
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.02rem;
}}

.state-badge {{
  display: inline-block;
  padding: .2rem .5rem;
  border-radius: 999px;
  border: 1px solid var(--hero-border);
  color: var(--accent-strong);
  background: var(--hero-start);
  font-family: 'IBM Plex Mono', monospace;
  font-size: .72rem;
}}

.approval-panel {{
  background: var(--warning-bg);
  border: 1px solid var(--hero-border);
  border-radius: 14px;
  padding: .8rem 1rem;
  margin: .72rem 0 .4rem;
}}

.approval-panel strong {{
  color: var(--warning);
}}

[data-testid="stChatInput"] {{
  background: var(--input-bg);
  border: 1px solid var(--outline);
  border-radius: 12px;
}}

div.stButton > button {{
  border-radius: 10px;
  border: 1px solid var(--outline);
}}

div.stButton > button:hover {{
  border-color: var(--accent);
}}

.theme-note {{
  color: var(--text-muted);
  font-size: .79rem;
  margin-top: .35rem;
}}
</style>
"""


def _to_money(value: float) -> str:
    return f"${value:,.2f}"


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and "text" in item:
                chunks.append(str(item["text"]))
        return "\n".join(chunks)
    return str(content)


def _find_pending_interrupt(state_snapshot: Any) -> bool:
    if not state_snapshot:
        return False
    task_interrupts = []
    for task in getattr(state_snapshot, "tasks", []) or []:
        task_interrupts.extend(getattr(task, "interrupts", []) or [])
    return len(task_interrupts) > 0


def _render_sidebar_metrics(state_values: dict[str, Any], theme_name: str) -> None:
    portfolio = state_values["portfolio"]
    st.markdown("<div class='section-title'>Portfolio</div>", unsafe_allow_html=True)
    st.markdown(
        (
            "<div class='metric-card'>"
            "<div class='metric-label'>Cash</div>"
            f"<div class='metric-value'>{_to_money(portfolio['cash_usd'])}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            "<div class='metric-card'>"
            "<div class='metric-label'>Total Value</div>"
            f"<div class='metric-value'>{_to_money(portfolio['total_value_usd'])}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    holdings = portfolio.get("holdings", {})
    if holdings:
        st.markdown("<div class='section-title'>Holdings</div>", unsafe_allow_html=True)
        for symbol, pos in sorted(holdings.items()):
            st.markdown(
                (
                    "<div class='metric-card'>"
                    f"<div class='metric-label'>{symbol}</div>"
                    f"<div class='metric-value'>{pos['qty']} share(s) @ {_to_money(pos['avg_cost'])}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
    else:
        st.caption("No open positions yet.")

    st.markdown("<div class='section-title'>Runtime</div>", unsafe_allow_html=True)
    st.markdown(
        (
            f"<span class='state-badge'>tokens={state_values.get('total_tokens', 0)}</span> "
            f"<span class='state-badge'>theme={theme_name.lower().replace(' ', '_')}</span>"
        ),
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_graph():
    db_path = os.getenv("SQLITE_DB_PATH", "checkpoints.db")
    return build_graph(str(PROJECT_DIR / db_path))


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "theme_key" not in st.session_state:
    st.session_state.theme_key = _initial_theme_key()

theme_keys = list(THEMES.keys())
if st.session_state.theme_key not in THEMES:
    st.session_state.theme_key = DEFAULT_THEME_KEY
active_theme = THEMES[st.session_state.theme_key]

st.markdown(_build_theme_css(active_theme), unsafe_allow_html=True)

graph = get_graph()
config = {"configurable": {"thread_id": st.session_state.thread_id}}
state_snapshot = graph.get_state(config)
state_values = ensure_state_defaults(state_snapshot.values if state_snapshot else None)
pending_interrupt = _find_pending_interrupt(state_snapshot)

with st.sidebar:
    selected_theme_key = st.selectbox(
        "Theme",
        options=theme_keys,
        index=theme_keys.index(st.session_state.theme_key),
        format_func=lambda key: THEMES[key]["name"],
    )
    if selected_theme_key != st.session_state.theme_key:
        st.session_state.theme_key = selected_theme_key
        st.rerun()

    st.markdown(
        "<div class='theme-note'>Themes adjust all surface tokens together so the interface remains balanced.</div>",
        unsafe_allow_html=True,
    )

    if st.button("Start New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    _render_sidebar_metrics(state_values, THEMES[st.session_state.theme_key]["name"])

st.markdown(
    f"""
    <div class="hero-card">
      <h1>Stock Trading Agent</h1>
      <p>{THEMES[st.session_state.theme_key]["name"]} theme is active. Ask for quotes, conversions, and portfolio insights. Trade actions pause for confirmation before execution.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

messages = state_values.get("messages", [])
for message in messages:
    if isinstance(message, HumanMessageType):
        with st.chat_message("user"):
            st.markdown(_message_text(message))
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(_message_text(message))

if pending_interrupt:
    st.markdown(
        "<div class='approval-panel'><strong>Trade approval needed.</strong> Approve to continue execution or reject to cancel this action.</div>",
        unsafe_allow_html=True,
    )
    col_approve, col_reject = st.columns(2)
    if col_approve.button("Approve Trade", use_container_width=True, type="primary"):
        try:
            graph.invoke({"human_approved": True}, config=config)
        except Exception as exc:
            st.error(f"Approval failed: {exc}")
        st.rerun()
    if col_reject.button("Reject Trade", use_container_width=True):
        try:
            graph.invoke({"human_approved": False}, config=config)
        except Exception as exc:
            st.error(f"Rejection failed: {exc}")
        st.rerun()
else:
    prompt = st.chat_input("Try: price of AAPL, convert 100 USD to INR, or show my portfolio")
    if prompt:
        try:
            result = graph.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
            if isinstance(result, dict) and result.get(INTERRUPT_KEY):
                st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")
        else:
            st.rerun()
