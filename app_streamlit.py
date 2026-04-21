from __future__ import annotations

import html

import streamlit as st

from config import AppTheme
from pipeline import load_components, run_pipeline

LENGTH_OPTIONS = ("short", "medium", "long")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Newsreader:opsz,wght@6..72,500;6..72,700&display=swap');

            :root {
                --bg: #edf2f7;
                --bg-deep: #dde7f1;
                --surface: rgba(255, 255, 255, 0.88);
                --surface-strong: #ffffff;
                --surface-soft: #f5f8fc;
                --ink: #152033;
                --muted: #5f6d86;
                --line: rgba(21, 32, 51, 0.10);
                --line-strong: rgba(21, 32, 51, 0.16);
                --accent: #0d6d68;
                --accent-strong: #0a5a55;
                --accent-soft: rgba(13, 109, 104, 0.10);
                --warn: #b76b10;
                --warn-soft: rgba(183, 107, 16, 0.12);
                --shadow: 0 18px 44px rgba(21, 32, 51, 0.08);
                --shadow-soft: 0 10px 24px rgba(21, 32, 51, 0.05);
                --radius-lg: 24px;
                --radius-md: 18px;
                --radius-sm: 14px;
            }

            .stApp {
                background:
                    radial-gradient(1200px 520px at 8% -10%, rgba(13, 109, 104, 0.13), transparent 62%),
                    radial-gradient(1100px 560px at 100% 0%, rgba(31, 91, 155, 0.12), transparent 55%),
                    linear-gradient(180deg, var(--bg), var(--bg-deep));
                color: var(--ink);
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            .main .block-container {
                max-width: 1180px;
                padding-top: 1.2rem;
                padding-bottom: 2.4rem;
            }

            h1, h2, h3, h4 {
                font-family: 'Newsreader', serif !important;
                color: var(--ink) !important;
                letter-spacing: 0.1px;
            }

            p, li, div, label, textarea, input, .stSelectbox, .stCaption {
                font-family: 'Plus Jakarta Sans', sans-serif !important;
            }

            .hero-shell {
                display: grid;
                grid-template-columns: minmax(0, 1.9fr) minmax(280px, 1fr);
                gap: 1rem;
                padding: 1.1rem;
                border-radius: 28px;
                border: 1px solid var(--line);
                background:
                    linear-gradient(140deg, rgba(255, 255, 255, 0.96), rgba(245, 248, 252, 0.92)),
                    var(--surface);
                box-shadow: var(--shadow);
                margin-bottom: 1.1rem;
            }

            .hero-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.42rem 0.72rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                color: var(--accent-strong);
                background: var(--accent-soft);
            }

            .hero-title {
                margin: 0.8rem 0 0.4rem 0;
                font-size: clamp(2.05rem, 3vw, 3.3rem);
                line-height: 0.96;
            }

            .hero-copy {
                margin: 0;
                max-width: 48rem;
                color: var(--muted);
                line-height: 1.65;
                font-size: 1rem;
            }

            .hero-note {
                display: grid;
                gap: 0.8rem;
                align-content: start;
            }

            .rail-card,
            .metric-tile,
            .summary-card,
            .history-card,
            .empty-state {
                border: 1px solid var(--line);
                border-radius: var(--radius-md);
                background: var(--surface-strong);
                box-shadow: var(--shadow-soft);
            }

            .rail-card {
                padding: 1rem 1.05rem;
            }

            .rail-card + .rail-card {
                margin-top: 0.85rem;
            }

            .rail-title,
            .surface-title,
            .section-title {
                margin: 0;
            }

            .rail-title {
                font-size: 1.15rem;
                margin-top: 0.12rem;
            }

            .rail-copy {
                margin: 0.48rem 0 0 0;
                color: var(--muted);
                line-height: 1.58;
            }

            .mini-list {
                display: grid;
                gap: 0.55rem;
                margin-top: 0.8rem;
            }

            .mini-item {
                display: flex;
                justify-content: space-between;
                gap: 0.8rem;
                padding: 0.58rem 0.72rem;
                border-radius: 12px;
                background: var(--surface-soft);
                color: var(--ink);
                font-size: 0.93rem;
            }

            .mini-item span:last-child {
                color: var(--muted);
            }

            .section-intro {
                margin: 1.35rem 0 0.85rem 0;
            }

            .section-kicker {
                margin: 0 0 0.28rem 0;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--accent-strong);
            }

            .section-title {
                font-size: clamp(1.5rem, 2vw, 2rem);
                line-height: 1.05;
            }

            .section-copy {
                margin: 0.38rem 0 0 0;
                color: var(--muted);
                line-height: 1.62;
                max-width: 44rem;
            }

            .surface-head {
                margin-bottom: 0.75rem;
            }

            .surface-kicker {
                margin: 0 0 0.18rem 0;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--accent-strong);
            }

            .surface-title {
                font-size: 1.35rem;
                margin-bottom: 0.22rem;
            }

            .surface-copy {
                margin: 0;
                color: var(--muted);
                line-height: 1.58;
            }

            .stTextArea textarea,
            .stTextInput input,
            .stSelectbox [data-baseweb="select"] > div {
                border-radius: 18px !important;
                border: 1px solid var(--line-strong) !important;
                background: rgba(255, 255, 255, 0.92) !important;
                box-shadow: none !important;
            }

            .stTextArea textarea {
                min-height: 320px;
                line-height: 1.62;
            }

            .stButton > button,
            .stFormSubmitButton > button {
                min-height: 3rem;
                border-radius: 16px !important;
                border: 1px solid transparent !important;
                color: #ffffff !important;
                background: linear-gradient(180deg, var(--accent), var(--accent-strong)) !important;
                font-weight: 700 !important;
                letter-spacing: 0.01em;
                box-shadow: 0 12px 24px rgba(13, 109, 104, 0.24);
                transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
            }

            .stButton > button:hover,
            .stFormSubmitButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 16px 28px rgba(13, 109, 104, 0.27);
                filter: saturate(1.03);
            }

            .stButton > button:focus,
            .stFormSubmitButton > button:focus,
            .stTextArea textarea:focus,
            .stTextInput input:focus,
            .stSelectbox [data-baseweb="select"] > div:focus-within {
                outline: none !important;
                border-color: rgba(13, 109, 104, 0.36) !important;
                box-shadow: 0 0 0 4px rgba(13, 109, 104, 0.10) !important;
            }

            .metric-tile {
                padding: 0.95rem 1rem;
            }

            .metric-label {
                margin: 0;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: var(--muted);
            }

            .metric-value {
                margin: 0.35rem 0 0.15rem 0;
                font-size: 1.7rem;
                font-weight: 800;
                line-height: 1;
                color: var(--ink);
            }

            .metric-note {
                margin: 0;
                color: var(--muted);
                font-size: 0.9rem;
            }

            .summary-card {
                padding: 1.25rem 1.3rem;
            }

            .summary-body {
                margin: 0.8rem 0 0 0;
                color: var(--ink);
                line-height: 1.72;
                white-space: pre-wrap;
            }

            .context-preview {
                margin-top: 0.75rem;
                padding: 0.8rem 0.9rem;
                border-radius: 14px;
                background: var(--surface-soft);
                color: var(--muted);
                line-height: 1.6;
                font-size: 0.94rem;
                white-space: pre-wrap;
            }

            .history-card {
                padding: 1rem;
                margin-bottom: 0.8rem;
            }

            .history-top {
                display: flex;
                justify-content: space-between;
                gap: 0.8rem;
                align-items: flex-start;
            }

            .history-question {
                margin: 0;
                font-weight: 800;
                color: var(--ink);
                line-height: 1.45;
            }

            .history-answer {
                margin: 0.55rem 0 0 0;
                color: var(--ink);
                line-height: 1.62;
                white-space: pre-wrap;
            }

            .history-note {
                margin: 0.65rem 0 0 0;
                color: var(--muted);
                font-size: 0.92rem;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                padding: 0.45rem 0.7rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 700;
                white-space: nowrap;
            }

            .badge-strong {
                background: var(--accent-soft);
                color: var(--accent-strong);
            }

            .badge-warn {
                background: var(--warn-soft);
                color: var(--warn);
            }

            .empty-state {
                padding: 1.05rem;
            }

            .empty-state p {
                margin: 0;
                color: var(--muted);
                line-height: 1.6;
            }

            .helper-list {
                display: grid;
                gap: 0.65rem;
                margin-top: 0.8rem;
            }

            .helper-item {
                padding: 0.72rem 0.82rem;
                border-radius: 12px;
                background: var(--surface-soft);
                color: var(--ink);
                line-height: 1.5;
                font-size: 0.93rem;
            }

            .stAlert {
                border-radius: 16px;
            }

            details {
                border: 1px solid var(--line);
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.74);
                padding: 0.35rem 0.8rem;
            }

            @media (max-width: 960px) {
                .hero-shell {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_components_cached():
    return load_components()


def init_state() -> None:
    if "pipeline_state" not in st.session_state:
        st.session_state.pipeline_state = None
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []


def render_metric_tile(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-tile">
            <p class="metric-label">{html.escape(label)}</p>
            <p class="metric-value">{html.escape(value)}</p>
            <p class="metric-note">{html.escape(note)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(theme: AppTheme) -> None:
    st.markdown(
        f"""
        <section class="hero-shell">
            <div>
                <span class="hero-kicker">Local multi-model workflow</span>
                <h1 class="hero-title">{html.escape(theme.title)}</h1>
                <p class="hero-copy">{html.escape(theme.subtitle)}. Build a clean summary, then pressure-test it with grounded questions in the same workspace.</p>
            </div>
            <div class="hero-note">
                <div class="rail-card">
                    <p class="surface-kicker">Flow</p>
                    <h3 class="rail-title">Summarize first, validate second</h3>
                    <p class="rail-copy">The interface keeps the writing path calm and then moves into Q&amp;A once a final summary exists.</p>
                    <div class="mini-list">
                        <div class="mini-item"><span>1. Chunk and summarize</span><span>BART</span></div>
                        <div class="mini-item"><span>2. Refine for length</span><span>DistilBART</span></div>
                        <div class="mini-item"><span>3. Ask grounded questions</span><span>RoBERTa QA</span></div>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_generation_form() -> tuple[str, str, bool]:
    st.markdown(
        """
        <div class="section-intro">
            <p class="section-kicker">Compose</p>
            <h2 class="section-title">Shape the source before the models touch it</h2>
            <p class="section-copy">Use the wide text canvas for the raw material, then tune the output length from the side rail.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("generate_form"):
        left, right = st.columns([1.65, 0.95], gap="large")

        with left:
            st.markdown(
                """
                <div class="surface-head">
                    <p class="surface-kicker">Source Text</p>
                    <h3 class="surface-title">Input document</h3>
                    <p class="surface-copy">Paste notes, articles, reports, or transcripts. The app will split long input automatically before summarizing.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            source_text = st.text_area(
                "Input text",
                label_visibility="collapsed",
                height=340,
                placeholder="Paste a long article, report, or notes here...",
            )
            st.caption("Tip: denser source material produces stronger Q&A context later.")

        with right:
            st.markdown(
                """
                <div class="rail-card">
                    <p class="surface-kicker">Settings</p>
                    <h3 class="rail-title">Output profile</h3>
                    <p class="rail-copy">Choose the summary length based on how much detail you want preserved for question answering.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            length = st.selectbox("Summary length", options=LENGTH_OPTIONS, index=1)
            submitted = st.form_submit_button("Generate Summary", use_container_width=True)
            st.markdown(
                """
                <div class="rail-card">
                    <p class="surface-kicker">Guidance</p>
                    <div class="helper-list">
                        <div class="helper-item"><strong>Short</strong> keeps only the sharpest takeaways.</div>
                        <div class="helper-item"><strong>Medium</strong> is the most balanced option for Q&amp;A.</div>
                        <div class="helper-item"><strong>Long</strong> preserves more context when precision matters.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    return source_text.strip(), length, submitted


def render_summary(state: dict) -> None:
    st.markdown(
        """
        <div class="section-intro">
            <p class="section-kicker">Result</p>
            <h2 class="section-title">Refined summary and context health</h2>
            <p class="section-copy">The final summary below is the context currently used by the Q&amp;A layer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_tile("Input words", str(len(state["raw_text"].split())), "Source size")
    with metric_cols[1]:
        render_metric_tile("Chunks", str(len(state.get("chunks", []))), "Preprocessed segments")
    with metric_cols[2]:
        render_metric_tile("Partial summaries", str(len(state.get("partial_summaries", []))), "Intermediate merges")
    with metric_cols[3]:
        render_metric_tile("Final words", str(len(state["refined_summary"].split())), state["length"].title())

    left, right = st.columns([1.55, 0.95], gap="large")

    with left:
        safe_summary = html.escape(state["refined_summary"])
        st.markdown(
            f"""
            <div class="summary-card">
                <p class="surface-kicker">Final Output</p>
                <h3 class="surface-title">Refined summary</h3>
                <p class="surface-copy">This is the polished version generated after abstractive summarization and refinement.</p>
                <p class="summary-body">{safe_summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        raw_preview = state["raw_summary"].strip()
        preview_text = raw_preview if len(raw_preview) <= 420 else f"{raw_preview[:420].rstrip()}..."
        st.markdown(
            f"""
            <div class="rail-card">
                <p class="surface-kicker">Intermediate Pipeline State</p>
                <h3 class="rail-title">Raw merged summary</h3>
                <p class="rail-copy">For comparison, this is the merged abstractive summary before the length-control refinement stage.</p>
                <div class="context-preview">{html.escape(preview_text or 'No intermediate summary available yet.')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Show merged summary before refinement"):
            st.write(state["raw_summary"] or "No raw summary available.")


def render_qa_history() -> None:
    if not st.session_state.qa_history:
        st.markdown(
            """
            <div class="empty-state">
                <p>No questions yet. Ask about facts, timelines, entities, or relationships inside the generated summary.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for item in st.session_state.qa_history[:12]:
        confidence_class = "badge-warn" if item["low_confidence"] else "badge-strong"
        confidence_text = f"Confidence {item['score']:.2f}"
        used_fallback = bool(item.get("used_fallback", False))
        followup = (
            "Low confidence. The summary may not support this answer strongly."
            if item["low_confidence"]
            else "Answer is grounded in the current summary context."
        )
        if used_fallback:
            followup = f"{followup} Retried with raw merged summary context."
        st.markdown(
            f"""
            <div class="history-card">
                <div class="history-top">
                    <p class="history-question">Q: {html.escape(item["question"])}</p>
                    <span class="badge {confidence_class}">{html.escape(confidence_text)}</span>
                </div>
                <p class="history-answer">A: {html.escape(item["answer"])}</p>
                <p class="history-note">{html.escape(followup)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_qa_interface() -> None:
    if not st.session_state.pipeline_state:
        return

    st.markdown(
        """
        <div class="section-intro">
            <p class="section-kicker">Validate</p>
            <h2 class="section-title">Ask grounded questions against the final summary</h2>
            <p class="section-copy">Use the composer on the left and review answer quality on the right. This keeps follow-up questions close to the summary they depend on.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="rail-card">
                <p class="surface-kicker">Question Composer</p>
                <h3 class="rail-title">Interrogate the summary</h3>
                <p class="rail-copy">Direct questions work best. Ask who, what, when, where, why, or how about details that should still be present after summarization.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("qa_form", clear_on_submit=True):
            question = st.text_input(
                "Question",
                label_visibility="collapsed",
                placeholder="Ask about the generated summary...",
            )
            ask_clicked = st.form_submit_button("Ask Question", use_container_width=True)

        st.markdown(
            """
            <div class="rail-card">
                <p class="surface-kicker">Good Prompts</p>
                <div class="helper-list">
                    <div class="helper-item">Who is the main subject and what do they do?</div>
                    <div class="helper-item">What event happened first, and what followed?</div>
                    <div class="helper-item">Which claim is supported directly by the summary?</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if ask_clicked and question.strip():
            components = get_components_cached()
            result = components.qa_model.answer(
                question=question.strip(),
                context=st.session_state.pipeline_state["refined_summary"],
            )
            used_fallback = False

            raw_context = st.session_state.pipeline_state.get("raw_summary", "").strip()
            refined_context = st.session_state.pipeline_state["refined_summary"].strip()
            if result["low_confidence"] and raw_context and raw_context != refined_context:
                fallback_result = components.qa_model.answer(
                    question=question.strip(),
                    context=raw_context,
                )
                if fallback_result["score"] > result["score"]:
                    result = fallback_result
                    used_fallback = True

            st.session_state.qa_history.insert(
                0,
                {
                    "question": question.strip(),
                    "answer": result["answer"],
                    "score": result["score"],
                    "low_confidence": result["low_confidence"],
                    "used_fallback": used_fallback,
                },
            )

    with right:
        st.markdown(
            """
            <div class="surface-head">
                <p class="surface-kicker">History</p>
                <h3 class="surface-title">Latest answers</h3>
                <p class="surface-copy">Recent Q&amp;A stays visible so you can quickly judge whether the summary is supporting the questions you care about.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_qa_history()


def main() -> None:
    st.set_page_config(page_title="Tri-Model AI Assistant", layout="wide")
    inject_styles()
    init_state()

    theme = AppTheme()
    render_header(theme)

    source_text, length, submitted = render_generation_form()

    if submitted:
        if not source_text:
            st.error("Please provide input text.")
        else:
            with st.spinner("Loading models and generating summary..."):
                components = get_components_cached()
                state = run_pipeline(raw_text=source_text, length=length, components=components)
            st.session_state.pipeline_state = state
            st.session_state.qa_history = []

    if st.session_state.pipeline_state:
        render_summary(st.session_state.pipeline_state)

    render_qa_interface()


if __name__ == "__main__":
    main()
