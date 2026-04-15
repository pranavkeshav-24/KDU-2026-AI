"""
Streamlit UI entrypoint for the Context-Aware Hybrid-Search RAG Teaching Assistant.
Provides:
  - Sidebar: PDF upload and URL ingestion with status feedback
  - Main area: Chat interface with source citation expander
"""
import sys
import os

# Ensure the rag_chatbot package root is on the path when run from parent dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from pipeline import RAGPipeline
from ingestion.loaders import save_uploaded_file
from config import settings

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title='RAG Teaching Assistant',
    page_icon='🎓',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: #0E1117;
        color: #e8e8e8;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0E1117;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #0E1117;
        border-radius: 12px;
        padding: 8px;
        margin: 4px 0;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Input box */
    [data-testid="stChatInput"] textarea {
        background: #0E1117;
        border-radius: 12px !important;
        color: #fff !important;
        border: 1px solid rgba(138,43,226,0.5) !important;
    }

    /* Success/info boxes */
    .stSuccess, .stInfo {
        background: rgba(0,200,100,0.1) !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #8a2be2, #4169e1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.4rem 1rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
    }

    /* Source chip */
    .source-chip {
        display: inline-block;
        background: rgba(138,43,226,0.25);
        border: 1px solid rgba(138,43,226,0.5);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
        color: #c8a8ff;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    h1, h2, h3 { color: #c8a8ff !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        color: #aaa;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        color: #c8a8ff !important;
        border-bottom-color: #8a2be2 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Pipeline singleton (loaded once, cached across reruns) ───────────────────
@st.cache_resource(show_spinner='Loading AI pipeline…')
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipeline = get_pipeline()


# ── Sidebar — Document Ingestion ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## RAG Teaching Assistant")
    st.caption("Hybrid-Search · Semantic Chunking · Cross-Encoder Reranking")
    st.divider()

    st.markdown("### Add Documents")
    tab_pdf, tab_url = st.tabs(['📄 PDF Upload', '🌐 Blog URL'])

    with tab_pdf:
        uploaded = st.file_uploader(
            'Upload a PDF',
            type=['pdf'],
            help='PDF will be semantically chunked and indexed.',
        )
        if uploaded:
            if st.button('Ingest PDF', key='btn_pdf', use_container_width=True):
                with st.spinner('Chunking and embedding…'):
                    try:
                        file_path = save_uploaded_file(uploaded, settings.UPLOAD_DIR)
                        result = pipeline.ingest(file_path, source_type='pdf')
                        st.success(
                            f"Ingested **{result['num_chunks']}** chunks "
                            f"from **{result['num_docs']}** page(s)"
                        )
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

    with tab_url:
        url_input = st.text_input(
            'Enter a blog / article URL',
            placeholder='https://example.com/article',
        )
        if url_input:
            if st.button('Ingest URL', key='btn_url', use_container_width=True):
                with st.spinner('Fetching and embedding…'):
                    try:
                        result = pipeline.ingest(url_input, source_type='url')
                        st.success(
                            f"Ingested **{result['num_chunks']}** chunks"
                        )
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

    st.divider()

    # ── Pipeline Stats
    st.markdown("### Pipeline Status")
    stats = pipeline.get_stats()
    col1, col2 = st.columns(2)
    col1.metric('ChromaDB', stats['chroma_count'], help='Total chunks in vector store')
    col2.metric('BM25 Index', stats['total_chunks_bm25'], help='Total chunks in BM25 index')

    if stats['bm25_ready']:
        st.success('Ready to answer questions')
    else:
        st.warning('No documents ingested yet')

    st.divider()

    # ── Clear chat
    if st.button('Clear Chat History', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style="text-align:center; font-size:0.7rem; color:#888; margin-top:16px;">
        Powered by BAAI/bge-small · ChromaDB · BM25<br>
        Cross-Encoder Reranking · OpenRouter Gemma 3 27B
    </div>
    """, unsafe_allow_html=True)


# ── Main Area — Chat Interface ───────────────────────────────────────────────
st.markdown("## Ask Questions About Your Documents")
st.caption(
    "Hybrid search (semantic + BM25) + cross-encoder reranking ensures accurate, grounded answers."
)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg.get('sources'):
            with st.expander('📎 Source Chunks Used', expanded=False):
                for i, chunk in enumerate(msg['sources'], 1):
                    score = chunk.metadata.get('rerank_score', 'N/A')
                    rrf = chunk.metadata.get('rrf_score', 'N/A')
                    src = chunk.metadata.get('source', 'unknown')
                    cid = chunk.metadata.get('chunk_id', '?')
                    page = chunk.metadata.get('page', '')

                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(
                            f"**Chunk {cid}** — "
                            f"<span class='source-chip'>{os.path.basename(str(src))}</span>"
                            + (f" <span class='source-chip'>Page {page}</span>" if page else ""),
                            unsafe_allow_html=True,
                        )
                    with col_b:
                        st.markdown(
                            f"<small>Rerank: **{score}**</small>",
                            unsafe_allow_html=True,
                        )
                    st.text(chunk.page_content[:400] + ('…' if len(chunk.page_content) > 400 else ''))
                    if i < len(msg['sources']):
                        st.divider()

# ── Chat Input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input('Ask a question about your documents…'):

    if not stats['bm25_ready'] and pipeline.get_stats()['chroma_count'] == 0:
        st.warning('⚠️ Please ingest at least one document (PDF or URL) before asking questions.')
    else:
        # Display user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        # Generate response
        with st.chat_message('assistant'):
            with st.spinner('Retrieving and generating…'):
                try:
                    result = pipeline.query(prompt)
                    answer = result['answer']
                    sources = result['sources']

                    st.markdown(answer)

                    # Source citations expander
                    with st.expander('📎 Source Chunks Used', expanded=False):
                        for i, chunk in enumerate(sources, 1):
                            score = chunk.metadata.get('rerank_score', 'N/A')
                            src = chunk.metadata.get('source', 'unknown')
                            cid = chunk.metadata.get('chunk_id', '?')
                            page = chunk.metadata.get('page', '')

                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.markdown(
                                    f"**Chunk {cid}** — "
                                    f"<span class='source-chip'>{os.path.basename(str(src))}</span>"
                                    + (f" <span class='source-chip'>Page {page}</span>" if page else ""),
                                    unsafe_allow_html=True,
                                )
                            with col_b:
                                st.markdown(
                                    f"<small>Rerank: **{score}**</small>",
                                    unsafe_allow_html=True,
                                )
                            st.text(chunk.page_content[:400] + ('…' if len(chunk.page_content) > 400 else ''))
                            if i < len(sources):
                                st.divider()

                    # Persist to session state
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': answer,
                        'sources': sources,
                    })

                except RuntimeError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"❌ Query failed: {e}")
