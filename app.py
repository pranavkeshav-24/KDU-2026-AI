from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import config, ensure_data_dirs
from src.llm.openai_client import OpenAIClient
from src.orchestrator import ProcessingOrchestrator
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.vector_store import VectorStore
from src.storage.db import Database
from src.upload_service import UploadService
from src.utils.file_utils import validate_upload


ensure_data_dirs(config)
db = Database(config.db_path)

st.set_page_config(page_title=config.app_name, page_icon="CAS", layout="wide")


def service_search() -> HybridSearch:
    llm = OpenAIClient(config)
    return HybridSearch(
        db=db,
        embedder=Embedder(llm, config),
        vector_store=VectorStore(db),
        keyword_index=KeywordIndex(db),
        app_config=config,
    )


def rows_as_dicts(rows) -> list[dict]:
    return [dict(row) for row in rows]


def selected_file_id(label: str = "File") -> str | None:
    files = db.list_files()
    if not files:
        st.info("No files have been processed yet.")
        return None
    options = {f"{row['file_name']} ({row['status']})": row["file_id"] for row in files}
    return options[st.selectbox(label, list(options.keys()))]


def page_upload() -> None:
    st.header("Upload & Process")
    mode = st.selectbox("Processing mode", ["balanced", "fast", "deep"], index=0)
    uploaded = st.file_uploader("Upload PDF, image, or audio", type=["pdf", "jpg", "jpeg", "png", "mp3", "wav"])

    if uploaded is None:
        return

    valid, message = validate_upload(uploaded.name, uploaded.size, config.max_file_size_mb)
    if valid:
        st.success(message)
    else:
        st.error(message)
        return

    estimate_cols = st.columns(4)
    estimate_cols[0].metric("File size", f"{uploaded.size / 1024 / 1024:.2f} MB")
    estimate_cols[1].metric("Mode", mode.title())
    estimate_cols[2].metric("Vision policy", "Selective")
    estimate_cols[3].metric("API readiness", "OpenAI" if config.openai_api_key else "Local fallback")

    if st.button("Process file", type="primary"):
        upload_service = UploadService(config, db)
        try:
            file_id, _path = upload_service.save_upload(uploaded, uploaded.name, uploaded.size, mode)
            status = st.status("Processing file", expanded=True)
            status.write("File saved and metadata record created.")
            document = ProcessingOrchestrator(db, config).process_file(file_id)
            status.write("Extraction, summarization, chunking, indexing, and cost logging completed.")
            status.update(label="Processing complete", state="complete")
            st.subheader("Summary")
            st.write(document.summary)
            if document.key_points:
                st.subheader("Key Points")
                for point in document.key_points:
                    st.write(f"- {point}")
            warnings = list(document.warnings)
            for page in document.pages:
                warnings.extend(page.warnings)
            if warnings:
                st.warning("\n".join(dict.fromkeys(warnings)))
        except Exception as exc:
            st.error(str(exc))


def page_processed_files() -> None:
    st.header("Processed Files")
    files = db.list_files()
    if not files:
        st.info("No files found.")
        return

    st.dataframe(
        [
            {
                "File": row["file_name"],
                "Type": row["file_type"],
                "Status": row["status"],
                "Mode": row["processing_mode"],
                "Created": row["created_at"],
            }
            for row in files
        ],
        use_container_width=True,
        hide_index=True,
    )

    file_id = selected_file_id("Open file")
    if not file_id:
        return
    file_row = db.get_file(file_id)
    output = db.get_output(file_id)
    stats = db.file_stats(file_id)
    pages = db.list_pages(file_id)

    st.subheader(file_row["file_name"])
    cols = st.columns(4)
    cols[0].metric("Pages", stats["pages"])
    cols[1].metric("Chunks", stats["chunks"])
    cols[2].metric("Vision pages", stats["vision_pages"])
    cols[3].metric("Status", file_row["status"])

    if output:
        st.write(output["summary"] or "")
        key_points = json.loads(output["key_points_json"] or "[]")
        tags = json.loads(output["topic_tags_json"] or "[]")
        if key_points:
            st.markdown("**Key points**")
            for point in key_points:
                st.write(f"- {point}")
        if tags:
            st.markdown("**Tags**")
            st.write(", ".join(tags))
        if output["accessibility_notes"]:
            st.markdown("**Accessibility notes**")
            st.write(output["accessibility_notes"])

        warnings = []
        for page in pages:
            if page["warning_message"]:
                warnings.append(f"Page {page['page_number']}: {page['warning_message']}")
        if warnings:
            st.markdown("**Warnings**")
            for warning in warnings:
                st.warning(warning)

        base = Path(file_row["file_name"]).stem
        st.download_button(
            "Download text",
            output["full_text"] or "",
            file_name=f"{base}.txt",
            mime="text/plain",
        )
        json_path = config.processed_dir / f"{file_id}.json"
        if json_path.exists():
            st.download_button(
                "Download JSON",
                json_path.read_text(encoding="utf-8"),
                file_name=f"{base}.json",
                mime="application/json",
            )

    if pages:
        with st.expander("Page inventory"):
            st.dataframe(rows_as_dicts(pages), use_container_width=True, hide_index=True)


def page_search() -> None:
    st.header("Search")
    query = st.text_input("Query")
    files = db.list_files()
    file_options = {"All files": None}
    file_options.update({row["file_name"]: row["file_id"] for row in files})
    selected_file = st.selectbox("File filter", list(file_options.keys()))
    mode = st.segmented_control("Search mode", ["Hybrid", "Semantic", "Keyword"], default="Hybrid")
    top_k = st.slider("Results", min_value=1, max_value=20, value=config.top_k_search_results)

    if st.button("Search", type="primary") and query.strip():
        results = service_search().search(query, mode=mode, top_k=top_k, file_id=file_options[selected_file])
        if not results:
            st.info("No matching chunks found.")
            return
        for result in results:
            with st.container(border=True):
                st.markdown(f"**{result.file_name}**")
                meta = f"Match: {result.match_type} | Score: {result.score:.4f} | Source: {result.source_type}"
                if result.page_start:
                    meta += f" | Page {result.page_start}"
                    if result.page_end and result.page_end != result.page_start:
                        meta += f"-{result.page_end}"
                st.caption(meta)
                st.write(result.chunk_text)


def page_cost_dashboard() -> None:
    st.header("Cost Dashboard")
    summary = db.usage_summary()
    files = db.list_files()
    cols = st.columns(5)
    cols[0].metric("Total API cost", f"${summary['total']:.6f}")
    cols[1].metric("Files", len(files))
    cols[2].metric("Vision required", summary["vision_required"])
    cols[3].metric("Vision avoided", summary["vision_avoided"])
    cols[4].metric("Query embeddings", summary["query_embeddings"])

    st.subheader("By Operation")
    st.dataframe(rows_as_dicts(summary["by_operation"]), use_container_width=True, hide_index=True)
    st.subheader("By File")
    st.dataframe(rows_as_dicts(summary["by_file"]), use_container_width=True, hide_index=True)
    st.subheader("By Model")
    st.dataframe(rows_as_dicts(summary["by_model"]), use_container_width=True, hide_index=True)
    st.caption(f"Local zero-cost operations logged: {summary['local_operations']}")


def page_settings() -> None:
    st.header("Settings")
    st.json(
        {
            "llm_model": config.llm_model,
            "vision_model": config.vision_model,
            "embedding_model": config.embedding_model,
            "whisper_model": config.whisper_model,
            "chunk_size_tokens": config.chunk_size_tokens,
            "chunk_overlap_tokens": config.chunk_overlap_tokens,
            "native_text_min_chars": config.native_text_min_chars,
            "text_area_threshold": config.text_area_threshold,
            "large_image_area_threshold": config.large_image_area_threshold,
            "scanned_page_image_area_threshold": config.scanned_page_image_area_threshold,
            "openai_configured": bool(config.openai_api_key),
            "openrouter_configured": bool(config.openrouter_api_key),
            "openrouter_model": config.openrouter_model,
        }
    )
    st.code("OPENAI_API_KEY=\nOPENROUTER_API_KEY=", language="text")


PAGES = {
    "Upload & Process": page_upload,
    "Processed Files": page_processed_files,
    "Search": page_search,
    "Cost Dashboard": page_cost_dashboard,
    "Settings": page_settings,
}

with st.sidebar:
    st.title(config.app_name)
    page = st.radio("Navigation", list(PAGES.keys()))

PAGES[page]()
