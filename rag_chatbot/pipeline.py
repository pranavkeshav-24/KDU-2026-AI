"""
End-to-end RAG pipeline orchestrator.
"""
import logging
import time
from typing import Dict, List, Optional

from langchain_core.documents import Document
from openai import RateLimitError

from config import settings
from embeddings.encoder import get_embedding_function
from generation.llm import build_rag_chain, get_llm
from generation.prompt import RAG_PROMPT, format_context
from ingestion.chunker import build_semantic_chunker, chunk_documents
from ingestion.loaders import load_pdf, load_url
from retrieval.bm25_search import BM25Index
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from vectorstore.chroma_store import get_collection_count, get_vectorstore, upsert_chunks

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Full RAG pipeline orchestrator."""

    def __init__(self) -> None:
        """Initialize all pipeline components once."""
        logger.info("Initializing RAG Pipeline...")

        # Embedding model (shared across ChromaDB and SemanticChunker)
        self.embedding_fn = get_embedding_function()

        # Vector store (persistent ChromaDB)
        self.vectorstore = get_vectorstore(self.embedding_fn)

        # Semantic chunker (uses same embedding model)
        self.chunker = build_semantic_chunker()

        # Cross-encoder reranker
        self.reranker = CrossEncoderReranker()

        # LLM and primary chain
        self.llm = get_llm()
        self.chain = build_rag_chain(self.llm, RAG_PROMPT)

        # Optional fallback chain
        self.fallback_chain = None
        if settings.LLM_FALLBACK_MODEL and settings.LLM_FALLBACK_MODEL != settings.LLM_MODEL:
            fallback_llm = get_llm(fallback=True)
            self.fallback_chain = build_rag_chain(fallback_llm, RAG_PROMPT)

        # BM25 index built/rebuilt after each ingestion
        self.bm25_index: Optional[BM25Index] = None

        # Track all chunks across sessions for BM25 (in-memory)
        self._all_chunks: List[Document] = []

        logger.info("RAG Pipeline ready.")

    def _invoke_with_retry(self, chain, payload: Dict, model_name: str) -> str:
        """Invoke a chain with exponential backoff on 429s."""
        retry_count = max(0, settings.LLM_MAX_RETRIES)

        for attempt in range(retry_count + 1):
            try:
                return chain.invoke(payload)
            except RateLimitError:
                if attempt >= retry_count:
                    raise

                delay = settings.LLM_RETRY_BASE_SECONDS * (2 ** attempt)
                logger.warning(
                    "Rate limited on model %s (attempt %s/%s). Retrying in %.2fs.",
                    model_name,
                    attempt + 1,
                    retry_count + 1,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError("Unexpected retry loop termination.")

    def ingest(self, source: str, source_type: str = 'pdf') -> Dict:
        """Ingest a PDF path or URL into ChromaDB and BM25."""
        logger.info("Ingesting %s: %s", source_type, source)

        # 1. Load
        if source_type == 'pdf':
            docs = load_pdf(source)
        else:
            docs = load_url(source)

        if not docs:
            raise ValueError(f"No content loaded from: {source}")

        # 2. Chunk
        chunks = chunk_documents(docs, self.chunker)

        # 3. Upsert into ChromaDB
        num_upserted = upsert_chunks(self.vectorstore, chunks)

        # 4. Accumulate chunks and rebuild BM25 index
        existing_keys = {c.page_content[:200] for c in self._all_chunks}
        new_chunks = [c for c in chunks if c.page_content[:200] not in existing_keys]
        self._all_chunks.extend(new_chunks)
        self.bm25_index = BM25Index(self._all_chunks)

        logger.info("Ingested %s chunks from %s", num_upserted, source)

        return {
            'num_chunks': num_upserted,
            'num_docs': len(docs),
            'source': source,
        }

    def query(self, question: str) -> Dict:
        """Run retrieval, reranking, and answer generation."""
        if self.bm25_index is None or len(self._all_chunks) == 0:
            raise RuntimeError(
                "No documents ingested. Please upload a PDF or enter a URL first."
            )

        # 1. Dense retrieval
        semantic_results = self.vectorstore.similarity_search_with_score(
            question, k=settings.TOP_K_RETRIEVE
        )

        # 2. Sparse retrieval
        bm25_results = self.bm25_index.search(question, k=settings.TOP_K_RETRIEVE)

        # 3. RRF fusion
        fused = reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            top_n=settings.TOP_K_RETRIEVE,
        )

        if not fused:
            fused = [doc for doc, _ in semantic_results[: settings.TOP_K_RERANK]]

        # 4. Cross-encoder reranking
        top_chunks = self.reranker.rerank(question, fused, top_k=settings.TOP_K_RERANK)

        # 5. Generate
        context = format_context(top_chunks)
        payload = {'context': context, 'question': question}

        try:
            answer = self._invoke_with_retry(self.chain, payload, settings.LLM_MODEL)
        except RateLimitError as primary_error:
            if self.fallback_chain is None:
                raise RuntimeError(
                    "The model provider is currently rate-limited. "
                    "Try again shortly or switch to a different paid model."
                ) from primary_error

            logger.warning(
                "Primary model %s was rate-limited. Falling back to %s.",
                settings.LLM_MODEL,
                settings.LLM_FALLBACK_MODEL,
            )

            try:
                answer = self._invoke_with_retry(
                    self.fallback_chain,
                    payload,
                    settings.LLM_FALLBACK_MODEL,
                )
            except RateLimitError as fallback_error:
                raise RuntimeError(
                    "Both primary and fallback providers are rate-limited right now. "
                    "Please retry in a moment."
                ) from fallback_error

        return {
            'answer': answer,
            'sources': top_chunks,
        }

    def get_stats(self) -> Dict:
        """Return pipeline statistics for the UI."""
        return {
            'total_chunks_bm25': len(self._all_chunks),
            'chroma_count': get_collection_count(self.vectorstore),
            'bm25_ready': self.bm25_index is not None,
        }
