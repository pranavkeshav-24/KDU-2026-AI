"""
ChromaDB vector store management.
Handles collection creation, document upsert (with deduplication), and similarity search.
"""
import os
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings

COLLECTION_NAME = 'rag_documents'


def get_vectorstore(embedding_fn) -> Chroma:
    """
    Return a persistent ChromaDB Chroma instance backed by the local filesystem.

    Uses cosine similarity as the HNSW distance metric.
    ID-based upsert in add_documents prevents re-ingesting the same chunks.

    Args:
        embedding_fn: HuggingFaceEmbeddings (or compatible) embedding function.

    Returns:
        Chroma vectorstore connected to the persistent collection.
    """
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_metadata={'hnsw:space': 'cosine'},  # explicit cosine metric
    )


def upsert_chunks(vectorstore: Chroma, chunks: List[Document]) -> int:
    """
    Add chunks to ChromaDB, deduplicating by source + chunk_id.

    Using deterministic IDs means re-ingesting the same document
    will overwrite existing entries rather than creating duplicates.

    Args:
        vectorstore: Target Chroma collection.
        chunks: List of Document chunks to upsert.

    Returns:
        Number of chunks upserted.
    """
    if not chunks:
        return 0

    ids = []
    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        chunk_id = chunk.metadata.get('chunk_id', 0)
        # Sanitize source to be a valid ID component
        safe_source = source.replace('/', '_').replace('\\', '_').replace(':', '')
        ids.append(f"{safe_source}__{chunk_id}")

    vectorstore.add_documents(documents=chunks, ids=ids)
    return len(chunks)


def similarity_search(
    vectorstore: Chroma,
    query: str,
    k: int = 20,
) -> List[Tuple[Document, float]]:
    """
    Run cosine similarity search against the ChromaDB collection.

    Args:
        vectorstore: Target Chroma collection.
        query: Raw query string (will be embedded internally).
        k: Number of results to return.

    Returns:
        List of (Document, similarity_score) tuples, sorted by score descending.
    """
    return vectorstore.similarity_search_with_score(query, k=k)


def get_collection_count(vectorstore: Chroma) -> int:
    """Return the number of documents in the collection."""
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0
