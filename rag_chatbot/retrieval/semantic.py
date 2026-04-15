"""
Dense semantic retrieval from ChromaDB.
Thin wrapper that exposes the similarity search with consistent return types.
"""
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings


def dense_retrieve(
    vectorstore: Chroma,
    query: str,
    k: int = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k documents using dense (cosine) similarity search.

    Args:
        vectorstore: ChromaDB Chroma collection.
        query: User query string.
        k: Number of results (defaults to settings.TOP_K_RETRIEVE).

    Returns:
        List of (Document, score) tuples sorted by similarity descending.
    """
    k = k or settings.TOP_K_RETRIEVE
    return vectorstore.similarity_search_with_score(query, k=k)
