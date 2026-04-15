"""
Reciprocal Rank Fusion (RRF) for combining dense and sparse retrieval results.

RRF is rank-based (not score-based), making it robust against the different
numerical scales of cosine similarity and BM25 scores.
k=60 is the empirically validated smoothing constant from the original RRF paper.
"""
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from config import settings

RRF_K: int = 60   # do not change without benchmarking


def _content_key(doc: Document) -> str:
    """
    Generate a deduplication key from document content.
    Uses first 200 characters to handle minor whitespace differences.
    """
    return doc.page_content[:200].strip()


def reciprocal_rank_fusion(
    semantic_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    top_n: int = None,
) -> List[Document]:
    """
    Combine dense and sparse retrieval results using Reciprocal Rank Fusion.

    Each result is scored by: 1 / (RRF_K + rank + 1)
    Results appearing in both lists accumulate scores from both,
    naturally elevating consistently highly-ranked items.

    Args:
        semantic_results: List of (Document, score) from ChromaDB.
        bm25_results: List of (Document, score) from BM25.
        top_n: Number of fused results to return (defaults to TOP_K_RETRIEVE).

    Returns:
        List of deduplicated Documents sorted by fused RRF score descending.
    """
    top_n = top_n or settings.TOP_K_RETRIEVE

    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}

    # Score semantic (dense) results
    for rank, (doc, _) in enumerate(semantic_results):
        key = _content_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        docs[key] = doc

    # Score BM25 (sparse) results
    for rank, (doc, _) in enumerate(bm25_results):
        key = _content_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        docs[key] = doc

    # Sort by fused score descending and return top_n
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    result = []
    for key, rrf_score in ranked[:top_n]:
        doc = docs[key]
        doc.metadata['rrf_score'] = round(rrf_score, 6)
        result.append(doc)

    return result
