"""
BM25 sparse retrieval index.
Provides exact keyword matching as the complement to dense semantic search.
BM25Okapi is the standard probabilistic ranking function used in IR systems.
"""
import re
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import settings


def tokenize(text: str) -> List[str]:
    """
    Lowercase, strip punctuation, split on whitespace.

    Args:
        text: Raw text string.

    Returns:
        List of lowercase tokens with punctuation removed.
    """
    return re.sub(r'[^\w\s]', '', text.lower()).split()


class BM25Index:
    """
    In-memory BM25 index built from a list of Document chunks.

    The index is rebuilt after each ingestion call since BM25 is not
    incrementally updatable in rank_bm25.
    """

    def __init__(self, chunks: List[Document]) -> None:
        """
        Build the BM25 index from document chunks.

        Args:
            chunks: List of Document objects to index.
        """
        self.chunks = chunks
        self.corpus = [tokenize(c.page_content) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def search(
        self,
        query: str,
        k: int = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search the BM25 index for the top-k results matching the query.

        Only returns chunks with BM25 score > 0 (i.e., at least one query
        term appears in the chunk). This avoids polluting RRF with zero-score noise.

        Args:
            query: User query string.
            k: Number of results (defaults to settings.TOP_K_RETRIEVE).

        Returns:
            List of (Document, bm25_score) tuples sorted by score descending.
        """
        k = k or settings.TOP_K_RETRIEVE
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(self.chunks[i], float(score)) for i, score in top_k if score > 0]

    def __len__(self) -> int:
        return len(self.chunks)
