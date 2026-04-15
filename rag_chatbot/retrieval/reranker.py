"""
Cross-encoder reranker for final top-k selection.

Cross-encoders process query and document together, allowing full token-level
attention between query and passage. This gives much higher relevance precision
than bi-encoders (used in retrieval), at the cost of being slower.

Used as a second-stage filter: retrieve broadly with bi-encoders, then
rerank the shortlist precisely with the cross-encoder.
"""
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import settings


class CrossEncoderReranker:
    """
    Thin wrapper around sentence_transformers CrossEncoder.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    Trained on MS MARCO passage ranking — optimised for query-passage relevance.
    """

    def __init__(self, model_name: str = None) -> None:
        """
        Load the cross-encoder model.

        Args:
            model_name: HuggingFace model identifier. Defaults to settings value.
        """
        model_name = model_name or settings.RERANKER_MODEL
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = None,
    ) -> List[Document]:
        """
        Rerank documents by cross-encoder relevance score.

        Attaches rerank_score to each document's metadata so the UI
        can display confidence levels alongside source citations.

        Args:
            query: User query string.
            docs: Candidate documents from RRF fusion.
            top_k: Number of documents to return (defaults to settings.TOP_K_RERANK).

        Returns:
            List of top_k Documents sorted by cross-encoder score descending.
        """
        if not docs:
            return []

        top_k = top_k or settings.TOP_K_RERANK

        # Build (query, passage) pairs for batch prediction
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        # Sort by score descending
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in ranked[:top_k]:
            doc.metadata['rerank_score'] = round(float(score), 4)
            result.append(doc)

        return result
