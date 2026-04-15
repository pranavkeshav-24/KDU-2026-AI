"""
Unit tests for hybrid search: specifically RRF edge cases and property-based tests.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.documents import Document
from retrieval.hybrid import reciprocal_rank_fusion, RRF_K


def make_doc(content: str, idx: int = 0) -> Document:
    return Document(page_content=content, metadata={'source': 'test', 'chunk_id': idx})


class TestRRFProperties:
    """Property-based tests for RRF correctness."""

    def test_rrf_k_value(self):
        """k=60 is the validated constant — do not change."""
        assert RRF_K == 60

    def test_higher_rank_gives_higher_score(self):
        """A doc at rank 0 should always score higher than rank 5."""
        doc_top = make_doc('top ranked document', 0)
        doc_low = make_doc('lower ranked document', 1)

        semantic = [(doc_top, 0.95), make_doc('middle', 2), make_doc('middle2', 3), make_doc('middle3', 4), make_doc('middle4', 5), (doc_low, 0.5)]
        semantic_proper = [(doc_top, 0.95), (make_doc('a',2), 0.8), (make_doc('b',3), 0.7), (make_doc('c',4), 0.6), (make_doc('d',5), 0.55), (doc_low, 0.5)]

        fused = reciprocal_rank_fusion(semantic_proper, [], top_n=10)
        contents = [d.page_content for d in fused]
        assert contents.index('top ranked document') < contents.index('lower ranked document')

    def test_dual_list_score_higher_than_single(self):
        """A doc in both lists should outscore a doc only in one."""
        shared = make_doc('appears in both dense and sparse', 0)
        dense_only = make_doc('only in dense results', 1)
        sparse_only = make_doc('only in bm25 results', 2)

        semantic = [(shared, 0.9), (dense_only, 0.95)]
        bm25 = [(shared, 20.0), (sparse_only, 18.0)]

        fused = reciprocal_rank_fusion(semantic, bm25, top_n=3)
        # shared appears in both — should win even if not top in either
        assert fused[0].page_content == 'appears in both dense and sparse'

    def test_output_length_bounded_by_top_n(self):
        docs_s = [(make_doc(f'sem {i}', i), float(10 - i)) for i in range(8)]
        docs_b = [(make_doc(f'bm {i}', i + 100), float(8 - i)) for i in range(8)]
        fused = reciprocal_rank_fusion(docs_s, docs_b, top_n=5)
        assert len(fused) <= 5

    def test_all_docs_have_rrf_score(self):
        docs = [(make_doc(f'doc {i}', i), 1.0) for i in range(3)]
        fused = reciprocal_rank_fusion(docs, [], top_n=5)
        for d in fused:
            assert 'rrf_score' in d.metadata
            assert isinstance(d.metadata['rrf_score'], float)
