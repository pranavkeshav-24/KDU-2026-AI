"""
Unit tests for retrieval modules: BM25 and dense semantic retrieval.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.documents import Document
from retrieval.bm25_search import BM25Index, tokenize
from retrieval.hybrid import reciprocal_rank_fusion


def make_doc(content: str, chunk_id: int = 0) -> Document:
    return Document(
        page_content=content,
        metadata={'source': 'test', 'chunk_id': chunk_id},
    )


class TestTokenizer:
    def test_lowercase(self):
        assert tokenize('Hello World') == ['hello', 'world']

    def test_strips_punctuation(self):
        assert tokenize('hello, world!') == ['hello', 'world']

    def test_empty_string(self):
        assert tokenize('') == []

    def test_numbers_preserved(self):
        tokens = tokenize('bm25 top 20 results')
        assert '20' in tokens


class TestBM25Index:
    @pytest.fixture
    def index(self):
        chunks = [
            make_doc('vector databases store embeddings for similarity search', 0),
            make_doc('BM25 is a probabilistic ranking function for keyword search', 1),
            make_doc('cross encoders score query document pairs accurately', 2),
            make_doc('reciprocal rank fusion combines multiple retrieval lists', 3),
            make_doc('semantic chunking preserves topic boundaries in text', 4),
        ]
        return BM25Index(chunks)

    def test_len(self, index):
        assert len(index) == 5

    def test_search_returns_results(self, index):
        results = index.search('keyword search ranking', k=3)
        assert len(results) >= 1
        # BM25 result that matches 'keyword search' best should rank high
        top_doc, top_score = results[0]
        assert top_score > 0

    def test_search_relevance(self, index):
        results = index.search('vector embeddings similarity', k=5)
        contents = [doc.page_content for doc, _ in results]
        assert any('vector' in c or 'embeddings' in c for c in contents)

    def test_no_results_for_unrelated_query(self, index):
        # Query completely unrelated to corpus
        results = index.search('zzzzxxx does not exist qqqqqq', k=5)
        # All scores should be 0, so returns empty list
        assert all(score == 0 for _, score in results) or len(results) == 0

    def test_search_top_k_limit(self, index):
        results = index.search('search retrieval ranking fusion chunking', k=2)
        assert len(results) <= 2


class TestRRF:
    def test_basic_fusion(self):
        doc_a = make_doc('vector database content', 0)
        doc_b = make_doc('BM25 keyword ranking', 1)
        doc_c = make_doc('cross encoder reranking', 2)

        semantic = [(doc_a, 0.9), (doc_b, 0.7)]
        bm25 = [(doc_b, 15.0), (doc_c, 12.0)]

        fused = reciprocal_rank_fusion(semantic, bm25, top_n=3)
        assert len(fused) >= 2
        # doc_b appears in both lists — should score highest
        assert fused[0].page_content == doc_b.page_content

    def test_deduplication(self):
        doc = make_doc('same content appears twice in both lists', 0)
        semantic = [(doc, 0.95)]
        bm25 = [(doc, 20.0)]

        fused = reciprocal_rank_fusion(semantic, bm25, top_n=5)
        # Should only appear once
        assert len(fused) == 1

    def test_rrf_score_attached(self):
        doc = make_doc('test document for rrf score check', 0)
        fused = reciprocal_rank_fusion([(doc, 0.9)], [], top_n=5)
        assert 'rrf_score' in fused[0].metadata
        assert fused[0].metadata['rrf_score'] > 0

    def test_empty_inputs(self):
        fused = reciprocal_rank_fusion([], [], top_n=5)
        assert fused == []

    def test_top_n_limit(self):
        docs = [(make_doc(f'doc {i}', i), float(i)) for i in range(10)]
        fused = reciprocal_rank_fusion(docs, [], top_n=3)
        assert len(fused) <= 3
