"""
Unit tests for the ingestion pipeline: loaders and semantic chunker.
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.documents import Document
from ingestion.loaders import load_url, save_uploaded_file
from ingestion.chunker import build_semantic_chunker, chunk_documents


class TestChunker:
    """Tests for semantic chunking logic."""

    @pytest.fixture(scope='class')
    def chunker(self):
        return build_semantic_chunker()

    def test_chunker_builds(self, chunker):
        """SemanticChunker should initialise without errors."""
        assert chunker is not None

    def test_chunk_documents_basic(self, chunker):
        """chunk_documents should split a multi-sentence Document."""
        docs = [
            Document(
                page_content=(
                    "Machine learning is a branch of artificial intelligence. "
                    "It allows computers to learn from data without being explicitly programmed. "
                    "Deep learning is a subset of machine learning. "
                    "Neural networks form the backbone of deep learning approaches. "
                    "Retrieval-augmented generation combines retrieval with generation models. "
                    "RAG systems retrieve relevant context before generating answers."
                ),
                metadata={'source': 'test', 'doc_title': 'test'},
            )
        ]
        chunks = chunk_documents(docs, chunker)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert 'chunk_id' in chunk.metadata
            assert 'char_count' in chunk.metadata
            assert 'word_count' in chunk.metadata
            assert chunk.metadata['char_count'] > 0

    def test_chunk_metadata_populated(self, chunker):
        """Each chunk should have correctly computed metadata."""
        docs = [
            Document(
                page_content="Vector databases store embeddings. They enable similarity search.",
                metadata={'source': 'meta_test'},
            )
        ]
        chunks = chunk_documents(docs, chunker)
        for chunk in chunks:
            assert chunk.metadata['char_count'] == len(chunk.page_content)
            assert chunk.metadata['word_count'] == len(chunk.page_content.split())


class TestSaveUpload:
    """Tests for file save utility."""

    def test_save_uploaded_file(self, tmp_path):
        """save_uploaded_file should write file content to disk."""
        class FakeUpload:
            name = 'test.pdf'
            def read(self):
                return b'%PDF-1.4 fake content'

        path = save_uploaded_file(FakeUpload(), upload_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith('test.pdf')
        with open(path, 'rb') as f:
            assert f.read() == b'%PDF-1.4 fake content'
