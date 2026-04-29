from pathlib import Path

from src.config import config
from src.llm.openai_client import OpenAIClient
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.vector_store import VectorStore
from src.storage.db import Database
from src.storage.models import Chunk


def test_hybrid_search_returns_keyword_result(tmp_path: Path):
    db = Database(tmp_path / "app.db")
    db.create_file("file_1", "demo.pdf", "pdf", 100, "demo.pdf", "balanced")
    chunks = [
        Chunk("chunk_1", "file_1", 0, "The accessibility suite tracks policy ABC123.", page_start=1, page_end=1),
        Chunk("chunk_2", "file_1", 1, "A different paragraph about summaries.", page_start=2, page_end=2),
    ]
    db.replace_chunks("file_1", chunks)
    llm = OpenAIClient(config)
    embedder = Embedder(llm, config)
    vectors, _usage, _provider = embedder.embed_chunks(chunks)
    VectorStore(db).upsert(vectors)

    search = HybridSearch(db, embedder, VectorStore(db), KeywordIndex(db), config)
    results = search.search("ABC123", mode="Hybrid", top_k=1)

    assert results
    assert results[0].chunk_id == "chunk_1"

