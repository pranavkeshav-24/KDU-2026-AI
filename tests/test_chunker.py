from src.retrieval.chunker import Chunker
from src.storage.models import PageContent, UnifiedDocument


def test_chunker_preserves_page_metadata():
    doc = UnifiedDocument(
        file_id="file_1",
        file_name="demo.pdf",
        file_type="pdf",
        processing_mode="balanced",
        full_text="",
        pages=[
            PageContent(page_number=1, text="alpha beta gamma"),
            PageContent(page_number=2, text="delta epsilon zeta"),
        ],
    )
    chunks = Chunker().chunk_document(doc)
    assert chunks
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert "Page 1" in chunks[0].chunk_text

