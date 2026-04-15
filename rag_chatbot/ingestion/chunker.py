"""
Semantic chunking using LangChain's SemanticChunker.
Splits documents at natural topic boundaries (embedding similarity drops)
rather than fixed character/token counts.
"""
from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


def build_semantic_chunker() -> SemanticChunker:
    """
    Build a SemanticChunker backed by BAAI/bge-small-en-v1.5.

    Uses percentile-based breakpoint detection: a chunk boundary is inserted
    wherever the similarity drop between consecutive sentence groups falls
    in the top 5% (95th percentile) of all drops in the document.

    Returns:
        Configured SemanticChunker instance.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=settings.BREAKPOINT_THRESHOLD_TYPE,
        breakpoint_threshold_amount=settings.BREAKPOINT_THRESHOLD_AMOUNT,
        add_start_index=True,  # preserves char offset in metadata
    )


def chunk_documents(
    docs: List[Document],
    chunker: SemanticChunker,
) -> List[Document]:
    """
    Split documents into semantically coherent chunks.

    Enriches each chunk with:
      - chunk_id: sequential index across all chunks
      - char_count: number of characters in the chunk
      - word_count: number of whitespace-separated tokens

    Args:
        docs: List of raw Document objects from loaders.
        chunker: Configured SemanticChunker instance.

    Returns:
        List[Document] where each item is a single coherent chunk.
    """
    chunks = chunker.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id']   = i
        chunk.metadata['char_count'] = len(chunk.page_content)
        chunk.metadata['word_count'] = len(chunk.page_content.split())

    return chunks
