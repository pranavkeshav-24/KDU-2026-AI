"""
Embedding model wrapper using BAAI/bge-small-en-v1.5 via sentence-transformers.
This is the shared embedding function used by both ChromaDB and SemanticChunker.
"""
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Create a HuggingFaceEmbeddings instance for BAAI/bge-small-en-v1.5.

    Model characteristics:
      - 33M parameters, 384-dimensional embeddings
      - Trained specifically for passage retrieval (MTEB benchmark)
      - normalize_embeddings=True is REQUIRED for cosine similarity correctness
      - Runs entirely on CPU — no GPU required

    Returns:
        HuggingFaceEmbeddings ready for use with ChromaDB and SemanticChunker.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'normalize_embeddings': True,   # REQUIRED for cosine similarity
            'batch_size': 32,               # tune to available RAM
        },
    )
