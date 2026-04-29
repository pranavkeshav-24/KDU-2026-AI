from __future__ import annotations

import hashlib
import math

from src.config import AppConfig, config
from src.llm.openai_client import OpenAIClient
from src.storage.models import Chunk
from src.utils.file_utils import new_id


def local_embedding(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class Embedder:
    def __init__(self, llm_client: OpenAIClient, app_config: AppConfig = config):
        self.llm_client = llm_client
        self.config = app_config

    def embed_chunks(self, chunks: list[Chunk]) -> tuple[list[tuple[str, str, str, list[float], str]], dict[str, int], str]:
        texts = [chunk.chunk_text for chunk in chunks]
        embeddings, usage, provider = self.llm_client.embed_texts(texts)
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vector_id = new_id("vec")
            chunk.vector_id = vector_id
            model = self.config.embedding_model if provider == "openai" else "local-hash-embedding"
            vectors.append((vector_id, chunk.chunk_id, chunk.file_id, embedding, model))
        return vectors, usage, provider

    def embed_query(self, query: str) -> tuple[list[float], dict[str, int], str]:
        embeddings, usage, provider = self.llm_client.embed_texts([query])
        return embeddings[0], usage, provider

