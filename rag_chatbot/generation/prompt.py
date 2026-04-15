"""
Prompt templates for the RAG generation chain.
Designed to enforce grounded answers — model only answers from provided context.
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from typing import List

RAG_PROMPT = ChatPromptTemplate.from_template("""You are a precise and helpful teaching assistant.
Answer the question ONLY using the context provided below.
If the context does not contain sufficient information to answer the question,
respond with: "I don't have enough context to answer that accurately."
Do not make up information or rely on prior knowledge outside the context.

Context:
{context}

Question: {question}

Answer (be clear and concise; cite the source chunk IDs in brackets, e.g. [Chunk 3]):""")


def format_context(chunks: List[Document]) -> str:
    """
    Format a list of Document chunks into a numbered context block for the LLM.

    Each chunk is prefixed with its chunk_id and source for easy citation.

    Args:
        chunks: Top-k reranked Document chunks.

    Returns:
        Single string with all chunks formatted for LLM injection.
    """
    parts = []
    for chunk in chunks:
        cid = chunk.metadata.get('chunk_id', '?')
        src = chunk.metadata.get('source', 'unknown')
        page = chunk.metadata.get('page', '')
        page_str = f" | Page {page}" if page else ''
        header = f"[Chunk {cid}] (Source: {src}{page_str})"
        parts.append(f"{header}\n{chunk.page_content}")

    return '\n\n---\n\n'.join(parts)
