"""
Unit tests for the generation pipeline: prompt formatting and LLM chain structure.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.documents import Document
from generation.prompt import format_context, RAG_PROMPT


def make_chunk(content: str, chunk_id: int, source: str = 'test.pdf', page: int = None) -> Document:
    meta = {'chunk_id': chunk_id, 'source': source}
    if page is not None:
        meta['page'] = page
    return Document(page_content=content, metadata=meta)


class TestFormatContext:
    def test_basic_formatting(self):
        chunks = [
            make_chunk('Embeddings map text to vector space.', 0),
            make_chunk('BM25 ranks by term frequency.', 1),
        ]
        ctx = format_context(chunks)
        assert '[Chunk 0]' in ctx
        assert '[Chunk 1]' in ctx
        assert 'test.pdf' in ctx
        assert '---' in ctx  # separator between chunks

    def test_page_number_included(self):
        chunk = make_chunk('Content from page 5.', 3, page=5)
        ctx = format_context([chunk])
        assert 'Page 5' in ctx

    def test_no_page_number_when_absent(self):
        chunk = make_chunk('Content without page.', 0)
        ctx = format_context([chunk])
        assert 'Page' not in ctx

    def test_empty_chunks(self):
        ctx = format_context([])
        assert ctx == ''

    def test_single_chunk_no_separator(self):
        chunk = make_chunk('Single chunk.', 0)
        ctx = format_context([chunk])
        assert '---' not in ctx


class TestRAGPrompt:
    def test_prompt_has_required_variables(self):
        """Prompt template should require 'context' and 'question'."""
        variables = RAG_PROMPT.input_variables
        assert 'context' in variables
        assert 'question' in variables

    def test_prompt_renders(self):
        """Prompt should render without errors with valid inputs."""
        messages = RAG_PROMPT.format_messages(
            context='[Chunk 0] test content',
            question='What is RAG?',
        )
        full_text = ' '.join(m.content for m in messages)
        assert 'What is RAG?' in full_text
        assert 'context' in full_text.lower() or '[Chunk 0]' in full_text

    def test_prompt_contains_grounding_instruction(self):
        """Prompt should instruct the model to answer ONLY from context."""
        template_str = str(RAG_PROMPT)
        assert 'ONLY' in template_str or 'only' in template_str
