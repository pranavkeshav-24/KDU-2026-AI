# tests/test_prompt_manager.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import os
import yaml
from prompts.manager import PromptManager


def _make_registry() -> str:
    """Creates a minimal temp prompt registry for testing."""
    tmpdir = tempfile.mkdtemp()
    categories = {
        "faq": {
            "metadata": {
                "name": "faq_handler",
                "category": "faq",
                "version": "v1",
                "created_at": "2026-04-19",
                "created_by": "test",
                "status": "active",
                "model_family": "any",
                "avg_tokens_system": 100,
                "avg_tokens_output": 100,
                "eval_score": 4.0,
                "notes": "Test FAQ prompt",
            },
            "system_prompt": "You are a helpful FAQ assistant for FixIt home services.",
            "user_template": "Customer question: {{ query }}\nAnswer:",
            "few_shot_examples": [],
        },
        "complaint": {
            "metadata": {
                "name": "complaint_handler",
                "category": "complaint",
                "version": "v1",
                "created_at": "2026-04-19",
                "created_by": "test",
                "status": "active",
                "model_family": "any",
                "avg_tokens_system": 180,
                "avg_tokens_output": 320,
                "eval_score": 4.2,
                "notes": "Test complaint prompt",
            },
            "system_prompt": "You are FixIt's empathetic senior customer support specialist.",
            "user_template": "Customer: {{ query }}\nService: {{ service_type }}",
            "few_shot_examples": [
                {
                    "input": "My plumber never showed up.",
                    "output": "I'm so sorry to hear that — I'm processing a full refund immediately.",
                }
            ],
        },
        "fallback": {
            "metadata": {
                "name": "fallback_handler",
                "category": "fallback",
                "version": "v1",
                "created_at": "2026-04-19",
                "created_by": "test",
                "status": "active",
                "model_family": "any",
                "avg_tokens_system": 50,
                "avg_tokens_output": 80,
                "eval_score": 3.5,
                "notes": "Generic fallback",
            },
            "system_prompt": "You are a helpful assistant for FixIt home services.",
            "user_template": "{{ query }}",
            "few_shot_examples": [],
        },
    }
    for cat, data in categories.items():
        cat_dir = os.path.join(tmpdir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        with open(os.path.join(cat_dir, "v1.yaml"), "w") as f:
            yaml.dump(data, f)
    return tmpdir


class TestPromptManager:

    def test_renders_faq_prompt(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        prompt = manager.render("faq", "v1", {
            "query": "What are your hours?",
            "service_type": "home_services"
        })
        assert "What are your hours?" in prompt
        assert "[SYSTEM]" in prompt
        assert "[USER]" in prompt

    def test_renders_complaint_prompt(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        prompt = manager.render("complaint", "v1", {
            "query": "I want a refund",
            "service_type": "plumbing"
        })
        assert "refund" in prompt
        assert "plumbing" in prompt

    def test_few_shot_examples_injected(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        prompt = manager.render("complaint", "v1", {
            "query": "test", "service_type": "test"
        })
        assert "## Examples" in prompt
        assert "My plumber never showed up" in prompt

    def test_fallback_when_version_missing(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        # Request non-existent version — should fallback to v1
        prompt = manager.render("faq", "v99", {
            "query": "test", "service_type": "test"
        })
        assert prompt is not None
        assert len(prompt) > 0

    def test_fallback_when_category_missing(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        # Unknown category — should use fallback prompt
        prompt = manager.render("unknown_category", "v1", {
            "query": "test", "service_type": "test"
        })
        assert prompt is not None
        assert len(prompt) > 0

    def test_template_variable_substitution(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        prompt = manager.render("faq", "v1", {
            "query": "UNIQUE_TEST_QUERY_XYZ",
            "service_type": "cleaning"
        })
        assert "UNIQUE_TEST_QUERY_XYZ" in prompt

    def test_list_versions(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        versions = manager.list_versions("faq")
        assert "v1" in versions

    def test_list_categories(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        categories = manager.list_categories()
        assert "faq" in categories
        assert "complaint" in categories
        assert "fallback" in categories

    def test_list_versions_missing_category_returns_empty(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        versions = manager.list_versions("nonexistent_category")
        assert versions == []

    def test_get_metadata(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        metadata = manager.get_metadata("faq", "v1")
        assert metadata["version"] == "v1"
        assert metadata["eval_score"] == 4.0
        assert metadata["status"] == "active"

    def test_cache_returns_same_object(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        p1 = manager.render("faq", "v1", {"query": "a", "service_type": "b"})
        p2 = manager.render("faq", "v1", {"query": "a", "service_type": "b"})
        assert p1 == p2

    def test_cache_invalidation(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        manager.render("faq", "v1", {"query": "a", "service_type": "b"})
        manager.invalidate_cache("faq")
        # After invalidation, should reload from file (not crash)
        p = manager.render("faq", "v1", {"query": "a", "service_type": "b"})
        assert p is not None

    def test_full_cache_invalidation(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        manager.render("faq", "v1", {"query": "a", "service_type": "b"})
        manager.render("complaint", "v1", {"query": "b", "service_type": "c"})
        manager.invalidate_cache()  # invalidate all
        p = manager.render("faq", "v1", {"query": "a", "service_type": "b"})
        assert p is not None

    def test_rendered_prompt_has_system_and_user_sections(self):
        registry = _make_registry()
        manager = PromptManager(registry_path=registry)
        prompt = manager.render("fallback", "v1", {"query": "help me"})
        assert "[SYSTEM]" in prompt
        assert "[USER]" in prompt
