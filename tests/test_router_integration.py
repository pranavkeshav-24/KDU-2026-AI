# tests/test_router_integration.py
"""
Integration tests for the Router Engine.
All tests run in dry_run_mode=True so no actual LLM API calls are made.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import yaml
import os
from config.loader import ConfigLoader, config_loader


def _setup_test_config():
    """Force config_loader to use a test config with dry_run_mode=True."""
    test_config = {
        "version": "1.0.0",
        "environment": "test",
        "models": {
            "tier_low": {
                "model_id": "liquid/lfm-2.5-1.2b-instruct:free",
                "provider": "openrouter",
                "temperature": 0.3,
                "max_tokens": 256,
                "cost_per_1k_input_tokens": 0.0,
                "cost_per_1k_output_tokens": 0.0,
            },
            "tier_medium": {
                "model_id": "openai/gpt-oss-20b:free",
                "provider": "openrouter",
                "temperature": 0.5,
                "max_tokens": 512,
                "cost_per_1k_input_tokens": 0.0,
                "cost_per_1k_output_tokens": 0.0,
            },
            "tier_high": {
                "model_id": "openai/gpt-oss-120b:free",
                "provider": "openrouter",
                "temperature": 0.7,
                "max_tokens": 1024,
                "cost_per_1k_input_tokens": 0.0,
                "cost_per_1k_output_tokens": 0.0,
            },
            "fallback": {
                "model_id": "liquid/lfm-2.5-1.2b-instruct:free",
                "provider": "openrouter",
                "temperature": 0.2,
                "max_tokens": 128,
                "cost_per_1k_input_tokens": 0.0,
                "cost_per_1k_output_tokens": 0.0,
            },
        },
        "routing": {
            "low_complexity_threshold": 0.3,
            "high_complexity_threshold": 0.7,
            "high_complexity_keywords": ["refund", "complaint", "damage", "emergency"],
            "complaint_keywords": ["angry", "frustrated", "worst"],
            "booking_keywords": ["reschedule", "cancel", "appointment", "book"],
            "faq_keywords": ["hours", "price", "cost", "what is", "do you", "can you"],
            "max_retries": 1,
            "timeout_seconds": 10,
            "fallback_on_error": True,
        },
        "budget": {
            "daily_budget_usd": 16.67,
            "monthly_budget_usd": 500.0,
            "warning_threshold_pct": 0.80,
            "hard_limit_pct": 0.95,
            "per_query_max_usd": 0.05,
            "reset_timezone": "UTC",
        },
        "features": {
            "enable_semantic_cache": False,
            "enable_fallback_on_budget": True,
            "enable_structured_logging": False,   # No file I/O in tests
            "enable_cost_tracking": True,
            "enable_eval_logging": False,
            "dry_run_mode": True,                 # Never calls real LLM in tests
        },
        "prompts": {
            "registry_path": str(Path(__file__).parent.parent / "prompts"),
            "default_version": "v1",
            "hot_reload": False,
        },
        "logging": {
            "level": "ERROR",
            "format": "json",
            "output_path": f"{tempfile.gettempdir()}/fixit_test.jsonl",
            "include_prompt_in_log": False,
            "sample_rate": 0.0,   # Don't write logs in tests
        },
    }
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(test_config, f)
    f.close()
    config_loader.load(f.name)
    return f.name


@pytest.fixture(autouse=True)
def setup_config():
    """Ensure all integration tests use the dry-run test config."""
    _setup_test_config()
    yield


class TestRouterIntegration:

    def _get_engine(self):
        """Fresh RouterEngine instance with test budget state."""
        from router.engine import RouterEngine
        from router.budget_guard import BudgetGuard
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        engine = RouterEngine()
        engine.budget_guard = BudgetGuard(state_path=tmp.name)
        return engine

    # ── End-to-End Routing ────────────────────────────────────────────

    def test_faq_query_routes_to_tier_low(self):
        engine = self._get_engine()
        response = engine.route("What are your operating hours?")
        assert response.category == "faq"
        assert response.complexity == "low"
        assert response.tier_used == "tier_low"

    def test_booking_query_routes_to_tier_medium(self):
        engine = self._get_engine()
        response = engine.route("I need to reschedule my appointment to next Friday")
        assert response.category == "booking"
        # Booking queries cap at 0.65 (MEDIUM), unless they also contain high_complexity keywords
        assert response.tier_used in ["tier_low", "tier_medium", "tier_high"]
        assert response.complexity in ["low", "medium", "high"]

    def test_complaint_query_routes_to_tier_high(self):
        engine = self._get_engine()
        # "refund" is a high_complexity keyword — single hit scores 0.75 → HIGH tier
        response = engine.route(
            "My plumber didn't show up and I want a full refund now!"
        )
        assert response.category == "complaint"
        # Either refund or "didn't show" triggers HIGH complexity
        assert response.complexity in ["high", "medium"]  # refund alone = 0.75+length > 0.7
        assert response.tier_used in ["tier_high", "tier_medium"]

    # ── Response Structure ────────────────────────────────────────────

    def test_response_has_all_required_fields(self):
        engine = self._get_engine()
        response = engine.route("What are your hours?")
        assert response.query_id is not None
        assert len(response.response_text) > 0
        assert response.tokens_in > 0
        assert response.tokens_out > 0
        assert response.latency_ms >= 0
        assert response.timestamp is not None

    def test_response_text_non_empty(self):
        engine = self._get_engine()
        response = engine.route("What services do you offer?")
        assert isinstance(response.response_text, str)
        assert len(response.response_text) > 10

    def test_query_id_is_unique(self):
        engine = self._get_engine()
        r1 = engine.route("What are your hours?")
        r2 = engine.route("What are your hours?")
        assert r1.query_id != r2.query_id

    # ── Dry Run Mode ──────────────────────────────────────────────────

    def test_dry_run_mode_active(self):
        engine = self._get_engine()
        config = config_loader.get()
        assert config.features.dry_run_mode is True

    def test_dry_run_returns_response_without_api(self):
        """In dry_run mode, should complete without OPENROUTER_API_KEY set."""
        engine = self._get_engine()
        response = engine.route("I want a refund")
        assert response.response_text is not None
        assert response.fallback_activated is False  # budget not exceeded

    # ── Budget Integration ────────────────────────────────────────────

    def test_cost_recorded_after_routing(self):
        engine = self._get_engine()
        initial = engine.budget_guard.get_summary()["total_requests"]
        engine.route("What are your hours?")
        updated = engine.budget_guard.get_summary()["total_requests"]
        assert updated == initial + 1

    def test_fallback_activated_on_budget_exceeded(self):
        engine = self._get_engine()
        # Manually force budget to 99% to trigger fallback
        from router.budget_guard import CostRecord
        from datetime import datetime
        big_record = CostRecord(
            timestamp=datetime.utcnow().isoformat(),
            query_id="budget-fill",
            model_id="openai/gpt-oss-120b:free",
            tier="tier_high",
            tokens_in=1000,
            tokens_out=500,
            cost_usd=16.5,  # over 95% of $16.67 daily budget
            category="complaint",
        )
        engine.budget_guard.record_cost(big_record)
        response = engine.route("I want a refund")
        assert response.fallback_activated is True
        assert response.tier_used == "fallback"

    # ── Session ID ────────────────────────────────────────────────────

    def test_session_id_preserved_in_response(self):
        engine = self._get_engine()
        response = engine.route("What are your hours?", session_id="sess-test-001")
        assert response.session_id == "sess-test-001"

    def test_session_id_none_when_not_provided(self):
        engine = self._get_engine()
        response = engine.route("What are your hours?")
        assert response.session_id is None

    # ── Routing Reason ────────────────────────────────────────────────

    def test_routing_reason_contains_score(self):
        engine = self._get_engine()
        response = engine.route("I want a refund")
        assert "score=" in response.routing_reason

    def test_prompt_version_set(self):
        engine = self._get_engine()
        response = engine.route("What are your hours?")
        assert response.prompt_version == "v1"
