# tests/test_config_loader.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import os
import yaml
import tempfile
from config.loader import ConfigLoader


def _write_temp_config(overrides: dict = None) -> str:
    """Helper: write a minimal valid config to a temp file."""
    base_config = {
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
            "high_complexity_keywords": ["refund", "complaint"],
            "complaint_keywords": ["angry", "frustrated"],
            "booking_keywords": ["reschedule", "book"],
            "faq_keywords": ["hours", "price"],
            "max_retries": 2,
            "timeout_seconds": 30,
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
            "enable_structured_logging": False,
            "enable_cost_tracking": True,
            "enable_eval_logging": False,
            "dry_run_mode": True,
        },
        "prompts": {
            "registry_path": "./prompts",
            "default_version": "v1",
            "hot_reload": False,
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "output_path": "./logs/test.jsonl",
            "include_prompt_in_log": False,
            "sample_rate": 1.0,
        },
    }
    if overrides:
        base_config.update(overrides)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(base_config, f)
    f.close()
    return f.name


class TestConfigLoader:

    def test_loads_valid_config(self):
        path = _write_temp_config()
        loader = ConfigLoader()
        config = loader.load(path)
        assert config.version == "1.0.0"
        assert config.environment == "test"

    def test_all_model_tiers_loaded(self):
        path = _write_temp_config()
        loader = ConfigLoader()
        config = loader.load(path)
        assert "tier_low" in config.models
        assert "tier_medium" in config.models
        assert "tier_high" in config.models
        assert "fallback" in config.models

    def test_model_provider_is_openrouter(self):
        path = _write_temp_config()
        loader = ConfigLoader()
        config = loader.load(path)
        for tier_name, tier in config.models.items():
            assert tier.provider == "openrouter", f"{tier_name} should use openrouter"

    def test_model_has_required_fields(self):
        path = _write_temp_config()
        loader = ConfigLoader()
        config = loader.load(path)
        tier = config.models["tier_high"]
        assert hasattr(tier, "model_id")
        assert hasattr(tier, "temperature")
        assert hasattr(tier, "max_tokens")
        assert hasattr(tier, "cost_per_1k_input_tokens")
        assert hasattr(tier, "cost_per_1k_output_tokens")

    def test_routing_config_loaded(self):
        path = _write_temp_config()
        loader = ConfigLoader()
        config = loader.load(path)
        assert config.routing.low_complexity_threshold == 0.3
        assert config.routing.high_complexity_threshold == 0.7
        assert isinstance(config.routing.high_complexity_keywords, list)
        assert "refund" in config.routing.high_complexity_keywords

    def test_budget_config_loaded(self):
        path = _write_temp_config()
        loader = ConfigLoader()
        config = loader.load(path)
        assert config.budget.daily_budget_usd == 16.67
        assert config.budget.hard_limit_pct == 0.95

    def test_env_var_overrides_budget(self):
        path = _write_temp_config()
        os.environ["FIXIT_DAILY_BUDGET"] = "25.00"
        loader = ConfigLoader()
        config = loader.load(path)
        assert config.budget.daily_budget_usd == 25.00
        del os.environ["FIXIT_DAILY_BUDGET"]

    def test_env_var_overrides_environment(self):
        path = _write_temp_config()
        os.environ["FIXIT_ENVIRONMENT"] = "staging"
        loader = ConfigLoader()
        config = loader.load(path)
        assert config.environment == "staging"
        del os.environ["FIXIT_ENVIRONMENT"]

    def test_feature_flags_type(self):
        path = _write_temp_config()
        loader = ConfigLoader()
        config = loader.load(path)
        assert isinstance(config.features.enable_fallback_on_budget, bool)
        assert isinstance(config.features.dry_run_mode, bool)
        assert isinstance(config.features.enable_structured_logging, bool)

    def test_dry_run_mode_default_true(self):
        path = _write_temp_config()
        loader = ConfigLoader()
        config = loader.load(path)
        assert config.features.dry_run_mode is True

    def test_prompts_config_loaded(self):
        path = _write_temp_config()
        loader = ConfigLoader()
        config = loader.load(path)
        assert "registry_path" in config.prompts
        assert "default_version" in config.prompts

    def test_reload_returns_updated_config(self):
        path = _write_temp_config()
        loader = ConfigLoader()
        config1 = loader.load(path)
        config2 = loader.reload()
        assert config1.version == config2.version
