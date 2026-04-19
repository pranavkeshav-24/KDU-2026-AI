# config/loader.py
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict
import threading


@dataclass
class ModelTier:
    model_id: str
    provider: str
    temperature: float
    max_tokens: int
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float


@dataclass
class BudgetConfig:
    daily_budget_usd: float
    monthly_budget_usd: float
    warning_threshold_pct: float
    hard_limit_pct: float
    per_query_max_usd: float
    reset_timezone: str


@dataclass
class RoutingConfig:
    low_complexity_threshold: float
    high_complexity_threshold: float
    high_complexity_keywords: list
    complaint_keywords: list
    booking_keywords: list
    faq_keywords: list
    max_retries: int
    timeout_seconds: int
    fallback_on_error: bool


@dataclass
class FeatureFlags:
    enable_semantic_cache: bool = False
    enable_fallback_on_budget: bool = True
    enable_structured_logging: bool = True
    enable_cost_tracking: bool = True
    enable_eval_logging: bool = True
    dry_run_mode: bool = True


@dataclass
class AppConfig:
    version: str
    environment: str
    models: Dict[str, ModelTier]
    routing: RoutingConfig
    budget: BudgetConfig
    features: FeatureFlags
    prompts: dict
    logging: dict


class ConfigLoader:
    """
    Loads configuration from YAML file.
    Supports hot-reload via reload() without restarting the server.

    AWS Future Replacement: AWS AppConfig
    ─────────────────────────────────────
    AppConfig provides versioned configurations, deployment strategies
    (canary, linear rollout), and automatic rollback when CloudWatch
    alarms fire. Every config change becomes a tracked deployment with
    full audit trail — no manual file edits in production.

    Migration path:
      1. Store config.yaml content in AppConfig as a YAML freeform config
      2. Use the AppConfig SDK to fetch and cache configs with TTL
      3. CloudWatch alarms trigger automatic rollback on error spikes
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._config = None
                cls._instance._config_path = None
        return cls._instance

    def load(self, config_path: str = None) -> AppConfig:
        if config_path is None:
            # Resolve relative to this file's location
            config_path = str(
                Path(__file__).parent / "config.yaml"
            )
        self._config_path = config_path
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        # Override with environment variables (12-factor app pattern)
        if os.getenv("FIXIT_ENVIRONMENT"):
            raw["environment"] = os.getenv("FIXIT_ENVIRONMENT")
        if os.getenv("FIXIT_DAILY_BUDGET"):
            raw["budget"]["daily_budget_usd"] = float(os.getenv("FIXIT_DAILY_BUDGET"))
        if os.getenv("LOG_LEVEL"):
            raw["logging"]["level"] = os.getenv("LOG_LEVEL")

        self._config = self._parse(raw)
        return self._config

    def get(self) -> AppConfig:
        if self._config is None:
            return self.load()
        return self._config

    def reload(self) -> AppConfig:
        """Hot-reload config without restart."""
        return self.load(self._config_path)

    def _parse(self, raw: dict) -> AppConfig:
        models = {
            tier: ModelTier(**data)
            for tier, data in raw["models"].items()
        }
        routing = RoutingConfig(**raw["routing"])
        budget = BudgetConfig(**raw["budget"])
        features = FeatureFlags(**raw["features"])

        return AppConfig(
            version=raw["version"],
            environment=raw["environment"],
            models=models,
            routing=routing,
            budget=budget,
            features=features,
            prompts=raw["prompts"],
            logging=raw["logging"],
        )


# Singleton accessor
config_loader = ConfigLoader()
