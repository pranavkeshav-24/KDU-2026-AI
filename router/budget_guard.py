# router/budget_guard.py
import json
import threading
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field, asdict
from config.loader import config_loader


@dataclass
class CostRecord:
    timestamp: str
    query_id: str
    model_id: str
    tier: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    category: str


@dataclass
class BudgetState:
    date: str                  # YYYY-MM-DD
    daily_spend_usd: float = 0.0
    monthly_spend_usd: float = 0.0
    total_requests: int = 0
    fallback_activations: int = 0
    records: list = field(default_factory=list)


class BudgetGuard:
    """
    Tracks per-request token costs and enforces daily/monthly budget limits.

    Local Implementation: In-memory state + JSON file persistence.
    State resets daily via check against stored date.

    AWS Future Replacement: Amazon DynamoDB + CloudWatch Budgets
    ─────────────────────────────────────────────────────────────
    Replace with DynamoDB for atomic cost increments across multiple
    Lambda instances. Use DynamoDB conditional writes (UpdateItem with
    ConditionExpression) to prevent race conditions when concurrent
    Lambda invocations update cost simultaneously.

    Additionally:
    - CloudWatch Custom Metrics for real-time budget dashboards
    - AWS Budgets for email/SNS alerts at 80% and 95% thresholds
    - DynamoDB TTL to automatically expire old daily records

    Migration path:
      1. Move budget state to a DynamoDB table (pk=date, sk=global)
      2. Use atomic ADD operations for cost increments
      3. Publish CloudWatch metric on each record_cost() call
      4. Set up AWS Budget with SNS → PagerDuty alerts
    """

    def __init__(self, state_path: str = "./logs/budget_state.json"):
        self._lock = threading.Lock()
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_or_initialize()

    def _load_or_initialize(self) -> BudgetState:
        today = str(date.today())
        if self._state_path.exists():
            try:
                with open(self._state_path) as f:
                    data = json.load(f)
                state = BudgetState(**{k: v for k, v in data.items() if k != "records"})
                state.records = data.get("records", [])
                if state.date != today:
                    # New day — reset daily counter, keep monthly
                    state.date = today
                    state.daily_spend_usd = 0.0
                    state.total_requests = 0
                    state.fallback_activations = 0
                    state.records = []
                return state
            except (json.JSONDecodeError, TypeError):
                pass
        return BudgetState(date=today)

    def _persist(self):
        with open(self._state_path, "w") as f:
            json.dump(asdict(self._state), f, indent=2)

    def estimate_cost(
        self, model_id: str, estimated_input_tokens: int, estimated_output_tokens: int
    ) -> float:
        config = config_loader.get()
        tier_config = self._get_model_config(model_id, config)
        if tier_config is None:
            return 0.001  # default estimate if model not found
        cost = (
            (estimated_input_tokens / 1000) * tier_config.cost_per_1k_input_tokens
            + (estimated_output_tokens / 1000) * tier_config.cost_per_1k_output_tokens
        )
        return round(cost, 6)

    def check_budget(self, estimated_cost: float) -> dict:
        """Returns routing decision based on current budget state."""
        config = config_loader.get()
        budget_cfg = config.budget
        with self._lock:
            hard_limit = budget_cfg.daily_budget_usd * budget_cfg.hard_limit_pct
            warning_limit = budget_cfg.daily_budget_usd * budget_cfg.warning_threshold_pct

            projected_spend = self._state.daily_spend_usd + estimated_cost

            if projected_spend > hard_limit:
                return {
                    "approved": False,
                    "use_fallback": True,
                    "reason": (
                        f"daily_budget_hard_limit: "
                        f"${self._state.daily_spend_usd:.4f} spent, "
                        f"limit=${hard_limit:.2f}"
                    ),
                    "budget_utilization": self._state.daily_spend_usd / budget_cfg.daily_budget_usd,
                }
            elif projected_spend > warning_limit:
                return {
                    "approved": True,
                    "use_fallback": False,
                    "warning": "approaching_daily_budget_limit",
                    "budget_utilization": self._state.daily_spend_usd / budget_cfg.daily_budget_usd,
                }
            return {
                "approved": True,
                "use_fallback": False,
                "budget_utilization": self._state.daily_spend_usd / budget_cfg.daily_budget_usd,
            }

    def record_cost(self, record: CostRecord):
        with self._lock:
            self._state.daily_spend_usd += record.cost_usd
            self._state.monthly_spend_usd += record.cost_usd
            self._state.total_requests += 1
            self._state.records.append(asdict(record))
            self._persist()

    def _get_model_config(self, model_id: str, config):
        for tier_name, tier in config.models.items():
            if tier.model_id == model_id:
                return tier
        return None

    def get_summary(self) -> dict:
        config = config_loader.get()
        with self._lock:
            daily_spend = self._state.daily_spend_usd
            total_requests = self._state.total_requests
            fallback_activations = self._state.fallback_activations
        return {
            "date": self._state.date,
            "daily_spend_usd": round(daily_spend, 4),
            "daily_budget_usd": config.budget.daily_budget_usd,
            "monthly_spend_usd": round(self._state.monthly_spend_usd, 4),
            "monthly_budget_usd": config.budget.monthly_budget_usd,
            "budget_utilization_pct": round(
                (daily_spend / config.budget.daily_budget_usd) * 100, 2
            ),
            "total_requests": total_requests,
            "fallback_activations": fallback_activations,
            "warning_threshold_pct": config.budget.warning_threshold_pct * 100,
            "hard_limit_pct": config.budget.hard_limit_pct * 100,
        }

    def increment_fallback_counter(self):
        with self._lock:
            self._state.fallback_activations += 1
