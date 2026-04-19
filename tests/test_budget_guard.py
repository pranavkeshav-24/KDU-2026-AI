# tests/test_budget_guard.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
from router.budget_guard import BudgetGuard, CostRecord
from datetime import datetime


def _make_guard() -> BudgetGuard:
    """Create a BudgetGuard with a fresh temp state file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    # Remove the file so BudgetGuard initialises fresh
    Path(tmp.name).unlink(missing_ok=True)
    return BudgetGuard(state_path=tmp.name)


def _make_record(**kwargs) -> CostRecord:
    defaults = dict(
        timestamp=datetime.utcnow().isoformat(),
        query_id="test-000",
        model_id="liquid/lfm-2.5-1.2b-instruct:free",
        tier="tier_low",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.001,
        category="faq",
    )
    defaults.update(kwargs)
    return CostRecord(**defaults)


class TestBudgetGuard:

    def test_cost_estimation_returns_non_negative(self):
        guard = _make_guard()
        cost = guard.estimate_cost("liquid/lfm-2.5-1.2b-instruct:free", 500, 200)
        assert cost >= 0

    def test_cost_estimation_unknown_model_returns_default(self):
        guard = _make_guard()
        # Unknown model_id should return 0.001 default
        cost = guard.estimate_cost("unknown-model-xyz", 500, 200)
        assert cost == 0.001

    def test_budget_approved_when_empty(self):
        guard = _make_guard()
        result = guard.check_budget(0.001)
        assert result["approved"] is True
        assert result["use_fallback"] is False

    def test_budget_denied_when_hard_limit_exceeded(self):
        guard = _make_guard()
        # Spend close to/past the hard limit ($16.67 * 0.95 = $15.84)
        record = _make_record(cost_usd=16.0, query_id="big-spender")
        guard.record_cost(record)
        result = guard.check_budget(5.0)
        assert result["approved"] is False
        assert result["use_fallback"] is True

    def test_budget_warning_when_near_limit(self):
        guard = _make_guard()
        # Spend at ~82% of daily budget ($16.67 * 0.80 = $13.34)
        record = _make_record(cost_usd=13.50, query_id="warning-zone")
        guard.record_cost(record)
        # Small additional cost should trigger warning but still be approved
        result = guard.check_budget(0.01)
        assert result["approved"] is True
        assert "warning" in result

    def test_cost_recorded_increases_spend(self):
        guard = _make_guard()
        initial = guard.get_summary()["daily_spend_usd"]
        record = _make_record(cost_usd=0.005, query_id="test-002")
        guard.record_cost(record)
        updated = guard.get_summary()["daily_spend_usd"]
        assert updated == pytest.approx(initial + 0.005, abs=1e-5)

    def test_multiple_records_accumulate(self):
        guard = _make_guard()
        for i in range(5):
            guard.record_cost(_make_record(cost_usd=0.01, query_id=f"q{i}"))
        summary = guard.get_summary()
        assert summary["daily_spend_usd"] == pytest.approx(0.05, abs=1e-5)
        assert summary["total_requests"] == 5

    def test_summary_has_required_fields(self):
        guard = _make_guard()
        summary = guard.get_summary()
        required = [
            "date", "daily_spend_usd", "daily_budget_usd",
            "monthly_spend_usd", "monthly_budget_usd",
            "budget_utilization_pct", "total_requests",
            "fallback_activations",
        ]
        for field in required:
            assert field in summary, f"Missing field: {field}"

    def test_budget_utilization_pct_is_percentage(self):
        guard = _make_guard()
        guard.record_cost(_make_record(cost_usd=1.667))  # 10% of $16.67
        summary = guard.get_summary()
        assert 9.0 <= summary["budget_utilization_pct"] <= 11.0

    def test_fallback_counter_increments(self):
        guard = _make_guard()
        initial = guard.get_summary()["fallback_activations"]
        guard.increment_fallback_counter()
        guard.increment_fallback_counter()
        assert guard.get_summary()["fallback_activations"] == initial + 2

    def test_state_persisted_to_file(self):
        """Verify state is written to the JSON file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        path = tmp.name
        guard = BudgetGuard(state_path=path)
        guard.record_cost(_make_record(cost_usd=0.123))
        # Load a second guard from same file
        guard2 = BudgetGuard(state_path=path)
        # State should be loaded (same day, so same daily spend)
        assert guard2.get_summary()["daily_spend_usd"] == pytest.approx(0.123, abs=1e-5)

    def test_category_recorded_in_record(self):
        guard = _make_guard()
        guard.record_cost(_make_record(category="complaint", cost_usd=0.002))
        summary = guard.get_summary()
        assert summary["total_requests"] == 1
