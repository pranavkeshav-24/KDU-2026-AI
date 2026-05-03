import pytest

from planner.schema import ExecutionPlan


def test_valid_plan_schema_round_trips():
    plan = ExecutionPlan.from_dict(
        {
            "plan_id": "plan_1",
            "goal": "Get John's salary",
            "steps": [
                {
                    "step_id": 1,
                    "action": "Retrieve salary",
                    "agent": "finance",
                    "tool": "get_salary",
                    "parameters": {"employee_id": "john"},
                    "depends_on": [],
                    "on_failure": "abort",
                }
            ],
        }
    )

    assert plan.steps[0].tool == "get_salary"


def test_plan_rejects_cycles():
    with pytest.raises(ValueError, match="cycle"):
        ExecutionPlan.from_dict(
            {
                "plan_id": "plan_bad",
                "goal": "cycle",
                "steps": [
                    {"step_id": 1, "action": "A", "agent": "finance", "tool": "get_salary", "parameters": {}, "depends_on": [2], "on_failure": "abort"},
                    {"step_id": 2, "action": "B", "agent": "hr", "tool": "get_pto_balance", "parameters": {}, "depends_on": [1], "on_failure": "abort"},
                ],
            }
        )

