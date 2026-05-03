import pytest

from circuit_breaker import get_circuit_state, reset_circuit
from executor.executor_agent import execute_plan
from memory.compaction import compact_memory
from planner.planner_agent import generate_plan
from tools.database_tools import query_internal_database
from tools.delegation_tools import coordinate_request


def test_phase1_database_tool_stops_at_three_invocations():
    reset_circuit("query_internal_database")
    outputs = [query_internal_database("active users") for _ in range(5)]
    state = get_circuit_state("query_internal_database")

    assert "[LOOP DETECTED]" in outputs[2]
    assert "[CIRCUIT OPEN]" in outputs[3]
    assert state.protected_call_count == 3


@pytest.mark.asyncio
async def test_phase2_coordinator_delegates_to_finance_and_hr():
    response = await coordinate_request("What is John's salary and how much PTO does he have?")

    assert "salary" in response.lower()
    assert "pto" in response.lower()


@pytest.mark.asyncio
async def test_phase4_compaction_preserves_facts_and_redacts_summary_values():
    result = await compact_memory(["ok", "Order ORD-9999 charged $99.00 on 01/02/2025. Routing 123456789."])

    assert "ORD-9999" in result["case_facts"]["order_ids"]
    assert "$99.00" not in result["summary"]


@pytest.mark.asyncio
async def test_phase5_plan_executes_in_dependency_order():
    plan = await generate_plan("Get John's compensation package and update his tax withholding")
    result = await execute_plan(plan, {})

    assert 1 in result["results"]
    assert any("withholding" in value.lower() for value in result["results"].values())
    assert "step_1_result" in result["final_memory"]

