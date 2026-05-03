from __future__ import annotations

from agents_runtime import AgentSpec
from config import GENERAL_AGENT_MODEL
from observability.logger import StructuredLogger
from planner.schema import ExecutionPlan
from tools.database_tools import query_internal_database
from tools.finance_tools import get_salary, get_transactions, update_banking_info, update_tax_withholding
from tools.hr_tools import get_employee_profile, get_pto_balance, update_employee_record


logger = StructuredLogger("executor")

executor_agent = AgentSpec(
    name="Executor",
    model=GENERAL_AGENT_MODEL,
    instructions=(
        "You are a task executor. Receive one step plus shared memory, invoke the exact tool, "
        "and return only the action result."
    ),
    tools=[get_salary, update_banking_info, get_transactions, update_tax_withholding, get_pto_balance, get_employee_profile, update_employee_record, query_internal_database],
)


TOOL_REGISTRY = {
    "get_salary": get_salary,
    "update_banking_info": update_banking_info,
    "get_transactions": get_transactions,
    "update_tax_withholding": update_tax_withholding,
    "get_pto_balance": get_pto_balance,
    "get_employee_profile": get_employee_profile,
    "update_employee_record": update_employee_record,
    "query_internal_database": query_internal_database,
}


async def execute_plan(plan: dict, shared_memory: dict | None = None) -> dict:
    execution_plan = ExecutionPlan.from_dict(plan)
    shared_memory = shared_memory or {}
    results: dict[int, str] = {}

    for step in sorted(execution_plan.steps, key=lambda item: item.step_id):
        try:
            _assert_dependencies(step.step_id, step.depends_on, results)
            resolved_params = _resolve_parameters(step.parameters, results)
            result = _execute_tool(step.tool, resolved_params)
            results[step.step_id] = result
            shared_memory[f"step_{step.step_id}_result"] = result
            logger.log("step_executed", plan_id=execution_plan.plan_id, step_id=step.step_id, status="completed", tool=step.tool)
        except Exception as exc:
            logger.log("step_executed", plan_id=execution_plan.plan_id, step_id=step.step_id, status="failed", tool=step.tool, error=str(exc))
            if step.on_failure == "skip":
                results[step.step_id] = f"[SKIPPED] {exc}"
                shared_memory[f"step_{step.step_id}_result"] = "SKIPPED"
                continue
            if step.on_failure == "retry":
                result = _execute_tool(step.tool, _resolve_parameters(step.parameters, results))
                results[step.step_id] = result
                shared_memory[f"step_{step.step_id}_result"] = result
                continue
            raise

    return {"plan_id": execution_plan.plan_id, "results": results, "final_memory": shared_memory}


def _assert_dependencies(step_id: int, depends_on: list[int], results: dict[int, str]) -> None:
    for dependency in depends_on:
        if dependency not in results:
            raise RuntimeError(f"Step {step_id} depends on step {dependency} which has not completed.")


def _resolve_parameters(parameters: dict, results: dict[int, str]) -> dict:
    resolved = {}
    for key, value in parameters.items():
        if isinstance(value, str) and value.startswith("step_") and value.endswith("_result"):
            ref_id = int(value.split("_")[1])
            resolved[key] = results.get(ref_id, "")
        else:
            resolved[key] = value
    return {key: value for key, value in resolved.items() if value is not None}


def _execute_tool(tool_name: str, parameters: dict) -> str:
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    return str(tool(**parameters))

