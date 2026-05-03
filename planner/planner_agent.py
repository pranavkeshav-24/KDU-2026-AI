from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from agents_runtime import AgentSpec, SafeRunner
from config import REASONING_MODEL
from observability.logger import StructuredLogger
from planner.schema import ExecutionPlan


logger = StructuredLogger("planner")

PLANNER_SYSTEM_PROMPT = """
You are a task planning agent. Given a user goal, decompose it into a minimal sequence of discrete steps.
Output only JSON with plan_id, goal, and steps. Each step requires step_id, action, agent, tool,
parameters, depends_on, and on_failure. Planner has no tools.
"""

planner_agent = AgentSpec(name="Planner", model=REASONING_MODEL, instructions=PLANNER_SYSTEM_PROMPT, tools=[])


async def generate_plan(goal: str, *, use_sdk: bool = False) -> dict:
    if use_sdk:
        result = await SafeRunner.run(planner_agent, goal, use_sdk=True)
        raw = _strip_markdown_json(result.final_output)
    else:
        raw = generate_plan_locally(goal)
    parsed = json.loads(raw)
    plan = ExecutionPlan.from_dict(parsed)
    logger.log("plan_generated", plan_id=plan.plan_id, step_count=len(plan.steps), goal=goal)
    return plan.to_dict()


def generate_plan_locally(goal: str) -> str:
    lowered = goal.lower()
    steps: list[dict] = []
    step_id = 1
    employee_id = "jane" if "jane" in lowered else "john"

    if "salary" in lowered or "compensation" in lowered:
        steps.append(
            {
                "step_id": step_id,
                "action": f"Retrieve {employee_id.title()}'s salary",
                "agent": "finance",
                "tool": "get_salary",
                "parameters": {"employee_id": employee_id},
                "depends_on": [],
                "on_failure": "abort",
            }
        )
        step_id += 1

    if "pto" in lowered or "vacation" in lowered:
        steps.append(
            {
                "step_id": step_id,
                "action": f"Retrieve {employee_id.title()}'s PTO balance",
                "agent": "hr",
                "tool": "get_pto_balance",
                "parameters": {"employee_id": employee_id},
                "depends_on": [],
                "on_failure": "skip",
            }
        )
        step_id += 1

    if "bank" in lowered or "routing" in lowered:
        routing = _first_match(r"\b\d{9}\b", goal)
        account = _first_match(r"\b\d{10,17}\b", goal)
        steps.append(
            {
                "step_id": step_id,
                "action": "Update banking information",
                "agent": "finance",
                "tool": "update_banking_info",
                "parameters": {"employee_id": employee_id, "routing_number": routing, "account_number": account},
                "depends_on": [],
                "on_failure": "abort",
            }
        )
        step_id += 1

    if "tax" in lowered or "withholding" in lowered:
        salary_step = next((step["step_id"] for step in steps if step["tool"] == "get_salary"), None)
        steps.append(
            {
                "step_id": step_id,
                "action": "Update tax withholding based on salary",
                "agent": "finance",
                "tool": "update_tax_withholding",
                "parameters": {"employee_id": employee_id, "basis": f"step_{salary_step}_result" if salary_step else "manual_review"},
                "depends_on": [salary_step] if salary_step else [],
                "on_failure": "abort",
            }
        )
        step_id += 1

    if not steps:
        steps.append(
            {
                "step_id": 1,
                "action": "Run database lookup",
                "agent": "db",
                "tool": "query_internal_database",
                "parameters": {"query": goal[:200]},
                "depends_on": [],
                "on_failure": "skip",
            }
        )

    return json.dumps(
        {
            "plan_id": f"plan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "goal": goal,
            "steps": steps,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        sort_keys=True,
    )


def _strip_markdown_json(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(0) if match else None

