from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal


AllowedAgent = Literal["finance", "hr", "db"]
FailurePolicy = Literal["abort", "skip", "retry"]


@dataclass
class PlanStep:
    step_id: int
    action: str
    agent: AllowedAgent
    tool: str
    parameters: dict
    depends_on: list[int] = field(default_factory=list)
    on_failure: FailurePolicy = "abort"

    @classmethod
    def from_dict(cls, data: dict) -> "PlanStep":
        required = {"step_id", "action", "agent", "tool", "parameters"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"Plan step missing required fields: {sorted(missing)}")
        return cls(
            step_id=int(data["step_id"]),
            action=str(data["action"]),
            agent=data["agent"],
            tool=str(data["tool"]),
            parameters=dict(data["parameters"]),
            depends_on=[int(value) for value in data.get("depends_on", [])],
            on_failure=data.get("on_failure", "abort"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionPlan:
    plan_id: str
    goal: str
    steps: list[PlanStep]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionPlan":
        required = {"plan_id", "goal", "steps"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"Plan missing required fields: {sorted(missing)}")
        steps = [PlanStep.from_dict(step) for step in data["steps"]]
        plan = cls(plan_id=str(data["plan_id"]), goal=str(data["goal"]), steps=steps, created_at=data.get("created_at") or datetime.now(timezone.utc).isoformat())
        validate_plan(plan)
        return plan

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at,
        }


def validate_plan(plan: ExecutionPlan) -> None:
    ids = [step.step_id for step in plan.steps]
    if len(ids) != len(set(ids)):
        raise ValueError("Plan contains duplicate step IDs.")
    id_set = set(ids)
    for step in plan.steps:
        if step.agent not in {"finance", "hr", "db"}:
            raise ValueError(f"Unsupported agent: {step.agent}")
        if step.on_failure not in {"abort", "skip", "retry"}:
            raise ValueError(f"Unsupported failure policy: {step.on_failure}")
        for dependency in step.depends_on:
            if dependency not in id_set:
                raise ValueError(f"Step {step.step_id} depends on unknown step {dependency}.")
            if dependency == step.step_id:
                raise ValueError(f"Step {step.step_id} cannot depend on itself.")
    _assert_acyclic(plan)


def _assert_acyclic(plan: ExecutionPlan) -> None:
    graph = {step.step_id: set(step.depends_on) for step in plan.steps}
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(step_id: int) -> None:
        if step_id in visited:
            return
        if step_id in visiting:
            raise ValueError("Plan dependency graph contains a cycle.")
        visiting.add(step_id)
        for dependency in graph[step_id]:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in graph:
        visit(step_id)

