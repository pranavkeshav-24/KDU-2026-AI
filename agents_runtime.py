from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable


try:
    from agents import Agent as SDKAgent
    from agents import Runner as SDKRunner

    AGENTS_SDK_AVAILABLE = True
except Exception:
    SDKAgent = None
    SDKRunner = None
    AGENTS_SDK_AVAILABLE = False


@dataclass
class LocalRunResult:
    final_output: str


@dataclass
class AgentSpec:
    name: str
    model: str
    instructions: str
    tools: list[Callable[..., Any]] = field(default_factory=list)

    def to_sdk_agent(self):
        if not AGENTS_SDK_AVAILABLE:
            return self
        return SDKAgent(name=self.name, model=self.model, instructions=self.instructions, tools=self.tools)


class SafeRunner:
    """Small adapter that keeps demos testable without making LLM calls by default."""

    @staticmethod
    async def run(agent: AgentSpec, prompt: str, *, use_sdk: bool = False) -> LocalRunResult:
        if use_sdk and AGENTS_SDK_AVAILABLE:
            result = await SDKRunner.run(agent.to_sdk_agent(), prompt)
            return LocalRunResult(final_output=str(result.final_output))
        if agent.name == "Coordinator":
            from tools.delegation_tools import coordinate_request

            return LocalRunResult(final_output=await coordinate_request(prompt))
        return LocalRunResult(final_output=local_agent_response(agent, prompt))


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.create_task(coro)


def local_agent_response(agent: AgentSpec, prompt: str) -> str:
    lowered = prompt.lower()
    if agent.name == "DatabaseAnalyst":
        for tool in agent.tools:
            if getattr(tool, "__name__", "") == "query_internal_database":
                return str(tool("active users"))
        return "No database tool is available."
    if agent.name == "FinanceAgent":
        from tools.finance_tools import route_finance_task

        return route_finance_task(prompt)
    if agent.name == "HRAgent":
        from tools.hr_tools import route_hr_task

        return route_hr_task(prompt)
    if agent.name == "Coordinator":
        from tools.delegation_tools import coordinate_request_sync

        return coordinate_request_sync(prompt)
    if agent.name == "Summarizer":
        words = [w for w in prompt.split() if not any(ch.isdigit() for ch in w)]
        return " ".join(words[:80]) or "No substantive content."
    if agent.name == "Planner":
        from planner.planner_agent import generate_plan_locally

        return generate_plan_locally(prompt)
    if "salary" in lowered:
        return "Salary lookup completed."
    return "Task completed."
