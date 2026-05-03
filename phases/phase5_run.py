from __future__ import annotations

import asyncio
import json

from executor.executor_agent import execute_plan
from planner.planner_agent import generate_plan


async def run_phase5() -> dict:
    plan = await generate_plan("Get John's compensation package and update his tax withholding")
    return await execute_plan(plan, shared_memory={})


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_phase5()), indent=2))

