from __future__ import annotations

import asyncio

from agents_config import database_analyst
from agents_runtime import SafeRunner


async def run_phase1() -> str:
    result = await SafeRunner.run(database_analyst, "Count the active users.")
    return result.final_output


if __name__ == "__main__":
    print(asyncio.run(run_phase1()))

