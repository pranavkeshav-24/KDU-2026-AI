from __future__ import annotations

import asyncio

from agents_config import coordinator
from agents_runtime import SafeRunner


async def run_phase2() -> str:
    result = await SafeRunner.run(coordinator, "What is John's salary and how much PTO does he have?")
    return result.final_output


if __name__ == "__main__":
    print(asyncio.run(run_phase2()))

