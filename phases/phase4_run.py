from __future__ import annotations

import asyncio
import json

from memory.compaction import compact_memory


async def run_phase4() -> dict:
    return await compact_memory(
        [
            "okay",
            "Customer reported order ORD-12345 for $420.55 on May 3, 2025.",
            "Routing number is 123456789 and account number is 987654321012.",
            "They want a refund review but did not provide card_number or cvv.",
        ]
    )


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_phase4()), indent=2))

