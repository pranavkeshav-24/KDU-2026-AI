from __future__ import annotations

import asyncio

from context import build_context_payload
from tools.delegation_tools import delegate_to_finance_agent


async def run_phase3() -> str:
    payload = build_context_payload(
        task_id="txn_20250503_001",
        user_intent="Update banking routing number",
        routing_number="123456789",
        required_action="update_banking_info",
    )
    return await delegate_to_finance_agent("Update the user's banking routing number.", payload.to_json())


if __name__ == "__main__":
    print(asyncio.run(run_phase3()))

