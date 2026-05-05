from __future__ import annotations

import pytest

from customer_service_system.core import Message
from customer_service_system.infrastructure import (
    ConcurrencyQueue,
    QueueFullError,
    SensitivePayloadError,
    prune_history,
    validate_no_sensitive_fields,
)


def test_sensitive_payload_validation_rejects_secrets():
    with pytest.raises(SensitivePayloadError):
        validate_no_sensitive_fields({"session_id": "s", "api_key": "do-not-cross"})


def test_sliding_window_pruning_keeps_recent_turns():
    history = [Message(role="user", content=f"turn {index} " * 20) for index in range(10)]

    pruned = prune_history(history, token_budget=40)

    assert pruned
    assert pruned[-1].content.startswith("turn 9")
    assert len(pruned) < len(history)


@pytest.mark.asyncio
async def test_concurrency_queue_rejects_when_depth_is_full():
    queue = ConcurrencyQueue(max_concurrent=1, max_depth=0)

    with pytest.raises(QueueFullError):
        async with queue.slot():
            pass
