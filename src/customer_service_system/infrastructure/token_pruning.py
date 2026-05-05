from __future__ import annotations

from ..core import Message


def estimate_tokens(text: str) -> int:
    """A deterministic approximation used for budgeting without provider calls."""

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def prune_history(
    history: list[Message],
    token_budget: int,
    dropped_summary: str | None = None,
) -> list[Message]:
    total = 0
    kept_reversed: list[Message] = []
    for message in reversed(history):
        message_tokens = estimate_tokens(message.role) + estimate_tokens(message.content)
        if kept_reversed and total + message_tokens > token_budget:
            break
        if not kept_reversed and message_tokens > token_budget:
            content_budget = max(0, token_budget * 4 - len(message.role) - 1)
            kept_reversed.append(Message(role=message.role, content=message.content[-content_budget:]))
            break
        kept_reversed.append(message)
        total += message_tokens

    kept = list(reversed(kept_reversed))
    dropped = len(history) - len(kept)
    if dropped > 0 and dropped_summary:
        return [Message(role="system", content=dropped_summary)] + kept
    return kept
