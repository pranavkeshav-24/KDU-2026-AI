from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

from config import CIRCUIT_BREAKER_COOLDOWN_SECONDS
from observability.logger import StructuredLogger


logger = StructuredLogger("circuit_breaker")


@dataclass
class CircuitState:
    failure_count: int = 0
    is_open: bool = False
    opened_at: float = 0.0
    cooldown_seconds: int = CIRCUIT_BREAKER_COOLDOWN_SECONDS
    blocked_calls: int = 0
    protected_call_count: int = 0


CIRCUIT_REGISTRY: dict[str, CircuitState] = {}


def circuit_breaker(tool_name: str, max_failures: int = 3, cooldown_seconds: int | None = None):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        state = CIRCUIT_REGISTRY.setdefault(tool_name, CircuitState())
        if cooldown_seconds is not None:
            state.cooldown_seconds = cooldown_seconds

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            state = CIRCUIT_REGISTRY[tool_name]
            if state.is_open:
                elapsed = time.time() - state.opened_at
                if elapsed >= state.cooldown_seconds:
                    state.is_open = False
                    state.failure_count = 0
                    logger.log("circuit_auto_reset", tool_name=tool_name, elapsed=round(elapsed, 3))
                else:
                    state.blocked_calls += 1
                    logger.log("circuit_blocked_call", tool_name=tool_name, failure_count=state.failure_count)
                    return (
                        f"[CIRCUIT OPEN] The tool '{tool_name}' is temporarily unavailable "
                        f"after {max_failures} consecutive failures. Please try again later or contact support."
                    )

            try:
                state.protected_call_count += 1
                result = func(*args, **kwargs)
                state.failure_count = 0
                return result
            except Exception as exc:
                state.failure_count += 1
                logger.log("tool_failure", tool_name=tool_name, failure_count=state.failure_count, error=str(exc))
                if state.failure_count >= max_failures:
                    state.is_open = True
                    state.opened_at = time.time()
                    logger.log("circuit_opened", tool_name=tool_name, failure_count=state.failure_count)
                    return (
                        f"[LOOP DETECTED] The tool '{tool_name}' has failed {max_failures} times in a row. "
                        "Automatic retries have been halted. Returning graceful fallback: unable to retrieve active user count."
                    )
                return f"[TOOL ERROR] '{tool_name}' failed: {exc}. Attempt {state.failure_count}/{max_failures}."

        return wrapper

    return decorator


def reset_circuit(tool_name: str | None = None) -> None:
    if tool_name is None:
        CIRCUIT_REGISTRY.clear()
        return
    CIRCUIT_REGISTRY[tool_name] = CircuitState()

def get_circuit_state(tool_name: str) -> CircuitState:
    return CIRCUIT_REGISTRY.setdefault(tool_name, CircuitState())

