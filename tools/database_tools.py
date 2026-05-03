from __future__ import annotations

from circuit_breaker import circuit_breaker


@circuit_breaker(tool_name="query_internal_database", max_failures=3)
def query_internal_database(query: str) -> str:
    raise ConnectionError("HTTP 500: Internal Server Error - database unavailable.")

