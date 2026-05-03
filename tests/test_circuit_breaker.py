from circuit_breaker import circuit_breaker, get_circuit_state, reset_circuit


def test_circuit_opens_after_three_failures_and_blocks_later_calls():
    reset_circuit("boom")
    calls = {"count": 0}

    @circuit_breaker("boom", max_failures=3)
    def failing_tool():
        calls["count"] += 1
        raise RuntimeError("fail")

    assert "[TOOL ERROR]" in failing_tool()
    assert "[TOOL ERROR]" in failing_tool()
    assert "[LOOP DETECTED]" in failing_tool()
    assert "[CIRCUIT OPEN]" in failing_tool()
    assert calls["count"] == 3
    assert get_circuit_state("boom").is_open


def test_circuit_success_resets_failure_count():
    reset_circuit("flaky")
    calls = {"count": 0}

    @circuit_breaker("flaky", max_failures=3)
    def flaky_tool():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("first fail")
        return "ok"

    assert "[TOOL ERROR]" in flaky_tool()
    assert flaky_tool() == "ok"
    assert get_circuit_state("flaky").failure_count == 0

