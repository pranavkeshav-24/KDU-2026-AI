from context import ContextPayload, build_context_payload


def test_context_payload_excludes_none_and_round_trips():
    payload = build_context_payload(
        task_id="task_1",
        user_intent="Update banking routing number",
        routing_number="123456789",
        required_action="update_banking_info",
    )
    data = payload.to_dict()
    assert data["routing_number"] == "123456789"
    assert "account_number" not in data
    assert "conversation_history" not in data
    assert ContextPayload.from_json(payload.to_json()).routing_number == "123456789"

