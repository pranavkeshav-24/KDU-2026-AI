from memory.extractor import extract_case_facts


def test_extracts_case_facts_and_flags_missing_fields():
    facts = extract_case_facts("Order ORD-12345 charged $420.55 on May 3, 2025. Routing 123456789. Account 987654321012.")

    assert "ORD-12345" in facts.order_ids
    assert {"raw": "$420.55"} in facts.transaction_amounts
    assert "123456789" in facts.routing_numbers
    assert "987654321012" in facts.account_numbers
    assert "May 3, 2025" in facts.dates
    assert facts.requires_user_input
    assert "cvv" in facts.missing_fields

