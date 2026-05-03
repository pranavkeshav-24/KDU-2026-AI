from __future__ import annotations

import re

from memory.case_facts import CaseFacts


PATTERNS = {
    "order_id": r"\b(ORD-\d{4,}|ORDER[#\s]?\d{4,})\b",
    "amount": r"\$[\d,]+(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)\b",
    "routing": r"\b\d{9}\b",
    "account": r"\b\d{10,17}\b",
    "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b",
}

REQUIRED_FIELDS = ["cvv", "expiry_date", "card_number"]


def extract_case_facts(text: str, required_fields: list[str] | None = None) -> CaseFacts:
    facts = CaseFacts()
    required_fields = required_fields or REQUIRED_FIELDS

    for order_id in re.findall(PATTERNS["order_id"], text, re.IGNORECASE):
        value = order_id if isinstance(order_id, str) else order_id[0]
        if value not in facts.order_ids:
            facts.order_ids.append(value)

    for amount in re.findall(PATTERNS["amount"], text):
        facts.transaction_amounts.append({"raw": amount})

    for routing in re.findall(PATTERNS["routing"], text):
        if routing not in facts.routing_numbers:
            facts.routing_numbers.append(routing)

    for account in re.findall(PATTERNS["account"], text):
        if account not in facts.routing_numbers and account not in facts.account_numbers:
            facts.account_numbers.append(account)

    for date in re.findall(PATTERNS["date"], text):
        if date not in facts.dates:
            facts.dates.append(date)

    lowered = text.lower()
    for field_name in required_fields:
        if field_name.lower() not in lowered:
            facts.missing_fields.append(field_name)
            facts.raw_flags.append(f"MISSING_REQUIRED_FIELD: '{field_name}' not found in input.")

    facts.requires_user_input = bool(facts.missing_fields)
    return facts

