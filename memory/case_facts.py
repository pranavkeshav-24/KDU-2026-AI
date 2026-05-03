from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class CaseFacts:
    order_ids: list[str] = field(default_factory=list)
    transaction_amounts: list[dict] = field(default_factory=list)
    account_numbers: list[str] = field(default_factory=list)
    routing_numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    requires_user_input: bool = False
    raw_flags: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

