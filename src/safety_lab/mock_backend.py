from __future__ import annotations

from safety_lab.models import CustomerRecord


CUSTOMER_DB: dict[str, CustomerRecord] = {
    "cust_001": CustomerRecord(
        user_id="cust_001",
        name="Maya Raman",
        email="maya.raman@example.com",
        ssn="123-45-6789",
    )
}


def get_customer_record(user_id: str = "cust_001") -> CustomerRecord:
    try:
        return CUSTOMER_DB[user_id]
    except KeyError as exc:
        raise ValueError(f"Unknown customer id: {user_id}") from exc

