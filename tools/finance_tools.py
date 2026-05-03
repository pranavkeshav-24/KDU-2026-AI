from __future__ import annotations

import json
import re


FINANCE_DATA = {
    "john": {
        "salary": "$124,000",
        "transactions": ["PAY-1001 salary deposit $4,769.23", "BEN-220 health premium $210.00"],
        "banking": {"routing_number": "111000025", "account_number": "123456789012"},
        "tax_withholding": "standard",
    },
    "jane": {
        "salary": "$118,500",
        "transactions": ["PAY-1002 salary deposit $4,557.69"],
        "banking": {"routing_number": "111000025", "account_number": "987654321098"},
        "tax_withholding": "standard",
    },
}


def normalize_employee_id(employee_id: str | None) -> str:
    if not employee_id:
        return "john"
    return employee_id.strip().lower().replace("'s", "")


def get_salary(employee_id: str) -> str:
    employee = normalize_employee_id(employee_id)
    record = FINANCE_DATA.get(employee)
    if not record:
        return f"No salary record found for {employee}."
    return f"{employee.title()}'s salary is {record['salary']}."


def update_banking_info(employee_id: str = "john", routing_number: str | None = None, account_number: str | None = None) -> str:
    employee = normalize_employee_id(employee_id)
    record = FINANCE_DATA.setdefault(employee, {"salary": "unknown", "transactions": [], "banking": {}, "tax_withholding": "standard"})
    if routing_number:
        record["banking"]["routing_number"] = routing_number
    if account_number:
        record["banking"]["account_number"] = account_number
    return f"Banking information updated for {employee.title()}."


def get_transactions(employee_id: str = "john", limit: int = 5) -> str:
    employee = normalize_employee_id(employee_id)
    transactions = FINANCE_DATA.get(employee, {}).get("transactions", [])
    return json.dumps(transactions[:limit])


def update_tax_withholding(employee_id: str = "john", basis: str | None = None) -> str:
    employee = normalize_employee_id(employee_id)
    FINANCE_DATA.setdefault(employee, {"salary": "unknown", "transactions": [], "banking": {}, "tax_withholding": "standard"})
    FINANCE_DATA[employee]["tax_withholding"] = "reviewed"
    suffix = f" based on {basis}" if basis else ""
    return f"Tax withholding marked for review for {employee.title()}{suffix}."


def route_finance_task(prompt: str) -> str:
    context = _extract_context(prompt)
    employee_id = context.get("employee_id") or _employee_from_text(prompt)
    if "routing" in prompt.lower() or context.get("required_action") == "update_banking_info":
        routing = context.get("routing_number") or _first_match(r"\b\d{9}\b", prompt)
        account = context.get("account_number") or _first_match(r"\b\d{10,17}\b", prompt)
        return update_banking_info(employee_id=employee_id, routing_number=routing, account_number=account)
    if "transaction" in prompt.lower():
        return get_transactions(employee_id=employee_id)
    if "tax" in prompt.lower() and "withholding" in prompt.lower():
        return update_tax_withholding(employee_id=employee_id, basis=context.get("basis"))
    return get_salary(employee_id=employee_id)


def _extract_context(prompt: str) -> dict:
    match = re.search(r"Context:\s*(\{.*?\})\s*(?:Task:|$)", prompt, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _employee_from_text(text: str) -> str:
    lowered = text.lower()
    for employee in FINANCE_DATA:
        if employee in lowered:
            return employee
    return "john"


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(0) if match else None

