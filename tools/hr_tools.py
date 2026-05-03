from __future__ import annotations

import json


HR_DATA = {
    "john": {"pto_balance": "14 days", "title": "Senior Analyst", "manager": "Priya Shah"},
    "jane": {"pto_balance": "9 days", "title": "Product Manager", "manager": "Luis Romero"},
}


def normalize_employee_id(employee_id: str | None) -> str:
    return (employee_id or "john").strip().lower().replace("'s", "")


def get_pto_balance(employee_id: str) -> str:
    employee = normalize_employee_id(employee_id)
    record = HR_DATA.get(employee)
    if not record:
        return f"No PTO record found for {employee}."
    return f"{employee.title()} has {record['pto_balance']} of PTO remaining."


def get_employee_profile(employee_id: str) -> str:
    employee = normalize_employee_id(employee_id)
    return json.dumps(HR_DATA.get(employee, {"error": f"No employee profile found for {employee}."}))


def update_employee_record(employee_id: str, field: str, value: str) -> str:
    employee = normalize_employee_id(employee_id)
    HR_DATA.setdefault(employee, {})
    HR_DATA[employee][field] = value
    return f"Employee record updated for {employee.title()}: {field}={value}."


def route_hr_task(prompt: str) -> str:
    employee = "jane" if "jane" in prompt.lower() else "john"
    if "profile" in prompt.lower() or "manager" in prompt.lower():
        return get_employee_profile(employee)
    if "update" in prompt.lower():
        return update_employee_record(employee, "note", "updated via HR agent")
    return get_pto_balance(employee)

