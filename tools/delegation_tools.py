from __future__ import annotations

import asyncio
import re

from agents_runtime import SafeRunner
from context import build_context_payload
from observability.logger import StructuredLogger


logger = StructuredLogger("delegation")


async def delegate_to_finance_agent(task: str, context_json: str) -> str:
    from agents_config import finance_agent

    logger.log("agent_delegated", from_agent="Coordinator", to_agent="FinanceAgent", task=task)
    prompt = f"Context: {context_json}\n\nTask: {task}"
    result = await SafeRunner.run(finance_agent, prompt)
    return result.final_output


async def delegate_to_hr_agent(task: str, context_json: str) -> str:
    from agents_config import hr_agent

    logger.log("agent_delegated", from_agent="Coordinator", to_agent="HRAgent", task=task)
    prompt = f"Context: {context_json}\n\nTask: {task}"
    result = await SafeRunner.run(hr_agent, prompt)
    return result.final_output


async def coordinate_request(request: str) -> str:
    employee_id = _extract_employee_id(request)
    outputs: list[str] = []
    if _is_finance_request(request):
        payload = build_context_payload(
            task_id="task_finance_001",
            user_intent=_finance_intent(request),
            employee_id=employee_id,
            routing_number=_first_match(r"\b\d{9}\b", request),
            account_number=_first_match(r"\b\d{10,17}\b", request),
            required_action="update_banking_info" if "bank" in request.lower() or "routing" in request.lower() else None,
        )
        outputs.append(await delegate_to_finance_agent(_finance_intent(request), payload.to_json()))
    if _is_hr_request(request):
        payload = build_context_payload(task_id="task_hr_001", user_intent=_hr_intent(request), employee_id=employee_id)
        outputs.append(await delegate_to_hr_agent(_hr_intent(request), payload.to_json()))
    if not outputs:
        return "No finance or HR delegation target was identified."
    return "\n".join(outputs)


def coordinate_request_sync(request: str) -> str:
    try:
        return asyncio.run(coordinate_request(request))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coordinate_request(request))


def _is_finance_request(request: str) -> bool:
    lowered = request.lower()
    return any(term in lowered for term in ("salary", "bank", "routing", "account", "transaction", "tax", "withholding"))


def _is_hr_request(request: str) -> bool:
    lowered = request.lower()
    return any(term in lowered for term in ("pto", "vacation", "profile", "manager", "employee record"))


def _extract_employee_id(request: str) -> str:
    lowered = request.lower()
    if "jane" in lowered:
        return "jane"
    if "john" in lowered:
        return "john"
    return "john"


def _finance_intent(request: str) -> str:
    lowered = request.lower()
    if "bank" in lowered or "routing" in lowered:
        return "Update banking information"
    if "transaction" in lowered:
        return "Retrieve recent transactions"
    if "tax" in lowered or "withholding" in lowered:
        return "Update tax withholding"
    return "Get employee salary"


def _hr_intent(request: str) -> str:
    if "profile" in request.lower() or "manager" in request.lower():
        return "Retrieve employee profile"
    return "Get employee PTO balance"


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(0) if match else None

