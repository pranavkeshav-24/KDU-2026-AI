from __future__ import annotations

from agents_runtime import AgentSpec
from config import GENERAL_AGENT_MODEL, REASONING_MODEL
from tools.database_tools import query_internal_database
from tools.delegation_tools import delegate_to_finance_agent, delegate_to_hr_agent
from tools.finance_tools import get_salary, get_transactions, update_banking_info, update_tax_withholding
from tools.hr_tools import get_employee_profile, get_pto_balance, update_employee_record


database_analyst = AgentSpec(
    name="DatabaseAnalyst",
    model=REASONING_MODEL,
    instructions=(
        "You are a data analyst. Use query_internal_database for internal metrics. "
        "If the tool returns an error or circuit-open message, acknowledge the failure clearly."
    ),
    tools=[query_internal_database],
)

finance_agent = AgentSpec(
    name="FinanceAgent",
    model=GENERAL_AGENT_MODEL,
    instructions=(
        "You are a finance specialist. Answer only finance-related queries using your tools. "
        "Be concise and return structured results."
    ),
    tools=[get_salary, update_banking_info, get_transactions, update_tax_withholding],
)

hr_agent = AgentSpec(
    name="HRAgent",
    model=GENERAL_AGENT_MODEL,
    instructions=(
        "You are an HR specialist. Answer only HR-related queries using your tools. "
        "Be concise and return structured results."
    ),
    tools=[get_pto_balance, get_employee_profile, update_employee_record],
)

coordinator = AgentSpec(
    name="Coordinator",
    model=GENERAL_AGENT_MODEL,
    instructions=(
        "You are a routing coordinator. Decompose multi-part user requests and delegate each part "
        "to the correct specialist agent. Never answer domain questions yourself."
    ),
    tools=[delegate_to_finance_agent, delegate_to_hr_agent],
)

