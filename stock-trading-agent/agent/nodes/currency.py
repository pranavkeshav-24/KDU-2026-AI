from langchain_core.messages import AIMessage
from ..parsing import extract_currency, extract_first_float, format_currency_amount, format_usd
from ..state import AgentState, ensure_state_defaults
from ..tools.currency import EXCHANGE_RATES, convert_currency

def currency_converter(state: AgentState) -> dict:
    safe_state = ensure_state_defaults(state)
    message_text = safe_state["messages"][-1].content if safe_state["messages"] else ""

    amount = extract_first_float(message_text)
    target_currency = (
        extract_currency(message_text, EXCHANGE_RATES.keys())
        or safe_state.get("currency")
        or "INR"
    )

    if amount is None:
        amount = 1.0
        prefix = "I used 1 USD because no amount was provided. "
    else:
        prefix = ""

    result = convert_currency.invoke(
        {"amount_usd": amount, "target_currency": target_currency}
    )
    if "error" in result:
        return {"messages": [AIMessage(content=result["error"])]}

    rate = float(result["rate"])
    converted_amount = float(result["converted_amount"])
    content = (
        f"{prefix}{format_usd(amount)} = {format_currency_amount(converted_amount, target_currency)} "
        f"at {rate:,.2f} {target_currency} per USD."
    )
    return {
        "currency": target_currency,
        "messages": [AIMessage(content=content)],
    }
