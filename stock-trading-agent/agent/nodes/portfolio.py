from langchain_core.messages import AIMessage
from ..parsing import format_usd
from ..state import AgentState, ensure_state_defaults
from ..tools.portfolio import calculate_portfolio

def portfolio_calculator(state: AgentState) -> dict:
    safe_state = ensure_state_defaults(state)
    portfolio = safe_state["portfolio"]
    result = calculate_portfolio.invoke(
        {
            "holdings": portfolio["holdings"],
            "cash_usd": portfolio["cash_usd"],
        }
    )

    lines = [
        f"Portfolio value: {format_usd(result['total_value_usd'])}.",
        f"Cash: {format_usd(result['cash_usd'])}.",
    ]
    if result["breakdown"]:
        positions = []
        for item in result["breakdown"]:
            pnl_label = format_usd(abs(item["pnl"]))
            pnl_prefix = "+" if item["pnl"] >= 0 else "-"
            positions.append(
                f"{item['symbol']} {item['qty']} sh @ {format_usd(item['market_price'])} ({pnl_prefix}{pnl_label} P/L)"
            )
        lines.append(f"Holdings: {'; '.join(positions)}.")
    else:
        lines.append("Holdings: none yet.")

    return {
        "portfolio": {
            "holdings": portfolio["holdings"],
            "cash_usd": round(float(result["cash_usd"]), 2),
            "total_value_usd": round(float(result["total_value_usd"]), 2),
        },
        "messages": [AIMessage(content=" ".join(lines))],
    }
