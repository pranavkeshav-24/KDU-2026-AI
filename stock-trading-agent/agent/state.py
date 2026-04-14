import os
from typing import Annotated, Any, Mapping, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

DEFAULT_INITIAL_CASH_USD = 10000.0


class Position(TypedDict):
    qty: int
    avg_cost: float


class Portfolio(TypedDict):
    holdings: dict[str, Position]
    cash_usd: float
    total_value_usd: float


class PendingOrder(TypedDict):
    action: str
    symbol: str
    qty: int
    price: float
    notional: float


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    portfolio: Portfolio
    intent: Optional[str]
    currency: Optional[str]
    pending_order: Optional[PendingOrder]
    human_approved: Optional[bool]
    total_tokens: int
    total_cost_usd: float


def get_initial_cash() -> float:
    raw_value = os.getenv("INITIAL_CASH_USD", str(DEFAULT_INITIAL_CASH_USD)).strip()
    try:
        return float(raw_value)
    except ValueError:
        return DEFAULT_INITIAL_CASH_USD


def default_portfolio(cash_usd: float | None = None) -> Portfolio:
    starting_cash = get_initial_cash() if cash_usd is None else float(cash_usd)
    return {
        "holdings": {},
        "cash_usd": round(starting_cash, 2),
        "total_value_usd": round(starting_cash, 2),
    }


def initialize_state(cash_usd: float | None = None) -> AgentState:
    portfolio = default_portfolio(cash_usd)
    return {
        "messages": [],
        "portfolio": portfolio,
        "intent": "chat",
        "currency": None,
        "pending_order": None,
        "human_approved": None,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
    }


def ensure_state_defaults(state: Mapping[str, Any] | None) -> AgentState:
    state_dict = dict(state or {})
    initial_state = initialize_state()

    raw_portfolio = state_dict.get("portfolio") or {}
    holdings: dict[str, Position] = {}
    for symbol, position in (raw_portfolio.get("holdings") or {}).items():
        qty = int(position.get("qty", 0))
        avg_cost = round(float(position.get("avg_cost", 0.0)), 2)
        if qty > 0:
            holdings[symbol] = {"qty": qty, "avg_cost": avg_cost}

    cash_usd = round(
        float(raw_portfolio.get("cash_usd", initial_state["portfolio"]["cash_usd"])),
        2,
    )
    total_value_usd = round(
        float(raw_portfolio.get("total_value_usd", cash_usd)),
        2,
    )

    return {
        "messages": list(state_dict.get("messages") or []),
        "portfolio": {
            "holdings": holdings,
            "cash_usd": cash_usd,
            "total_value_usd": total_value_usd,
        },
        "intent": state_dict.get("intent", initial_state["intent"]),
        "currency": state_dict.get("currency", initial_state["currency"]),
        "pending_order": state_dict.get("pending_order"),
        "human_approved": state_dict.get("human_approved"),
        "total_tokens": int(state_dict.get("total_tokens", 0)),
        "total_cost_usd": round(float(state_dict.get("total_cost_usd", 0.0)), 6),
    }
