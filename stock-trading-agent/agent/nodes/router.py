from ..state import AgentState, ensure_state_defaults

INTENT_KEYWORDS = {
    "buy": ["buy", "purchase", "acquire"],
    "sell": ["sell", "offload", "dump"],
    "price": ["price", "quote", "how much", "cost of", "ticker"],
    "convert": ["convert", "inr", "eur", "gbp", "jpy", "rupee", "euro", "yen", "pound"],
    "portfolio": ["portfolio", "holdings", "total value", "how am i doing", "cash", "positions"],
}

def router(state: AgentState) -> dict:
    safe_state = ensure_state_defaults(state)
    last_msg = safe_state["messages"][-1].content.lower() if safe_state["messages"] else ""
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in last_msg for kw in keywords):
            msg_update = {"intent": intent}
            if intent == "convert":
                if "inr" in last_msg or "rupee" in last_msg:
                    msg_update["currency"] = "INR"
                elif "eur" in last_msg or "euro" in last_msg:
                    msg_update["currency"] = "EUR"
                elif "gbp" in last_msg or "pound" in last_msg:
                    msg_update["currency"] = "GBP"
                elif "jpy" in last_msg or "yen" in last_msg:
                    msg_update["currency"] = "JPY"
            return msg_update
    return {"intent": "chat"}
