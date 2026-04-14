import re
from collections.abc import Iterable

from .state import PendingOrder

NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
INTEGER_PATTERN = re.compile(r"\b\d+\b")
WORD_PATTERN = re.compile(r"[A-Z]{2,10}")


def extract_symbol(text: str, candidates: Iterable[str]) -> str | None:
    candidate_set = {candidate.upper() for candidate in candidates}
    for token in WORD_PATTERN.findall(text.upper()):
        if token in candidate_set:
            return token
    return None


def extract_currency(text: str, supported_codes: Iterable[str]) -> str | None:
    supported = {code.upper() for code in supported_codes}
    for token in WORD_PATTERN.findall(text.upper()):
        if token in supported:
            return token
    return None


def extract_first_float(text: str) -> float | None:
    match = NUMBER_PATTERN.search(text)
    if not match:
        return None
    return float(match.group(0))


def extract_first_int(text: str) -> int | None:
    match = INTEGER_PATTERN.search(text)
    if not match:
        return None
    return int(match.group(0))


def format_usd(amount: float) -> str:
    return f"${amount:,.2f}"


def format_currency_amount(amount: float, currency_code: str) -> str:
    code = currency_code.upper()
    if code == "USD":
        return format_usd(amount)
    return f"{code} {amount:,.2f}"


def describe_order(order: PendingOrder) -> str:
    verb = "Buy" if order["action"] == "buy" else "Sell"
    return (
        f"{verb} {order['qty']} share(s) of {order['symbol']} "
        f"at {format_usd(order['price'])} for {format_usd(order['notional'])}"
    )
