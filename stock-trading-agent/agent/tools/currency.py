from langchain_core.tools import tool
from pydantic import BaseModel, Field

EXCHANGE_RATES = {"INR": 83.5, "EUR": 0.92, "GBP": 0.79, "JPY": 149.0}

class ConvertInput(BaseModel):
    amount_usd: float = Field(..., gt=0, description="Amount in USD to convert")
    target_currency: str = Field(..., description="Target ISO currency code")

@tool(args_schema=ConvertInput)
def convert_currency(amount_usd: float, target_currency: str) -> dict:
    """Convert a USD amount to the target currency."""
    code = target_currency.upper()
    if code not in EXCHANGE_RATES: return {"error": f"Unsupported currency: {code}"}
    return {"amount_usd": amount_usd, "target_currency": code, "converted_amount": round(amount_usd * EXCHANGE_RATES[code], 2), "rate": EXCHANGE_RATES[code]}