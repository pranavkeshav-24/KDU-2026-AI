from langchain_core.tools import tool
from pydantic import BaseModel, Field

MOCK_PRICES = {
    "AAPL": 189.30, "GOOG": 175.20, "MSFT": 415.50,
    "TSLA": 177.80, "AMZN": 185.60, "RELIANCE": 2945.0,
    "TCS": 3820.0, "INFY": 1470.0,
}

class FetchPriceInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol in uppercase, e.g. AAPL")

@tool(args_schema=FetchPriceInput)
def fetch_stock_price(symbol: str) -> dict:
    """Fetch the current mock market price for a stock ticker."""
    symbol = symbol.upper().strip()
    if symbol not in MOCK_PRICES:
        return {"error": f"Unknown ticker: {symbol}. Available: {list(MOCK_PRICES)}"}
    return {"symbol": symbol, "price_usd": MOCK_PRICES[symbol], "currency": "USD", "source": "mock_api"}