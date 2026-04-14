from langchain_core.tools import tool
from .stock_price import fetch_stock_price

@tool
def calculate_portfolio(holdings: dict, cash_usd: float) -> dict:
    """Compute total portfolio value given holdings and cash."""
    total = cash_usd
    breakdown = []
    for symbol, data in holdings.items():
        price_result = fetch_stock_price.invoke({'symbol': symbol})
        if 'error' in price_result: continue
        market_val = price_result['price_usd'] * data['qty']
        total += market_val
        breakdown.append({
            'symbol': symbol, 'qty': data['qty'],
            'market_price': price_result['price_usd'],
            'market_value': round(market_val, 2),
            'avg_cost': data['avg_cost'],
            'pnl': round(market_val - data['avg_cost'] * data['qty'], 2),
        })
    return {'total_value_usd': round(total, 2), 'cash_usd': cash_usd, 'breakdown': breakdown}