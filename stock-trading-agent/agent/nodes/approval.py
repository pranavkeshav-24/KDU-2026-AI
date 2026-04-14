from langchain_core.messages import AIMessage
from ..state import AgentState

def human_approval(state: AgentState) -> dict:
    # Look for pending_order extracted in tools or router, normally set beforehand
    return {} # handled by intercept

def execute_trade(state: AgentState) -> dict:
    if not state.get('human_approved'):
        return {'messages': [AIMessage(content='Trade cancelled by user.')], 'human_approved': None, 'pending_order': None}
        
    order = state.get('pending_order', {})
    if not order:
        return {'messages': [AIMessage(content='No pending order found.')], 'human_approved': None}
        
    cost = order['qty'] * order['price']
    cash = state['portfolio']['cash_usd']
    if cost > cash:
        return {'messages': [AIMessage(content='Insufficient funds.')], 'human_approved': None, 'pending_order': None}
        
    holdings = dict(state['portfolio']['holdings'])
    sym = order['symbol']
    existing = holdings.get(sym, {'qty': 0, 'avg_cost': order['price']})
    new_qty = existing['qty'] + order['qty']
    new_avg = (existing['qty']*existing['avg_cost'] + order['qty']*order['price']) / new_qty
    holdings[sym] = {'qty': new_qty, 'avg_cost': round(new_avg, 2)}
    new_cash = cash - cost
    
    return {
        'portfolio': {**state['portfolio'], 'holdings': holdings, 'cash_usd': new_cash},
        'pending_order': None,
        'human_approved': None,
        'messages': [AIMessage(content=f'Bought {order["qty"]} x {sym} @ ${order["price"]}')]
    }