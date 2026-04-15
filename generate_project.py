import os

files = {
    "stock-trading-agent/requirements.txt": """langgraph>=0.2.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langsmith>=0.2.0
openai>=1.40.0
streamlit>=1.38.0
pydantic>=2.7.0
python-dotenv>=1.0.0
pytest>=8.0.0
""",
    "stock-trading-agent/.env.example": """OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxxxxxxxxxxx
LANGCHAIN_PROJECT="stock-trading-agent"
INITIAL_CASH_USD=10000.0
SQLITE_DB_PATH=checkpoints.db
""",
    "stock-trading-agent/agent/state.py": """from typing import Annotated, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class Portfolio(TypedDict):
    holdings: dict[str, dict]
    cash_usd: float
    total_value_usd: float

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    portfolio: Portfolio
    intent: Optional[str]
    currency: Optional[str]
    pending_order: Optional[dict]
    human_approved: Optional[bool]
    total_tokens: int
    total_cost_usd: float

def initialize_state(cash_usd=10000.0) -> AgentState:
    return {
        'messages': [],
        'portfolio': {'holdings': {}, 'cash_usd': cash_usd, 'total_value_usd': cash_usd},
        'intent': 'chat',
        'currency': None,
        'pending_order': None,
        'human_approved': None,
        'total_tokens': 0,
        'total_cost_usd': 0.0
    }
""",
    "stock-trading-agent/agent/tools/stock_price.py": """from langchain_core.tools import tool
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
    \"\"\"Fetch the current mock market price for a stock ticker.\"\"\"
    symbol = symbol.upper().strip()
    if symbol not in MOCK_PRICES:
        return {"error": f"Unknown ticker: {symbol}. Available: {list(MOCK_PRICES)}"}
    return {"symbol": symbol, "price_usd": MOCK_PRICES[symbol], "currency": "USD", "source": "mock_api"}
""",
    "stock-trading-agent/agent/tools/currency.py": """from langchain_core.tools import tool
from pydantic import BaseModel, Field

EXCHANGE_RATES = {"INR": 83.5, "EUR": 0.92, "GBP": 0.79, "JPY": 149.0}

class ConvertInput(BaseModel):
    amount_usd: float = Field(..., gt=0, description="Amount in USD to convert")
    target_currency: str = Field(..., description="Target ISO currency code")

@tool(args_schema=ConvertInput)
def convert_currency(amount_usd: float, target_currency: str) -> dict:
    \"\"\"Convert a USD amount to the target currency.\"\"\"
    code = target_currency.upper()
    if code not in EXCHANGE_RATES: return {"error": f"Unsupported currency: {code}"}
    return {"amount_usd": amount_usd, "target_currency": code, "converted_amount": round(amount_usd * EXCHANGE_RATES[code], 2), "rate": EXCHANGE_RATES[code]}
""",
    "stock-trading-agent/agent/tools/portfolio.py": """from langchain_core.tools import tool
from .stock_price import fetch_stock_price

@tool
def calculate_portfolio(holdings: dict, cash_usd: float) -> dict:
    \"\"\"Compute total portfolio value given holdings and cash.\"\"\"
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
""",
    "stock-trading-agent/agent/nodes/__init__.py": "",
    "stock-trading-agent/agent/tools/__init__.py": "from .stock_price import fetch_stock_price\nfrom .currency import convert_currency\nfrom .portfolio import calculate_portfolio\nALL_TOOLS = [fetch_stock_price, convert_currency, calculate_portfolio]",
    "stock-trading-agent/agent/nodes/router.py": """from ..state import AgentState

INTENT_KEYWORDS = {
    "buy": ["buy", "purchase", "acquire"],
    "sell": ["sell", "offload", "dump"],
    "price": ["price", "quote", "how much", "cost of"],
    "convert": ["inr", "eur", "convert", "rupee", "euro"],
    "portfolio": ["portfolio", "holdings", "total value", "how am i doing"],
}

def router(state: AgentState) -> dict:
    last_msg = state['messages'][-1].content.lower() if state['messages'] else ""
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in last_msg for kw in keywords):
            msg_update = {'intent': intent}
            if intent == 'convert':
                msg_update['currency'] = 'INR' if 'inr' in last_msg else 'EUR'
            return msg_update
    return {'intent': 'chat'}
""",
    "stock-trading-agent/agent/nodes/llm_agent.py": """import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, RemoveMessage
from ..state import AgentState
from ..tools import ALL_TOOLS

MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

# RATE LIMITING: Simple in-memory rate limiter tracker
last_call_time = 0

def get_llm_with_fallback() -> ChatOpenAI:
    global last_call_time
    # RATE LIMITING: ensure at least 1s between LLM calls to avoid API limits on free tiers
    elapsed = time.time() - last_call_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
        
    for model in MODELS:
        try:
            llm = ChatOpenAI(
                model=model,
                base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
                api_key=os.getenv('OPENROUTER_API_KEY', 'dummy-key'),
                temperature=0.1,
                max_retries=1
            )
            # test basic config
            last_call_time = time.time()
            return llm
        except Exception:
            continue
    raise RuntimeError('All LLM models unavailable.')

def llm_agent(state: AgentState) -> dict:
    llm = get_llm_with_fallback()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    # CONTEXT COMPACTION: Ensure we don't pass unbounded message history to free LLMs
    messages = state['messages']
    messages_to_remove = []
    
    # KEEP only last 10 messages (plus system) to prevent context window overflow
    if len(messages) > 10:
        for m in messages[:-10]:
            messages_to_remove.append(RemoveMessage(id=m.id))
        messages = messages[-10:]
            
    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        # Provide fallback safe response on error
        from langchain_core.messages import AIMessage
        response = AIMessage(content=f"Error contacting LLM: {str(e)}")

    usage = response.response_metadata.get('token_usage', {}) if hasattr(response, 'response_metadata') else {}
    tokens = usage.get('total_tokens', 0)
    
    pending = None
    if response.tool_calls:
        # Check if doing a stock buy natively via tools pending
        # Here we just parse tool calls; Actual HITL happens if buy is matched
        for tc in response.tool_calls:
            pass # normal logic handled by graph routing

    return {
        'messages': messages_to_remove + [response],
        'total_tokens': state.get('total_tokens', 0) + tokens,
    }
""",
    "stock-trading-agent/agent/nodes/approval.py": """from langchain_core.messages import AIMessage
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
""",
    "stock-trading-agent/agent/nodes/currency.py": """from langchain_core.messages import AIMessage
from ..state import AgentState
from ..tools.currency import convert_currency

def currency_converter(state: AgentState) -> dict:
    # Mock implementation of direct tool access based on extracted intent
    return {'messages': [AIMessage(content=f"Currency converter triggered intent: {state.get('currency')}")]}
""",
    "stock-trading-agent/agent/nodes/portfolio.py": """from langchain_core.messages import AIMessage
from ..state import AgentState
from ..tools.portfolio import calculate_portfolio

def portfolio_calculator(state: AgentState) -> dict:
    res = calculate_portfolio.invoke({'holdings': state['portfolio']['holdings'], 'cash_usd': state['portfolio']['cash_usd']})
    msg = f"Portfolio Total: ${res['total_value_usd']} | Cash: ${res['cash_usd']}"
    return {'messages': [AIMessage(content=msg)]}
""",
    "stock-trading-agent/agent/nodes/responder.py": """from ..state import AgentState

def responder(state: AgentState) -> dict:
    # Placeholder: The LLM or graph directly responds
    return {}
""",
    "stock-trading-agent/agent/graph.py": """from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from .state import AgentState
from .nodes.router import router
from .nodes.llm_agent import llm_agent
from .nodes.currency import currency_converter
from .nodes.portfolio import portfolio_calculator
from .nodes.approval import human_approval, execute_trade
from .nodes.responder import responder
from .tools import ALL_TOOLS

def route_intent(state: AgentState) -> str:
    return state.get('intent', 'chat')

def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # Check if buy exists
        for call in last_message.tool_calls:
            if call['name'] == 'buy_stock': 
                return 'buy_done'
        return 'tools'
    return 'end'

def build_graph(db_path='checkpoints.db'):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    builder = StateGraph(AgentState)
    
    builder.add_node('router', router)
    builder.add_node('llm_agent', llm_agent)
    builder.add_node('tool_executor', ToolNode(ALL_TOOLS))
    builder.add_node('currency_converter', currency_converter)
    builder.add_node('portfolio_calculator', portfolio_calculator)
    builder.add_node('human_approval', human_approval)
    builder.add_node('execute_trade', execute_trade)
    builder.add_node('responder', responder)
    
    builder.set_entry_point('router')
    
    builder.add_conditional_edges('router', route_intent, {
        'convert': 'currency_converter',
        'portfolio': 'portfolio_calculator',
        'buy': 'llm_agent',
        'sell': 'llm_agent',
        'price': 'llm_agent',
        'chat': 'llm_agent',
    })
    
    builder.add_conditional_edges('llm_agent', should_continue, {
        'tools': 'tool_executor',
        'buy_done': 'human_approval',
        'end': 'responder',
    })
    
    builder.add_edge('tool_executor', 'llm_agent')
    builder.add_edge('human_approval', 'execute_trade')
    builder.add_edge('execute_trade', 'responder')
    builder.add_edge('currency_converter', 'responder')
    builder.add_edge('portfolio_calculator', 'responder')
    builder.add_edge('responder', END)
    
    return builder.compile(
        checkpointer=memory,
        interrupt_before=['execute_trade']
    )
""",
    "stock-trading-agent/app.py": """import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from agent.graph import build_graph

st.set_page_config(page_title="Stock Trading Agent", layout="wide")

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    
# Init graph
@st.cache_resource
def get_graph():
    return build_graph()

graph = get_graph()
config = {'configurable': {'thread_id': st.session_state.thread_id}}

st.title("🤖 Stock Trading Agent")

# Sidebar
with st.sidebar:
    if st.button("New Session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.header("📊 Portfolio Summary")
    state_snap = graph.get_state(config)
    if state_snap and 'portfolio' in state_snap.values:
        port = state_snap.values['portfolio']
        st.write(f"Cash: ${port.get('cash_usd', 10000.0):.2f}")
    
    st.header("🔭 Observability")
    if state_snap and 'total_tokens' in state_snap.values:
        st.write(f"Tokens: {state_snap.values.get('total_tokens', 0)}")

# Main chat
user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").write(user_input)
    try:
        result = graph.invoke({'messages': [HumanMessage(content=user_input)]}, config=config)
        st.chat_message("agent").write(result['messages'][-1].content)
    except Exception as e:
        if "GraphInterrupted" in str(type(e)):
            st.warning("⚠️ APPROVAL REQUIRED for Trade")
            col1, col2 = st.columns(2)
            if col1.button("✅ Approve"):
                result = graph.invoke({"human_approved": True}, config=config)
                st.chat_message("agent").write(result['messages'][-1].content)
                st.rerun()
            if col2.button("❌ Reject"):
                result = graph.invoke({"human_approved": False}, config=config)
                st.chat_message("agent").write(result['messages'][-1].content)
                st.rerun()
        else:
            st.error(f"Error: {e}")
"""
}

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.strip())
print("All files created successfully.")
