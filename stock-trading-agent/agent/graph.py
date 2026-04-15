from langgraph.graph import StateGraph, END
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