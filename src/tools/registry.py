import json

from src.tools.weather import get_weather, SCHEMA as WEATHER_SCHEMA
from src.tools.calculator import calculate, SCHEMA as CALCULATOR_SCHEMA
from src.tools.search import search_web, SCHEMA as SEARCH_SCHEMA

# Aggregate schemas for the LLM
TOOL_SCHEMAS = [
    WEATHER_SCHEMA,
    CALCULATOR_SCHEMA,
    SEARCH_SCHEMA
]

def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Executes the appropriate tool function based on the tool name.
    """
    if tool_name == "get_weather":
        return get_weather(arguments.get("location"))
    elif tool_name == "calculate":
        return calculate(arguments.get("expression"))
    elif tool_name == "search_web":
        return search_web(arguments.get("query"))
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})