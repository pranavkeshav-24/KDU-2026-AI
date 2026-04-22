import os
import json
import requests
import ast
import operator
import math

def get_weather(location: str):
    """
    Dummy weather function or Open-Meteo API.
    For simplicity, we'll use Open-Meteo, which is free and requires no API key.
    """
    try:
        # Geocode the location
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geocode_response = requests.get(geocode_url).json()
        if "results" not in geocode_response:
            return json.dumps({"error": f"Could not find coordinates for {location}"})
            
        lat = geocode_response["results"][0]["latitude"]
        lon = geocode_response["results"][0]["longitude"]
        
        # Get weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_response = requests.get(weather_url).json()
        
        if "current_weather" in weather_response:
            current = weather_response["current_weather"]
            return json.dumps({
                "location": location,
                "temperature": current.get("temperature"),
                "windspeed": current.get("windspeed"),
                "weathercode": current.get("weathercode"),
                "unit": "Celsius"
            })
        else:
            return json.dumps({"error": "Weather data unavailable."})
    except Exception as e:
        return json.dumps({"error": str(e)})


def calculate(expression: str):
    """
    A safe calculator that evaluates basic math expressions.
    """
    try:
        # A simple and safe evaluation for math expressions, if needed we can parse ast
        # For a full production system, we'd use a robust sandbox or ast.literal_eval + restricted operators
        # Alternatively, just use Python's eval but since this is local sandbox, we'll use a simple controlled eval.
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": f"Failed to compute '{expression}': {str(e)}"})


def search_web(query: str):
    """
    Search the web using Serper.dev API.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return json.dumps({"error": "SERPER_API_KEY environment variable is missing."})
        
    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, data=payload)
        return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch current weather information for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform complex mathematical computations. Provide a valid math expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid mathematical expression, e.g. '25 * 4 + 10' or 'math.sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Retrieve relevant information based on user queries from the web. Use this for general knowledge, current events, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def execute_tool(tool_name: str, arguments: dict):
    if tool_name == "get_weather":
        return get_weather(arguments.get("location"))
    elif tool_name == "calculate":
        return calculate(arguments.get("expression"))
    elif tool_name == "search_web":
        return search_web(arguments.get("query"))
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
