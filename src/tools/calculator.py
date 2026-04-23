import json
import math

def calculate(expression: str) -> str:
    """
    A safe calculator that evaluates basic math expressions.
    """
    try:
        # A simple and safe evaluation for math expressions.
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": f"Failed to compute '{expression}': {str(e)}"})

SCHEMA = {
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
}