import os
import json
import requests

def search_web(query: str) -> str:
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

SCHEMA = {
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