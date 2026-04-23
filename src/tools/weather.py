import os
import json
import requests

def get_weather(location: str) -> str:
    """
    Fetch weather using the free Open-Meteo API.
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
        return json.dumps({"error": "Weather data unavailable."})
    except Exception as e:
        return json.dumps({"error": str(e)})

SCHEMA = {
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
}