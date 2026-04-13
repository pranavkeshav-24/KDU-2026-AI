"""LangChain weather tool wrapper with a graceful offline fallback."""

from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain_core.tools import Tool

from config import settings


def _weather_unavailable(_: str) -> str:
    return (
        "Weather lookup is unavailable right now because "
        "OPENWEATHERMAP_API_KEY is not configured."
    )


def build_weather_tool():
    """
    Return the live weather tool when configured, otherwise provide a safe
    placeholder tool so local development and tests stay functional.
    """
    description = (
        "Weather lookup tool. Use it to retrieve current weather conditions, "
        "temperature, humidity, and forecast context."
    )

    if not settings.OPENWEATHERMAP_API_KEY:
        return Tool(
            name="open_weather_map",
            description=description,
            func=_weather_unavailable,
        )

    try:
        wrapper = OpenWeatherMapAPIWrapper(
            openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY
        )
        return OpenWeatherMapQueryRun(api_wrapper=wrapper)
    except Exception:
        return Tool(
            name="open_weather_map",
            description=description,
            func=_weather_unavailable,
        )
