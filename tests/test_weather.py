import pytest

from tools.weather import build_weather_tool


def test_weather_tool_builder():
    """
    Test that the tool is instantiated correctly via Langchain.
    """
    tool = build_weather_tool()
    
    assert tool.name == "open_weather_map"
    assert "Weather" in tool.description or "weather" in tool.description.lower()
