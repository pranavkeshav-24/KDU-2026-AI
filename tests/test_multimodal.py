import pytest

from tools.image_analysis import build_vision_message
from models.router import classify_task
from models.schemas import ChatRequest


def test_build_vision_message():
    """
    Validates base64 packaging output mapping required by gemini and openai models
    """
    fake_img = b"fakeimagebytes"
    user_text = "What is this?"
    
    msg_payload = build_vision_message(fake_img, user_text)
    
    assert len(msg_payload.content) == 2
    
    # Assert Base64 block format injected
    img_block = msg_payload.content[0]
    assert img_block["type"] == "image_url"
    assert "base64,ZmFrZWltYWdlYnl0ZXM=" in img_block["image_url"]["url"]
    
    # Assert Text block
    text_block = msg_payload.content[1]
    assert text_block["type"] == "text"
    assert text_block["text"] == "What is this?"


def test_router_classification():
    """
    Validates non-LLM dependency heuristics mapping
    """
    # Vision
    req = ChatRequest(message="test", thread_id="1")
    assert classify_task(req, has_image=True) == "vision"
    
    # Structured
    req = ChatRequest(message="What is the weather outside?", thread_id="1")
    assert classify_task(req, has_image=False) == "structured"
    
    # Fast
    req = ChatRequest(message="Hi bot", thread_id="1")
    assert classify_task(req, has_image=False) == "fast"
    
    # Reasoning
    req = ChatRequest(message="Please explain the fundamental nature of quantum entanglement deeply", thread_id="1")
    assert classify_task(req, has_image=False) == "reasoning"
