"""Base64 image encoding for Multimodal LLM prompts."""

import base64

from langchain_core.messages import HumanMessage


def build_vision_message(image_bytes: bytes, user_text: str) -> HumanMessage:
    """
    Constructs a HumanMessage formatted for Langchain's multimodal support.
    It takes the raw bytes, base64 encodes it, and creates a dual-block output payload.
    """
    # Ensure it's a decodable valid string directly from bytes
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    return HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    # Usually LLMs specify mime type. Adjust 'jpeg' if dynamically analyzing.
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            },
            {
                "type": "text", 
                "text": user_text
            },
        ]
    )
