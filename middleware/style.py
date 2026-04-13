"""FastAPI Middleware to dynamically adjust Assistant System Prompt Persona."""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings


class StyleMiddleware(BaseHTTPMiddleware):
    """
    Reads the 'X-Response-Style' header and maps it to a specific system prompt 
    directive. Defaults to the environment configuration.
    """
    
    STYLE_MAPPINGS = {
        "expert": "Use precise technical language. Assume the user has domain expertise. Include caveats and edge cases.",
        "child": "Use simple words, short sentences, and friendly analogies suitable for a 10-year-old.",
        "casual": "Conversational tone. Use contractions. Keep it light and friendly.",
        "formal": "Professional, structured responses. Avoid colloquialisms. Be concise and authoritative."
    }
    
    async def dispatch(self, request: Request, call_next):
        style = request.headers.get("X-Response-Style", settings.DEFAULT_STYLE).lower()
        
        # Fallback to expert if an unknown style token is passed
        persona_prompt = self.STYLE_MAPPINGS.get(style, self.STYLE_MAPPINGS["expert"])
        
        request.state.persona = persona_prompt
        return await call_next(request)
