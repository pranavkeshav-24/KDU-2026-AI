"""FastAPI Middleware to automatically inject User Context into requests."""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from db.sqlite_store import get_user_profile


class UserContextMiddleware(BaseHTTPMiddleware):
    """
    Reads the 'X-User-Id' header and attaches the user's SQLite profile 
    to `request.state.user_profile`. Allows downstream LLM prompt injection.
    """
    
    async def dispatch(self, request: Request, call_next):
        uid = request.headers.get("X-User-Id", "anonymous")
        profile = await get_user_profile(uid)
        
        request.state.user_profile = profile
        return await call_next(request)
