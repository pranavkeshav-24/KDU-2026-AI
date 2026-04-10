"""Health check endpoints — no authentication required."""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import settings
from app.core.database import async_session

health_router = APIRouter(tags=["Health"])


@health_router.get(
    "/health",
    summary="Application health check",
    response_description="Application is healthy",
)
async def health() -> dict:
    """Returns application status and version. Always returns 200 if the app is running."""
    return {"status": "healthy", "version": settings.APP_VERSION}


@health_router.get(
    "/health/db",
    summary="Database connectivity check",
    responses={503: {"description": "Database unavailable"}},
)
async def health_db() -> dict:
    """Executes a lightweight ``SELECT 1`` to verify database connectivity."""
    try:
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "database": str(exc)},
        )
