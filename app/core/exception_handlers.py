"""Global FastAPI exception handlers — converts all errors to a standardised response."""

from __future__ import annotations

from asgi_correlation_id import correlation_id
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.core.exceptions import AppException
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Response Builder ──────────────────────────────────────────────────────────

def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: list | None = None,
) -> JSONResponse:
    """Build a standardised JSON error body."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details,
                "request_id": correlation_id.get() or "",
            }
        },
    )


# ── Handlers ──────────────────────────────────────────────────────────────────

async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    logger.warning(
        "application_error",
        error_code=exc.error_code,
        message=exc.message,
        path=str(request.url),
    )
    return _error_response(exc.status_code, exc.error_code, exc.message)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = [
        {"field": " → ".join(str(loc) for loc in err["loc"]), "message": err["msg"]}
        for err in exc.errors()
    ]
    logger.warning(
        "validation_error",
        path=str(request.url),
        details=details,
    )
    return _error_response(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "VALIDATION_ERROR",
        "Request validation failed",
        details,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        path=str(request.url),
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "INTERNAL_ERROR",
        "An internal server error occurred",  # never expose internal details
    )


async def rate_limit_exception_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    logger.warning("rate_limit_exceeded", path=str(request.url), limit=str(exc.detail))
    return _error_response(
        status.HTTP_429_TOO_MANY_REQUESTS,
        "RATE_LIMIT_EXCEEDED",
        f"Rate limit exceeded: {exc.detail}",
    )
