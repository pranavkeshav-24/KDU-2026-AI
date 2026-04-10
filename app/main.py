"""FastAPI application factory with lifespan, middleware, and exception handlers."""

from __future__ import annotations

from contextlib import asynccontextmanager

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.database import engine
from app.core.exception_handlers import (
    app_exception_handler,
    generic_exception_handler,
    rate_limit_exception_handler,
    validation_exception_handler,
)
from app.core.exceptions import AppException
from app.core.logging import configure_logging, get_logger
from app.health.router import health_router
from app.users.router import auth_router, user_router

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(
        log_level=settings.LOG_LEVEL,
        json_format=settings.LOG_JSON_FORMAT,
    )
    logger.info(
        "application_starting",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT.value,
    )
    yield
    await engine.dispose()
    logger.info("application_shutdown")


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    _docs_url = "/docs" if not settings.is_production else None
    _redoc_url = "/redoc" if not settings.is_production else None

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-ready FastAPI template with JWT auth, RBAC, "
            "async PostgreSQL, structured logging, and 70%+ test coverage."
        ),
        docs_url=_docs_url,
        redoc_url=_redoc_url,
        lifespan=lifespan,
    )

    # ── Rate Limiter ──────────────────────────────────────────────────────────
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    # ── Middleware (outermost → innermost) ────────────────────────────────────
    app.add_middleware(
        CorrelationIdMiddleware,
        header_name="X-Request-ID",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ────────────────────────────────────────────────────
    app.add_exception_handler(AppException, app_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, generic_exception_handler)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
    app.include_router(user_router, prefix=settings.API_V1_PREFIX)
    app.include_router(health_router)

    return app


app = create_app()
