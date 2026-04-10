"""Structured logging configuration via structlog."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO", json_format: bool = True) -> None:
    """Configure structlog with shared processors and the appropriate renderer.

    Call this once at application startup (inside the lifespan handler).

    Args:
        log_level:   Standard Python log level name (DEBUG, INFO, WARNING, ERROR).
        json_format: True → JSON output (production); False → coloured console (dev).
    """
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,  # injects request_id, user_id, etc.
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)  # type: ignore[assignment]

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to funnel through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(log_level.upper()),
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Return a structlog bound logger (drop-in replacement for logging.getLogger)."""
    return structlog.get_logger(name)
