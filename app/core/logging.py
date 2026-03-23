"""
app/core/logging.py
───────────────────
Structured logging with structlog.
JSON in production, colored console in local dev.
"""
from __future__ import annotations

import logging
import sys

import structlog

from app.core.config import Environment, settings


def setup_logging() -> None:
    """Configure structlog — call once at application startup."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.ENVIRONMENT == Environment.LOCAL and settings.LOG_FORMAT == "console":
        processors: list[structlog.types.Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Suppress noisy third-party loggers
    for noisy_lib in ("httpx", "httpcore", "qdrant_client"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
