"""Structured logging via ``structlog``.

Call :func:`configure_logging` once at the entry point of any script. Library
modules only call :func:`get_logger` and never configure anything, so importing
them has no global side effects.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import structlog

from config import get_settings

__all__ = ["configure_logging", "get_logger"]

_CONFIGURED = False


def configure_logging(level: Optional[str] = None, json_output: Optional[bool] = None) -> None:
    """Configure structlog for the current process (idempotent).

    ``json_output`` defaults to the ``PDIFF_LOG_JSON`` setting: human-readable
    console output during development, JSON on the cluster where logs are
    shipped to files and parsed.
    """
    global _CONFIGURED
    settings = get_settings()
    resolved_level = (level or settings.log_level).upper()
    resolved_json = settings.log_json if json_output is None else json_output

    logging.basicConfig(format="%(message)s", stream=sys.stdout,
                        level=getattr(logging, resolved_level, logging.INFO))
    renderer: Any = (structlog.processors.JSONRenderer() if resolved_json
                     else structlog.dev.ConsoleRenderer(colors=False))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, resolved_level, logging.INFO)),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> Any:
    """Return a bound structlog logger, configuring defaults on first use."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
