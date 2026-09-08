"""structlog configuration.

JSON to stdout so a systemd/Docker log collector can parse it; optional
human-readable console rendering for interactive runs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

from tradebot.config import PROJECT_ROOT

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    json_logs: bool = True,
    log_file: str | Path | None = None,
) -> None:
    """Idempotent. Safe to call from every entry point."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        path = Path(log_file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )
    # Alpaca's HTTP stack is chatty at INFO and leaks request URLs.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Logger for a module. Call configure_logging() first."""
    return structlog.get_logger(name)
