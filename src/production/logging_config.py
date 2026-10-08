"""Structured (JSON-lines) logging with secret redaction."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

LOGGER_NAME = "freight_rate"
REDACTED = "***REDACTED***"
SENSITIVE_KEYS = ("api_key", "apikey", "password", "passwd", "secret", "token", "authorization", "credential")

_STANDARD_ATTRS = set(logging.LogRecord("x", 0, "x", 0, "x", (), None).__dict__) | {"message", "asctime"}


def _is_sensitive(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(s in lowered for s in SENSITIVE_KEYS)


def redact(value: Any) -> Any:
    """Recursively mask values stored under sensitive-looking keys."""
    if isinstance(value, dict):
        return {k: (REDACTED if _is_sensitive(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            payload[key] = REDACTED if _is_sensitive(key) else redact(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)  # server-side logs only
        return json.dumps(payload, default=str)


def configure_logging(level: Optional[str] = None) -> logging.Logger:
    """Idempotently configure the project logger (JSON to stderr)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel((level or "INFO").upper())
    if not any(getattr(h, "_freight_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        handler._freight_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
