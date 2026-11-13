"""Structured JSON logging with automatic secret redaction."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

_TOKEN_PATTERN = re.compile(
    r"(?:Bearer\s+[A-Za-z0-9._\-]{8,}|"
    r"(?:sk|tvly|BSA|exa|AIza|ghp)_[A-Za-z0-9_\-]{6,}|"
    r"(?:BSA|tvly-)[A-Za-z0-9_\-]{8,})",
    re.IGNORECASE,
)


class SecretRedactingFilter(logging.Filter):
    """Mask configured secrets and bearer/token patterns in log records."""

    def __init__(self, secrets: list[str] | None = None) -> None:
        super().__init__()
        self._secrets = [s for s in (secrets or []) if s]

    def update_secrets(self, secrets: list[str]) -> None:
        """Replace tracked secret literals."""
        self._secrets = [s for s in secrets if s]

    def _redact(self, text: str) -> str:
        masked = _TOKEN_PATTERN.sub("[REDACTED_SECRET]", text)
        for secret in self._secrets:
            if secret in masked:
                masked = masked.replace(secret, "[REDACTED_SECRET]")
        return masked

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._redact(a) if isinstance(a, str) else a for a in record.args
                )
        extra_data = getattr(record, "extra_data", None)
        if isinstance(extra_data, dict):
            record.extra_data = {
                k: self._redact(v) if isinstance(v, str) else v
                for k, v in extra_data.items()
            }
        return True


class StructuredJsonFormatter(logging.Formatter):
    """Emit compact single-line JSON log records."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_data = getattr(record, "extra_data", None)
        if isinstance(extra_data, dict) and extra_data:
            payload["context"] = extra_data
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_SECRET_FILTER = SecretRedactingFilter()


def configure_logging(level: str = "INFO", secrets: list[str] | None = None) -> None:
    """Configure root logger with structured JSON formatting and secret redaction."""
    _SECRET_FILTER.update_secrets(secrets or [])
    root = logging.getLogger("web_intelligence")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJsonFormatter())
    handler.addFilter(_SECRET_FILTER)
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger namespaced under web_intelligence."""
    return logging.getLogger(f"web_intelligence.{name}")
