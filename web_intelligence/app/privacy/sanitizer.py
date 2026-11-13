"""Sanitization of outbound search queries to minimize accidental disclosure."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{8,}\b", re.IGNORECASE)
_SECRET_KEY_RE = re.compile(
    r"\b(?:sk|tvly|BSA|exa|AIza|ghp|xoxb|pat)[-_]?[A-Za-z0-9_\-]{8,}\b",
    re.IGNORECASE,
)
_UNIX_PATH_RE = re.compile(r"(?:/Users|/home|/var|/etc)/[^\s\"']+")
_WIN_PATH_RE = re.compile(r"[A-Za-z]:\\(?:Users|Windows|Program Files)\\[^\s\"']+")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\w)")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SanitizedQueryResult:
    """Result of sanitizing a query prior to external transmission."""

    sanitized_query: str
    redactions_applied: list[str] = field(default_factory=list)
    was_modified: bool = False
    truncated: bool = False


class QuerySanitizer:
    """Strips accidental PII, secret tokens, and local filesystem paths from outbound queries."""

    def __init__(self, max_query_length: int = 300) -> None:
        self._max_length = max(32, max_query_length)

    def sanitize(self, raw_query: str) -> SanitizedQueryResult:
        """Return a sanitized outbound search string plus redaction categories."""
        text = raw_query.strip()
        redactions: list[str] = []

        replacements: list[tuple[str, re.Pattern[str], str]] = [
            ("api_token", _BEARER_RE, ""),
            ("api_token", _SECRET_KEY_RE, ""),
            ("email_address", _EMAIL_RE, ""),
            ("local_path", _UNIX_PATH_RE, ""),
            ("local_path", _WIN_PATH_RE, ""),
            ("phone_number", _PHONE_RE, ""),
        ]

        for label, pattern, repl in replacements:
            updated, count = pattern.subn(repl, text)
            if count > 0:
                if label not in redactions:
                    redactions.append(label)
                text = updated

        cleaned = _WHITESPACE_RE.sub(" ", text).strip(" ,;:")
        truncated = False
        if len(cleaned) > self._max_length:
            cleaned = cleaned[: self._max_length].rsplit(" ", 1)[0].strip() or cleaned[: self._max_length]
            truncated = True
            redactions.append("length_cap")

        return SanitizedQueryResult(
            sanitized_query=cleaned,
            redactions_applied=redactions,
            was_modified=bool(redactions) or cleaned != raw_query.strip(),
            truncated=truncated,
        )
