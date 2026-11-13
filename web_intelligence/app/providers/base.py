"""Search provider abstraction, exceptions, and retry utilities."""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TypeVar

from web_intelligence.app.schemas.search import NormalizedSearchResult

T = TypeVar("T")


class ProviderError(RuntimeError):
    """Base error for search provider failures."""


class ProviderAuthError(ProviderError):
    """Raised when provider API credentials are missing or rejected."""


class ProviderRateLimitError(ProviderError):
    """Raised when a search provider enforces rate limiting (HTTP 429)."""


class ProviderTimeoutError(ProviderError):
    """Raised when a search provider request exceeds the configured timeout."""


class ProviderResponseError(ProviderError):
    """Raised when a search provider returns malformed or unexpected JSON."""


def parse_optional_datetime(raw: str | None) -> datetime | None:
    """Parse an ISO/common date string safely, returning None when absent or invalid."""
    if not raw or not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    try:
        from dateutil import parser as date_parser

        dt = date_parser.parse(cleaned, fuzzy=False)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError, OverflowError):
        return None


async def execute_with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 2,
    base_delay_seconds: float = 0.25,
) -> T:
    """Retry transient provider errors (rate limits or timeouts) with exponential backoff."""
    attempt = 0
    while True:
        try:
            return await operation()
        except (ProviderRateLimitError, ProviderTimeoutError):
            if attempt >= max_retries:
                raise
            jitter = random.uniform(0.0, 0.15)
            delay = base_delay_seconds * (2**attempt) + jitter
            attempt += 1
            await asyncio.sleep(delay)


class SearchProvider(ABC):
    """Abstract provider interface normalizing search requests across backends."""

    name: str
    is_mock: bool = False

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if this provider has the credentials/setup required to run."""

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int = 6,
    ) -> list[NormalizedSearchResult]:
        """Execute a search query and return normalized results."""
