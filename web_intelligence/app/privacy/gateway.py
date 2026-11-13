"""Centralized outbound HTTP gateway supporting proxies and query disclosure logging."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from web_intelligence.app.core.config import Settings
from web_intelligence.app.core.logging import get_logger
from web_intelligence.app.privacy.policy import PrivacyPolicy
from web_intelligence.app.privacy.sanitizer import QuerySanitizer, SanitizedQueryResult

logger = get_logger("privacy.gateway")


@dataclass(frozen=True)
class OutboundDisclosure:
    """Record of an outbound search query sent to an external provider."""

    provider: str
    sanitized_query: str
    redactions_applied: list[str]
    disclosed_at: datetime
    is_external: bool


class PrivacyGateway:
    """Manages outbound HTTP clients, proxy settings, query sanitization, and disclosures."""

    def __init__(
        self,
        settings: Settings,
        *,
        sanitizer: QuerySanitizer | None = None,
        policy: PrivacyPolicy | None = None,
        transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        self._settings = settings
        self.sanitizer = sanitizer or QuerySanitizer()
        self.policy = policy or PrivacyPolicy.from_settings(settings)
        self._transport_factory = transport_factory

    def prepare_search_query(self, raw_query: str) -> SanitizedQueryResult:
        """Sanitize a search query before sending it to any search provider."""
        return self.sanitizer.sanitize(raw_query)

    def record_search_disclosure(
        self,
        *,
        provider: str,
        sanitized: SanitizedQueryResult,
        is_external: bool,
    ) -> OutboundDisclosure:
        """Create an explicit disclosure entry for an outbound search request."""
        disclosure = OutboundDisclosure(
            provider=provider,
            sanitized_query=sanitized.sanitized_query,
            redactions_applied=list(sanitized.redactions_applied),
            disclosed_at=datetime.now(timezone.utc),
            is_external=is_external,
        )
        if self._settings.disclose_outbound_queries and is_external:
            logger.info(
                "Outbound search query dispatched",
                extra={
                    "extra_data": {
                        "provider": provider,
                        "redactions": sanitized.redactions_applied,
                        "proxy_active": bool(self.policy.proxy_url),
                    }
                },
            )
        return disclosure

    def create_http_client(
        self,
        *,
        is_external: bool = True,
        timeout_seconds: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.AsyncClient:
        """Create a tightly configured AsyncClient with proxy/timeout/TLS safeguards."""
        if is_external and self._transport_factory is None:
            self.policy.assert_live_network_allowed("outbound HTTP request")

        timeout = httpx.Timeout(
            timeout=timeout_seconds or self._settings.request_timeout_seconds,
            connect=self._settings.connect_timeout_seconds,
        )
        headers = {
            "User-Agent": self._settings.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if extra_headers:
            headers.update(extra_headers)

        client_kwargs: dict[str, object] = {
            "timeout": timeout,
            "follow_redirects": False,
            "trust_env": False,
            "verify": True,
            "headers": headers,
        }
        if self._transport_factory is not None:
            client_kwargs["transport"] = self._transport_factory()
        elif self.policy.proxy_url:
            client_kwargs["proxy"] = self.policy.proxy_url

        return httpx.AsyncClient(**client_kwargs)  # type: ignore[arg-type]
