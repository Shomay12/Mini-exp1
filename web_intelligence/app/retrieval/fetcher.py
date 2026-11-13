"""SSRF-safe asynchronous webpage fetcher with redirect, size, and rate controls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from web_intelligence.app.core.config import Settings
from web_intelligence.app.core.logging import get_logger
from web_intelligence.app.core.security import (
    AsyncResolver,
    DomainRateLimiter,
    SSRFError,
    URLValidationError,
    validate_url_for_fetch,
)
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.privacy.policy import PrivacyPolicyViolation
from web_intelligence.app.retrieval.robots import RobotsTxtManager
from web_intelligence.app.schemas.documents import ExtractionStatus

logger = get_logger("retrieval.fetcher")

_ALLOWED_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)


@dataclass
class FetchOutcome:
    """Raw HTTP fetch outcome before content extraction."""

    original_url: str
    final_url: str
    success: bool
    status: ExtractionStatus
    http_status: int | None = None
    content_type: str | None = None
    body_bytes: bytes = b""
    redirect_chain: list[str] = field(default_factory=list)
    error: str | None = None


class WebFetcher:
    """Downloads webpages safely with per-hop SSRF validation, robots.txt, and byte limits."""

    def __init__(
        self,
        settings: Settings,
        gateway: PrivacyGateway,
        *,
        robots_manager: RobotsTxtManager | None = None,
        rate_limiter: DomainRateLimiter | None = None,
        dns_resolver: AsyncResolver | None = None,
    ) -> None:
        self._settings = settings
        self._gateway = gateway
        self._dns_resolver = dns_resolver
        self._robots = robots_manager or RobotsTxtManager(
            settings, gateway, dns_resolver=dns_resolver
        )
        self._rate_limiter = rate_limiter or DomainRateLimiter(
            settings.per_host_min_interval_seconds
        )
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_fetches)

    async def fetch_many(self, urls: list[str]) -> list[FetchOutcome]:
        """Fetch multiple URLs concurrently within configured concurrency and host limits."""
        tasks = [self.fetch(url) for url in urls]
        return list(await asyncio.gather(*tasks))

    async def fetch(self, url: str) -> FetchOutcome:
        """Fetch a single URL safely, enforcing SSRF checks on initial URL and every redirect."""
        async with self._semaphore:
            return await self._fetch_guarded(url)

    async def _fetch_guarded(self, url: str) -> FetchOutcome:
        redirect_chain: list[str] = []
        current_url = url.strip()

        try:
            async with self._gateway.create_http_client(is_external=True) as client:
                for hop in range(self._settings.max_redirects + 1):
                    try:
                        clean_url, _ = await validate_url_for_fetch(
                            current_url,
                            allowed_ports=self._settings.allowed_ports,
                            resolver=self._dns_resolver,
                        )
                    except (SSRFError, URLValidationError) as exc:
                        return FetchOutcome(
                            original_url=url,
                            final_url=current_url,
                            success=False,
                            status=ExtractionStatus.BLOCKED_SSRF,
                            redirect_chain=redirect_chain,
                            error=str(exc),
                        )

                    allowed, robots_reason = await self._robots.is_allowed(clean_url)
                    if not allowed:
                        return FetchOutcome(
                            original_url=url,
                            final_url=clean_url,
                            success=False,
                            status=ExtractionStatus.BLOCKED_ROBOTS,
                            redirect_chain=redirect_chain,
                            error=f"Blocked by robots.txt ({robots_reason}).",
                        )

                    await self._rate_limiter.acquire(clean_url)

                    try:
                        async with client.stream("GET", clean_url) as response:
                            if response.status_code in (301, 302, 303, 307, 308):
                                location = response.headers.get("Location")
                                if not location:
                                    return FetchOutcome(
                                        original_url=url,
                                        final_url=clean_url,
                                        success=False,
                                        status=ExtractionStatus.HTTP_ERROR,
                                        http_status=response.status_code,
                                        redirect_chain=redirect_chain,
                                        error="Redirect response missing Location header.",
                                    )
                                if hop >= self._settings.max_redirects:
                                    return FetchOutcome(
                                        original_url=url,
                                        final_url=clean_url,
                                        success=False,
                                        status=ExtractionStatus.HTTP_ERROR,
                                        http_status=response.status_code,
                                        redirect_chain=redirect_chain,
                                        error=f"Exceeded max redirects ({self._settings.max_redirects}).",
                                    )
                                next_url = urljoin(clean_url, location)
                                redirect_chain.append(clean_url)
                                current_url = next_url
                                continue

                            if response.status_code >= 400:
                                return FetchOutcome(
                                    original_url=url,
                                    final_url=clean_url,
                                    success=False,
                                    status=ExtractionStatus.HTTP_ERROR,
                                    http_status=response.status_code,
                                    redirect_chain=redirect_chain,
                                    error=f"HTTP error {response.status_code}.",
                                )

                            raw_ct = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                            if raw_ct and not any(raw_ct.startswith(allowed_ct) for allowed_ct in _ALLOWED_CONTENT_TYPES):
                                return FetchOutcome(
                                    original_url=url,
                                    final_url=clean_url,
                                    success=False,
                                    status=ExtractionStatus.UNSUPPORTED_CONTENT_TYPE,
                                    http_status=response.status_code,
                                    content_type=raw_ct,
                                    redirect_chain=redirect_chain,
                                    error=f"Unsupported Content-Type: {raw_ct}",
                                )

                            content_length = response.headers.get("Content-Length")
                            if content_length and content_length.isdigit():
                                if int(content_length) > self._settings.max_download_bytes:
                                    return FetchOutcome(
                                        original_url=url,
                                        final_url=clean_url,
                                        success=False,
                                        status=ExtractionStatus.TOO_LARGE,
                                        http_status=response.status_code,
                                        content_type=raw_ct or None,
                                        redirect_chain=redirect_chain,
                                        error=f"Content-Length {content_length} exceeds limit.",
                                    )

                            chunks: list[bytes] = []
                            total_bytes = 0
                            async for chunk in response.aiter_bytes():
                                total_bytes += len(chunk)
                                if total_bytes > self._settings.max_download_bytes:
                                    return FetchOutcome(
                                        original_url=url,
                                        final_url=clean_url,
                                        success=False,
                                        status=ExtractionStatus.TOO_LARGE,
                                        http_status=response.status_code,
                                        content_type=raw_ct or None,
                                        redirect_chain=redirect_chain,
                                        error=f"Response exceeded {self._settings.max_download_bytes} byte limit.",
                                    )
                                chunks.append(chunk)

                            return FetchOutcome(
                                original_url=url,
                                final_url=clean_url,
                                success=True,
                                status=ExtractionStatus.SUCCESS,
                                http_status=response.status_code,
                                content_type=raw_ct or None,
                                body_bytes=b"".join(chunks),
                                redirect_chain=redirect_chain,
                            )
                    except httpx.TimeoutException as exc:
                        return FetchOutcome(
                            original_url=url,
                            final_url=clean_url,
                            success=False,
                            status=ExtractionStatus.TIMEOUT,
                            redirect_chain=redirect_chain,
                            error=f"Request timed out: {exc}",
                        )
                    except httpx.HTTPError as exc:
                        return FetchOutcome(
                            original_url=url,
                            final_url=clean_url,
                            success=False,
                            status=ExtractionStatus.FAILED,
                            redirect_chain=redirect_chain,
                            error=f"HTTP client error: {exc}",
                        )
        except PrivacyPolicyViolation as exc:
            return FetchOutcome(
                original_url=url,
                final_url=current_url,
                success=False,
                status=ExtractionStatus.FAILED,
                redirect_chain=redirect_chain,
                error=str(exc),
            )

        return FetchOutcome(
            original_url=url,
            final_url=current_url,
            success=False,
            status=ExtractionStatus.FAILED,
            redirect_chain=redirect_chain,
            error="Unexpected redirect loop termination.",
        )
