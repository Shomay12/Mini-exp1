"""SSRF-safe robots.txt fetcher and per-origin policy cache."""

from __future__ import annotations

from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from web_intelligence.app.core.config import Settings
from web_intelligence.app.core.logging import get_logger
from web_intelligence.app.core.security import (
    AsyncResolver,
    SSRFError,
    URLValidationError,
    validate_url_for_fetch,
)
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.privacy.policy import PrivacyPolicyViolation

logger = get_logger("retrieval.robots")

_MAX_ROBOTS_BYTES = 256_000


class RobotsTxtManager:
    """Checks and caches robots.txt crawl permissions safely per origin."""

    def __init__(
        self,
        settings: Settings,
        gateway: PrivacyGateway,
        *,
        dns_resolver: AsyncResolver | None = None,
    ) -> None:
        self._settings = settings
        self._gateway = gateway
        self._dns_resolver = dns_resolver
        self._cache: dict[str, tuple[RobotFileParser | None, str]] = {}

    async def is_allowed(self, url: str) -> tuple[bool, str]:
        """Return (allowed, reason) for fetching the given URL."""
        if not self._settings.respect_robots_txt:
            return True, "robots_enforcement_disabled"

        parsed = urlparse(url)
        origin = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
        cached = self._cache.get(origin)
        if cached is not None:
            parser, policy_note = cached
            if parser is None:
                return policy_note != "disallow_all", policy_note
            allowed = parser.can_fetch(self._settings.user_agent, url)
            return allowed, ("allowed" if allowed else "disallowed_by_robots_txt")

        robots_url = urljoin(origin + "/", "robots.txt")
        parser, policy_note = await self._fetch_and_parse(robots_url)
        self._cache[origin] = (parser, policy_note)
        if parser is None:
            return policy_note != "disallow_all", policy_note
        allowed = parser.can_fetch(self._settings.user_agent, url)
        return allowed, ("allowed" if allowed else "disallowed_by_robots_txt")

    async def _fetch_and_parse(self, robots_url: str) -> tuple[RobotFileParser | None, str]:
        current = robots_url
        try:
            async with self._gateway.create_http_client(
                is_external=True, timeout_seconds=min(self._settings.request_timeout_seconds, 6.0)
            ) as client:
                for _ in range(min(self._settings.max_redirects, 3) + 1):
                    await validate_url_for_fetch(
                        current,
                        allowed_ports=self._settings.allowed_ports,
                        resolver=self._dns_resolver,
                    )
                    resp = await client.get(current)
                    if resp.status_code in (301, 302, 303, 307, 308):
                        location = resp.headers.get("Location")
                        if not location:
                            break
                        current = urljoin(current, location)
                        continue
                    if resp.status_code in (401, 403):
                        return None, "disallow_all"
                    if resp.status_code == 404 or resp.status_code == 410 or resp.status_code >= 500:
                        return None, f"robots_unavailable_http_{resp.status_code}"
                    if resp.status_code == 200:
                        raw_text = resp.content[:_MAX_ROBOTS_BYTES].decode("utf-8", errors="replace")
                        parser = RobotFileParser()
                        parser.parse(raw_text.splitlines())
                        return parser, "robots_parsed"
                    return None, f"robots_status_{resp.status_code}"
        except (SSRFError, URLValidationError) as exc:
            logger.warning(
                "Blocked SSRF/invalid target when checking robots.txt",
                extra={"extra_data": {"url": robots_url, "error": str(exc)}},
            )
            return None, "disallow_all"
        except (httpx.HTTPError, PrivacyPolicyViolation) as exc:
            logger.debug(
                "robots.txt unavailable; proceeding cautiously",
                extra={"extra_data": {"url": robots_url, "error": str(exc)}},
            )
            return None, "robots_fetch_error"

        return None, "robots_redirect_limit"
