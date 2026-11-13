"""Official Brave Web Search API adapter."""

from __future__ import annotations

import httpx

from web_intelligence.app.core.security import canonicalize_url, extract_domain
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    SearchProvider,
    execute_with_retries,
    parse_optional_datetime,
)
from web_intelligence.app.schemas.search import NormalizedSearchResult


class BraveSearchProvider(SearchProvider):
    """Search adapter for the official Brave Search REST API."""

    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None, gateway: PrivacyGateway) -> None:
        self._api_key = api_key.strip() if api_key else None
        self._gateway = gateway

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 6,
    ) -> list[NormalizedSearchResult]:
        if not self._api_key:
            raise ProviderAuthError("Brave API key is not configured.")

        async def _call() -> list[NormalizedSearchResult]:
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self._api_key or "",
            }
            async with self._gateway.create_http_client(
                is_external=True, extra_headers=headers
            ) as client:
                try:
                    response = await client.get(
                        self.endpoint,
                        params={"q": query, "count": min(max_results, 20)},
                    )
                except httpx.TimeoutException as exc:
                    raise ProviderTimeoutError(f"Brave request timed out: {exc}") from exc
                except httpx.HTTPError as exc:
                    raise ProviderError(f"Brave network failure: {exc}") from exc

            if response.status_code in (401, 403):
                raise ProviderAuthError(f"Brave authentication failed (HTTP {response.status_code}).")
            if response.status_code == 429:
                raise ProviderRateLimitError("Brave rate limit exceeded (HTTP 429).")
            if response.status_code >= 500:
                raise ProviderTimeoutError(f"Brave transient server error (HTTP {response.status_code}).")
            if response.status_code >= 400:
                raise ProviderError(f"Brave HTTP error {response.status_code}.")

            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderResponseError("Brave returned invalid JSON.") from exc

            if not isinstance(payload, dict):
                raise ProviderResponseError("Brave response payload is not a JSON object.")

            web_block = payload.get("web") or {}
            raw_results = web_block.get("results") if isinstance(web_block, dict) else None
            if raw_results is None:
                return []
            if not isinstance(raw_results, list):
                raise ProviderResponseError("Brave 'web.results' field is malformed.")

            normalized: list[NormalizedSearchResult] = []
            for rank, item in enumerate(raw_results[:max_results], start=1):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                title = str(item.get("title") or "").strip()
                if not url or not title:
                    continue
                normalized.append(
                    NormalizedSearchResult(
                        title=title,
                        url=url,
                        canonical_url=canonicalize_url(url),
                        domain=extract_domain(url),
                        snippet=str(item.get("description") or "").strip(),
                        published_at=parse_optional_datetime(
                            item.get("page_age") or item.get("age")
                        ),
                        provider=self.name,
                        rank=rank,
                        raw_score=None,
                    )
                )
            return normalized

        return await execute_with_retries(_call)
