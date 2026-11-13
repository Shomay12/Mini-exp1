"""Official Exa Search API adapter."""

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


class ExaSearchProvider(SearchProvider):
    """Search adapter for the official Exa Search REST API."""

    name = "exa"
    endpoint = "https://api.exa.ai/search"

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
            raise ProviderAuthError("Exa API key is not configured.")

        async def _call() -> list[NormalizedSearchResult]:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-api-key": self._api_key or "",
            }
            body = {
                "query": query,
                "numResults": min(max_results, 20),
                "type": "auto",
            }
            async with self._gateway.create_http_client(
                is_external=True, extra_headers=headers
            ) as client:
                try:
                    response = await client.post(self.endpoint, json=body)
                except httpx.TimeoutException as exc:
                    raise ProviderTimeoutError(f"Exa request timed out: {exc}") from exc
                except httpx.HTTPError as exc:
                    raise ProviderError(f"Exa network failure: {exc}") from exc

            if response.status_code in (401, 403):
                raise ProviderAuthError(f"Exa authentication failed (HTTP {response.status_code}).")
            if response.status_code == 429:
                raise ProviderRateLimitError("Exa rate limit exceeded (HTTP 429).")
            if response.status_code >= 500:
                raise ProviderTimeoutError(f"Exa transient server error (HTTP {response.status_code}).")
            if response.status_code >= 400:
                raise ProviderError(f"Exa HTTP error {response.status_code}.")

            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderResponseError("Exa returned invalid JSON.") from exc

            if not isinstance(payload, dict):
                raise ProviderResponseError("Exa response payload is not a JSON object.")

            raw_results = payload.get("results")
            if raw_results is None:
                return []
            if not isinstance(raw_results, list):
                raise ProviderResponseError("Exa 'results' field is malformed.")

            normalized: list[NormalizedSearchResult] = []
            for rank, item in enumerate(raw_results[:max_results], start=1):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                title = str(item.get("title") or url).strip()
                if not url:
                    continue
                score_val = item.get("score")
                raw_score = float(score_val) if isinstance(score_val, (int, float)) else None
                highlights = item.get("highlights")
                snippet = ""
                if isinstance(highlights, list) and highlights:
                    snippet = " ".join(str(h) for h in highlights[:2]).strip()
                elif item.get("text"):
                    snippet = str(item.get("text"))[:280].strip()

                normalized.append(
                    NormalizedSearchResult(
                        title=title,
                        url=url,
                        canonical_url=canonicalize_url(url),
                        domain=extract_domain(url),
                        snippet=snippet,
                        published_at=parse_optional_datetime(item.get("publishedDate")),
                        provider=self.name,
                        rank=rank,
                        raw_score=raw_score,
                    )
                )
            return normalized

        return await execute_with_retries(_call)
