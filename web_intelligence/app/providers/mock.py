"""Offline MockSearchProvider enabling full pipeline verification without API keys."""

from __future__ import annotations

import re

from web_intelligence.app.core.security import canonicalize_url, extract_domain
from web_intelligence.app.providers.base import SearchProvider, parse_optional_datetime
from web_intelligence.app.providers.fixtures import DEFAULT_FIXTURES, FixturePage
from web_intelligence.app.schemas.search import NormalizedSearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class MockSearchProvider(SearchProvider):
    """Deterministic offline search provider backed by controlled fixture documents."""

    name = "mock"
    is_mock = True

    def __init__(self, fixtures: list[FixturePage] | None = None) -> None:
        self._fixtures: list[FixturePage] = list(fixtures) if fixtures is not None else list(DEFAULT_FIXTURES)

    @property
    def is_configured(self) -> bool:
        return True

    def set_fixtures(self, fixtures: list[FixturePage]) -> None:
        """Replace fixture catalog (useful for deterministic unit tests)."""
        self._fixtures = list(fixtures)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 6,
    ) -> list[NormalizedSearchResult]:
        query_tokens = set(_TOKEN_RE.findall(query.lower()))
        if not query_tokens:
            return []

        scored: list[tuple[int, int, FixturePage]] = []
        for idx, page in enumerate(self._fixtures):
            haystack_tokens = set(
                _TOKEN_RE.findall(
                    f"{page.title} {page.snippet} {' '.join(page.keywords)}".lower()
                )
            )
            overlap = len(query_tokens & haystack_tokens)
            scored.append((overlap, -idx, page))

        # Keep overlapping matches first; if none overlap on a novel offline query,
        # return top fixtures only if query is meaningful? No—wait:
        # for empty-search-results tests or unmatched topics, only return matches with overlap > 0!
        matched = [item for item in scored if item[0] > 0]
        matched.sort(key=lambda item: (item[0], item[1]), reverse=True)

        results: list[NormalizedSearchResult] = []
        for rank, (overlap, _, page) in enumerate(matched[:max_results], start=1):
            results.append(
                NormalizedSearchResult(
                    title=page.title,
                    url=page.url,
                    canonical_url=canonicalize_url(page.url),
                    domain=extract_domain(page.url),
                    snippet=page.snippet,
                    published_at=parse_optional_datetime(page.published_at),
                    provider=self.name,
                    rank=rank,
                    raw_score=float(overlap),
                )
            )
        return results
