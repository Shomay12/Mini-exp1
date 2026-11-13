"""Integrated web search and webpage retrieval pipeline."""

from __future__ import annotations

from web_intelligence.app.core.config import Settings
from web_intelligence.app.core.security import AsyncResolver
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.providers.router import SearchProviderRouter
from web_intelligence.app.retrieval.extractor import ContentExtractor
from web_intelligence.app.retrieval.fetcher import WebFetcher
from web_intelligence.app.retrieval.normalizer import DocumentNormalizer
from web_intelligence.app.schemas.documents import RetrievedDocument
from web_intelligence.app.schemas.search import NormalizedSearchResult


class WebRetrievalEngine:
    """Coordinates safe fetching, content extraction, and document normalization."""

    def __init__(
        self,
        settings: Settings,
        gateway: PrivacyGateway,
        *,
        router: SearchProviderRouter | None = None,
        fetcher: WebFetcher | None = None,
        extractor: ContentExtractor | None = None,
        normalizer: DocumentNormalizer | None = None,
        dns_resolver: AsyncResolver | None = None,
    ) -> None:
        self.settings = settings
        self.gateway = gateway
        self.router = router or SearchProviderRouter(settings, gateway)
        self.fetcher = fetcher or WebFetcher(settings, gateway, dns_resolver=dns_resolver)
        self.extractor = extractor or ContentExtractor()
        self.normalizer = normalizer or DocumentNormalizer()

    async def retrieve_results(
        self,
        search_results: list[NormalizedSearchResult],
        *,
        max_fetch_urls: int | None = None,
    ) -> list[tuple[NormalizedSearchResult, RetrievedDocument]]:
        """Fetch, extract, and normalize candidate search result URLs."""
        limit = max_fetch_urls or self.settings.max_fetch_urls
        selected = search_results[:limit]
        urls = [hit.url for hit in selected]
        outcomes = await self.fetcher.fetch_many(urls)

        documents: list[tuple[NormalizedSearchResult, RetrievedDocument]] = []
        for hit, outcome in zip(selected, outcomes, strict=True):
            extracted = (
                self.extractor.extract(
                    outcome.body_bytes,
                    final_url=outcome.final_url,
                    content_type=outcome.content_type,
                )
                if outcome.success
                else None
            )
            doc = self.normalizer.normalize(
                outcome,
                extracted,
                fallback_title=hit.title,
            )
            documents.append((hit, doc))
        return documents
