"""Multi-provider search router with failover, sanitization, and diagnostics."""

from __future__ import annotations

import time
from collections.abc import Sequence

from web_intelligence.app.core.config import Settings
from web_intelligence.app.core.logging import get_logger
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.privacy.policy import PrivacyPolicyViolation
from web_intelligence.app.providers.base import ProviderError, SearchProvider
from web_intelligence.app.providers.brave import BraveSearchProvider
from web_intelligence.app.providers.exa import ExaSearchProvider
from web_intelligence.app.providers.mock import MockSearchProvider
from web_intelligence.app.providers.tavily import TavilySearchProvider
from web_intelligence.app.schemas.search import (
    DisclosedSearchQuery,
    NormalizedSearchResult,
    ProviderDiagnostic,
    ProviderStatus,
)

logger = get_logger("providers.router")


class SearchProviderRouter:
    """Executes sanitized searches across configured providers with resilient failover."""

    def __init__(
        self,
        settings: Settings,
        gateway: PrivacyGateway,
        *,
        providers: Sequence[SearchProvider] | None = None,
    ) -> None:
        self._settings = settings
        self._gateway = gateway
        if providers is not None:
            self._registry: dict[str, SearchProvider] = {p.name: p for p in providers}
        else:
            brave_key = settings.brave_api_key.get_secret_value() if settings.brave_api_key else None
            tavily_key = settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else None
            exa_key = settings.exa_api_key.get_secret_value() if settings.exa_api_key else None
            built: list[SearchProvider] = [
                MockSearchProvider(),
                BraveSearchProvider(brave_key, gateway),
                TavilySearchProvider(tavily_key, gateway),
                ExaSearchProvider(exa_key, gateway),
            ]
            self._registry = {p.name: p for p in built}

    def get_provider(self, name: str) -> SearchProvider | None:
        """Return a registered provider by name."""
        return self._registry.get(name.lower().strip())

    def list_statuses(self) -> list[ProviderStatus]:
        """Return availability and configuration status for all registered providers."""
        enabled_set = set(self._settings.search_providers)
        statuses: list[ProviderStatus] = []
        for name, provider in self._registry.items():
            enabled = name in enabled_set and (not provider.is_mock or self._settings.enable_mock_provider)
            notes = ""
            if provider.is_mock:
                notes = "Offline deterministic fixture provider (no API credentials required)."
            elif not provider.is_configured:
                notes = "API key not configured."
            else:
                notes = "Configured via environment secret."
            statuses.append(
                ProviderStatus(
                    name=name,
                    configured=provider.is_configured,
                    enabled=enabled,
                    is_mock=provider.is_mock,
                    notes=notes,
                )
            )
        return statuses

    def _select_providers(self, requested: list[str] | None) -> list[SearchProvider]:
        candidates = requested if requested else self._settings.search_providers
        selected: list[SearchProvider] = []
        for name in candidates:
            provider = self._registry.get(name.lower().strip())
            if provider is None:
                continue
            if provider.is_mock and not self._settings.enable_mock_provider:
                continue
            selected.append(provider)
        return selected

    async def search(
        self,
        raw_query: str,
        *,
        max_results: int = 6,
        providers: list[str] | None = None,
        intent: str = "exploratory",
        failover_only: bool = True,
    ) -> tuple[
        list[NormalizedSearchResult],
        list[DisclosedSearchQuery],
        list[ProviderDiagnostic],
        str,
        list[str],
    ]:
        """Sanitize query, dispatch to configured providers with failover, and deduplicate hits."""
        sanitized = self._gateway.prepare_search_query(raw_query)
        if not sanitized.sanitized_query:
            return [], [], [], "", sanitized.redactions_applied

        selected = self._select_providers(providers)
        results: list[NormalizedSearchResult] = []
        disclosed: list[DisclosedSearchQuery] = []
        diagnostics: list[ProviderDiagnostic] = []
        seen_urls: set[str] = set()

        for provider in selected:
            if not provider.is_configured:
                diagnostics.append(
                    ProviderDiagnostic(
                        provider=provider.name,
                        success=False,
                        result_count=0,
                        error=f"Provider '{provider.name}' is not configured.",
                    )
                )
                continue

            start = time.perf_counter()
            disclosure = self._gateway.record_search_disclosure(
                provider=provider.name,
                sanitized=sanitized,
                is_external=not provider.is_mock,
            )

            try:
                hits = await provider.search(
                    sanitized.sanitized_query,
                    max_results=max_results,
                )
                latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
                diagnostics.append(
                    ProviderDiagnostic(
                        provider=provider.name,
                        success=True,
                        result_count=len(hits),
                        latency_ms=latency_ms,
                    )
                )
                disclosed.append(
                    DisclosedSearchQuery(
                        provider=provider.name,
                        original_subquery=raw_query,
                        sanitized_query=sanitized.sanitized_query,
                        redactions_applied=sanitized.redactions_applied,
                        intent=intent,
                        result_count=len(hits),
                        disclosed_at=disclosure.disclosed_at,
                        is_external=disclosure.is_external,
                    )
                )
                for hit in hits:
                    dedup_key = hit.canonical_url or hit.url
                    if dedup_key not in seen_urls:
                        seen_urls.add(dedup_key)
                        results.append(hit)

                if failover_only and hits:
                    break
            except (ProviderError, PrivacyPolicyViolation) as exc:
                latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
                logger.warning(
                    "Search provider failed; trying next configured provider",
                    extra={"extra_data": {"provider": provider.name, "error": str(exc)}},
                )
                diagnostics.append(
                    ProviderDiagnostic(
                        provider=provider.name,
                        success=False,
                        result_count=0,
                        latency_ms=latency_ms,
                        error=str(exc),
                    )
                )
                disclosed.append(
                    DisclosedSearchQuery(
                        provider=provider.name,
                        original_subquery=raw_query,
                        sanitized_query=sanitized.sanitized_query,
                        redactions_applied=sanitized.redactions_applied,
                        intent=intent,
                        result_count=0,
                        disclosed_at=disclosure.disclosed_at,
                        is_external=disclosure.is_external,
                    )
                )

        return (
            results,
            disclosed,
            diagnostics,
            sanitized.sanitized_query,
            sanitized.redactions_applied,
        )
