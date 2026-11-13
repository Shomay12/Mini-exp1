"""Unit tests for search providers, retry/failover routing, and error handling."""

from __future__ import annotations

import httpx
import pytest

from web_intelligence.app.core.config import Settings
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.providers.base import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from web_intelligence.app.providers.brave import BraveSearchProvider
from web_intelligence.app.providers.exa import ExaSearchProvider
from web_intelligence.app.providers.mock import MockSearchProvider
from web_intelligence.app.providers.router import SearchProviderRouter
from web_intelligence.app.providers.tavily import TavilySearchProvider


def _gateway_with_handler(handler) -> PrivacyGateway:
    settings = Settings(search_providers=["brave", "tavily", "exa"], enable_mock_provider=True)
    return PrivacyGateway(settings, transport_factory=lambda: httpx.MockTransport(handler))


async def test_mock_provider_returns_matches_and_empty_for_unmatched() -> None:
    provider = MockSearchProvider()
    hits = await provider.search("small language models quantized", max_results=4)
    assert len(hits) >= 2
    assert all(hit.provider == "mock" for hit in hits)

    empty_hits = await provider.search("xyznonexistenttoken999", max_results=4)
    assert empty_hits == []


async def test_brave_provider_normalization() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-Subscription-Token") == "test-brave-key"
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Brave Hit",
                            "url": "https://example.org/post?utm_source=x",
                            "description": "Normalized snippet",
                            "page_age": "2026-03-01T10:00:00Z",
                        }
                    ]
                }
            },
        )

    provider = BraveSearchProvider("test-brave-key", _gateway_with_handler(handler))
    results = await provider.search("small language models", max_results=5)
    assert len(results) == 1
    assert results[0].canonical_url == "https://example.org/post"
    assert results[0].provider == "brave"
    assert results[0].published_at is not None


async def test_tavily_and_exa_normalization() -> None:
    def tavily_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Tavily Doc",
                        "url": "https://tavily.example.com/a",
                        "content": "Tavily snippet",
                        "score": 0.91,
                        "published_date": "2026-02-05",
                    }
                ]
            },
        )

    def exa_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-api-key") == "test-exa-key"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Exa Paper",
                        "url": "https://exa.example.edu/paper",
                        "highlights": ["First highlight", "Second highlight"],
                        "publishedDate": "2026-01-15T00:00:00.000Z",
                        "score": 0.88,
                    }
                ]
            },
        )

    tavily = TavilySearchProvider("test-tavily-key", _gateway_with_handler(tavily_handler))
    exa = ExaSearchProvider("test-exa-key", _gateway_with_handler(exa_handler))

    t_hits = await tavily.search("local llm")
    e_hits = await exa.search("local llm")
    assert t_hits[0].raw_score == pytest.approx(0.91)
    assert e_hits[0].snippet == "First highlight Second highlight"


async def test_provider_failure_modes() -> None:
    def auth_fail(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    def rate_limit(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "too many requests"})

    def bad_json(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    def timeout_err(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(ProviderAuthError):
        await BraveSearchProvider("k", _gateway_with_handler(auth_fail)).search("test")

    with pytest.raises(ProviderRateLimitError):
        await TavilySearchProvider("k", _gateway_with_handler(rate_limit)).search("test")

    with pytest.raises(ProviderResponseError):
        await ExaSearchProvider("k", _gateway_with_handler(bad_json)).search("test")

    with pytest.raises(ProviderTimeoutError):
        await BraveSearchProvider("k", _gateway_with_handler(timeout_err)).search("test")


async def test_router_failover_when_primary_provider_fails() -> None:
    def failing_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    settings = Settings(search_providers=["brave", "mock"], enable_mock_provider=True)
    gateway = PrivacyGateway(
        settings, transport_factory=lambda: httpx.MockTransport(failing_handler)
    )
    failing_brave = BraveSearchProvider("fake-key", gateway)
    mock = MockSearchProvider()
    router = SearchProviderRouter(settings, gateway, providers=[failing_brave, mock])

    results, disclosed, diagnostics, sanitized, _ = await router.search(
        "small language models contact user@example.com",
        max_results=3,
    )
    assert "user@example.com" not in sanitized
    assert len(results) >= 1
    assert diagnostics[0].provider == "brave" and diagnostics[0].success is False
    assert diagnostics[1].provider == "mock" and diagnostics[1].success is True
    assert len(disclosed) == 2
