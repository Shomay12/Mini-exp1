"""Unit tests for SSRF-safe fetching, redirects, robots.txt, extraction, and normalization."""

from __future__ import annotations

import httpx
import pytest

from web_intelligence.app.core.config import Settings
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.retrieval.extractor import ContentExtractor, validate_publication_date
from web_intelligence.app.retrieval.fetcher import WebFetcher
from web_intelligence.app.retrieval.normalizer import DocumentNormalizer
from web_intelligence.app.schemas.documents import ExtractionStatus


async def _public_resolver(hostname: str, port: int) -> list[str]:
    return ["93.184.216.34"]


def _make_fetcher(handler, *, max_download_bytes: int = 2_000_000) -> WebFetcher:
    settings = Settings(
        per_host_min_interval_seconds=0.0,
        max_download_bytes=max_download_bytes,
    )
    gateway = PrivacyGateway(settings, transport_factory=lambda: httpx.MockTransport(handler))
    return WebFetcher(settings, gateway, dns_resolver=_public_resolver)


async def test_redirect_to_private_ip_blocked_as_ssrf() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data/"})

    fetcher = _make_fetcher(handler)
    outcome = await fetcher.fetch("https://public-site.example.com/start")
    assert outcome.success is False
    assert outcome.status == ExtractionStatus.BLOCKED_SSRF


async def test_robots_txt_disallow_blocks_fetch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, text="<html><body>Secret</body></html>")

    fetcher = _make_fetcher(handler)
    outcome = await fetcher.fetch("https://example.org/private/report")
    assert outcome.success is False
    assert outcome.status == ExtractionStatus.BLOCKED_ROBOTS


async def test_timeout_and_size_limit_and_content_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/timeout":
            raise httpx.ReadTimeout("simulated timeout")
        if request.url.path == "/huge":
            return httpx.Response(
                200,
                content=b"<html>" + b"A" * 20000 + b"</html>",
                headers={"Content-Type": "text/html"},
            )
        if request.url.path == "/binary":
            return httpx.Response(
                200,
                content=b"%PDF-1.4...",
                headers={"Content-Type": "application/pdf"},
            )
        return httpx.Response(200, text="<html><body>ok</body></html>")

    fetcher = _make_fetcher(handler, max_download_bytes=12_000)

    timeout_out = await fetcher.fetch("https://example.org/timeout")
    huge_out = await fetcher.fetch("https://example.org/huge")
    bin_out = await fetcher.fetch("https://example.org/binary")

    assert timeout_out.status == ExtractionStatus.TIMEOUT
    assert huge_out.status == ExtractionStatus.TOO_LARGE
    assert bin_out.status == ExtractionStatus.UNSUPPORTED_CONTENT_TYPE


async def test_valid_redirect_malformed_html_and_missing_dates() -> None:
    malformed_html = """<html><head><title>Malformed Page</title>
    <meta name="date" content="not-a-valid-date"></head>
    <body><article><h2>Heading One<p>First unclosed paragraph with enough informative content for extraction.
    <p>Second paragraph discussing small language models and citations.</article></body>"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/old":
            return httpx.Response(301, headers={"Location": "https://example.org/new"})
        return httpx.Response(200, text=malformed_html, headers={"Content-Type": "text/html; charset=utf-8"})

    fetcher = _make_fetcher(handler)
    outcome = await fetcher.fetch("https://example.org/old")
    assert outcome.success is True
    assert outcome.final_url == "https://example.org/new"
    assert outcome.redirect_chain == ["https://example.org/old"]

    extracted = ContentExtractor().extract(
        outcome.body_bytes, final_url=outcome.final_url, content_type=outcome.content_type
    )
    doc = DocumentNormalizer().normalize(outcome, extracted)

    assert doc.title == "Malformed Page"
    assert doc.published_at is None
    assert doc.author is None
    assert "small language models" in doc.main_content
    assert doc.content_hash != ""
    assert any("unparseable" in w.lower() for w in doc.source_metadata.warnings)


def test_implausible_publication_dates_rejected() -> None:
    future_dt, future_warn = validate_publication_date("2099-05-01")
    ancient_dt, ancient_warn = validate_publication_date("1850-01-01")
    good_dt, good_warn = validate_publication_date("2026-02-10")

    assert future_dt is None and future_warn is not None
    assert ancient_dt is None and ancient_warn is not None
    assert good_dt is not None and good_dt.year == 2026 and good_warn is None


def test_prose_year_mention_does_not_invent_publication_date() -> None:
    html = (
        "<html><head><title>Undated Page</title></head>"
        "<body><article><p>The model was released in January 2026 under Apache 2.0 "
        "with full local retrieval support.</p></article></body></html>"
    )
    extracted = ContentExtractor().extract(
        html.encode("utf-8"),
        final_url="https://example.org/undated",
        content_type="text/html",
    )
    assert extracted.published_at is None
