"""Unit tests for URL validation, SSRF protection, canonicalization, and config security."""

from __future__ import annotations

import logging

import pytest

from web_intelligence.app.core.config import Settings
from web_intelligence.app.core.logging import SecretRedactingFilter
from web_intelligence.app.core.security import (
    SSRFError,
    URLValidationError,
    canonicalize_url,
    is_blocked_ip,
    validate_url_for_fetch,
    validate_url_syntax,
)


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.10",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "::1",
        "::ffff:127.0.0.1",
        "fe80::1",
        "fc00::1",
    ],
)
def test_blocked_ips_detected(ip: str) -> None:
    assert is_blocked_ip(ip) is True


@pytest.mark.parametrize("ip", ["93.184.216.34", "1.1.1.1", "8.8.8.8", "2606:2800:220:1:248:1893:25c8:1946"])
def test_public_ips_allowed(ip: str) -> None:
    assert is_blocked_ip(ip) is False


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "ftp://example.com/file",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "https://user:pass@example.com/path",
        "http://example.com\n/path",
        "https:///missing-host",
    ],
)
def test_invalid_urls_rejected(bad_url: str) -> None:
    with pytest.raises((URLValidationError, SSRFError)):
        validate_url_syntax(bad_url)


@pytest.mark.parametrize(
    "ssrf_url",
    [
        "http://localhost/admin",
        "http://127.0.0.1/latest/meta-data",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://[::1]/",
        "https://example.com:6379/",
        "http://internal.service.local/",
    ],
)
def test_ssrf_targets_blocked(ssrf_url: str) -> None:
    with pytest.raises(SSRFError):
        validate_url_syntax(ssrf_url)


async def test_dns_rebinding_private_ip_blocked() -> None:
    async def evil_resolver(hostname: str, port: int) -> list[str]:
        return ["127.0.0.1"]

    with pytest.raises(SSRFError, match="restricted IP"):
        await validate_url_for_fetch("https://example.com/article", resolver=evil_resolver)


async def test_dns_public_ip_allowed() -> None:
    async def public_resolver(hostname: str, port: int) -> list[str]:
        return ["93.184.216.34"]

    clean, ips = await validate_url_for_fetch("https://example.com/article", resolver=public_resolver)
    assert clean == "https://example.com/article"
    assert ips == ["93.184.216.34"]


def test_canonicalize_url_normalizes_tracking_and_ports() -> None:
    url_a = "HTTPS://Example.COM:443/docs//guide/?utm_source=x&b=2&a=1#section"
    url_b = "https://example.com/docs/guide?a=1&b=2&fbclid=abc"
    assert canonicalize_url(url_a) == "https://example.com/docs/guide?a=1&b=2"
    assert canonicalize_url(url_a) == canonicalize_url(url_b)


def test_safe_config_never_exposes_secrets() -> None:
    settings = Settings(
        brave_api_key="BSA_super_secret_value_12345",
        tavily_api_key="tvly-super-secret-67890",
        exa_api_key="",
        enable_telemetry=True,
    )
    safe = settings.to_safe_dict()
    serialized = str(safe)
    assert "BSA_super_secret_value_12345" not in serialized
    assert "tvly-super-secret-67890" not in serialized
    assert safe["configured_credentials"] == {"brave": True, "tavily": True, "exa": False}
    assert safe["privacy"]["enable_telemetry"] is False


def test_log_filter_redacts_secrets() -> None:
    log_filter = SecretRedactingFilter(["my-custom-secret-token"])
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Calling provider with key my-custom-secret-token and Bearer abcd1234efgh",
        args=(),
        exc_info=None,
    )
    log_filter.filter(record)
    assert "my-custom-secret-token" not in record.msg
    assert "abcd1234efgh" not in record.msg
    assert "[REDACTED_SECRET]" in record.msg
