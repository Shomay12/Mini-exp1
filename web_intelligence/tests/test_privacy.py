"""Unit tests for privacy sanitization, policy checks, and gateway disclosures."""

from __future__ import annotations

import pytest

from web_intelligence.app.core.config import Settings
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.privacy.policy import PrivacyPolicy, PrivacyPolicyViolation
from web_intelligence.app.privacy.sanitizer import QuerySanitizer


def test_query_sanitizer_strips_pii_secrets_and_paths() -> None:
    sanitizer = QuerySanitizer(max_query_length=200)
    raw = (
        "Research local LLMs for alice@example.com using /Users/alice/secret/notes.txt "
        "with token sk-1234567890abcdef and phone 415-555-0199"
    )
    result = sanitizer.sanitize(raw)
    assert "alice@example.com" not in result.sanitized_query
    assert "/Users/alice" not in result.sanitized_query
    assert "sk-1234567890abcdef" not in result.sanitized_query
    assert "415-555-0199" not in result.sanitized_query
    assert "Research local LLMs" in result.sanitized_query
    assert set(result.redactions_applied) >= {
        "email_address",
        "local_path",
        "api_token",
        "phone_number",
    }
    assert result.was_modified is True


def test_query_sanitizer_caps_length() -> None:
    sanitizer = QuerySanitizer(max_query_length=40)
    result = sanitizer.sanitize("small language models quantized inference benchmarks on consumer laptops")
    assert len(result.sanitized_query) <= 40
    assert result.truncated is True
    assert "length_cap" in result.redactions_applied


def test_privacy_policy_blocks_live_network_when_disabled() -> None:
    settings = Settings(allow_live_network=False)
    gateway = PrivacyGateway(settings)
    with pytest.raises(PrivacyPolicyViolation):
        gateway.create_http_client(is_external=True)


def test_privacy_policy_disclosure_does_not_claim_zero_footprint() -> None:
    policy = PrivacyPolicy(
        allow_live_network=True,
        disclose_outbound_queries=True,
        retention_days=30,
    )
    notice = policy.disclosure_notice()
    assert "observe" in notice.lower()
    assert "zero footprint" not in notice.lower()
