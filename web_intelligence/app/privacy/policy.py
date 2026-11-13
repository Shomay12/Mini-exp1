"""Privacy policy enforcement and disclosure metadata."""

from __future__ import annotations

from dataclasses import dataclass

from web_intelligence.app.core.config import Settings


class PrivacyPolicyViolation(PermissionError):
    """Raised when an operation violates configured privacy boundaries."""


@dataclass(frozen=True)
class PrivacyPolicy:
    """Explicit privacy boundaries governing outbound search, retrieval, and local storage."""

    allow_live_network: bool
    disclose_outbound_queries: bool
    retention_days: int
    telemetry_enabled: bool = False
    cloud_llm_required: bool = False
    proxy_url: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "PrivacyPolicy":
        """Build policy directly from runtime settings."""
        return cls(
            allow_live_network=settings.allow_live_network,
            disclose_outbound_queries=settings.disclose_outbound_queries,
            retention_days=settings.retention_days,
            telemetry_enabled=False,
            cloud_llm_required=False,
            proxy_url=settings.https_proxy or settings.http_proxy,
        )

    def assert_live_network_allowed(self, target_description: str) -> None:
        """Fail closed if live network egress is disabled."""
        if not self.allow_live_network:
            raise PrivacyPolicyViolation(
                f"Live network access is disabled by privacy policy (blocked: {target_description})."
            )

    def disclosure_notice(self) -> str:
        """Explain network visibility accurately without overclaiming anonymity."""
        if self.proxy_url:
            return (
                "Outbound search queries and webpage fetches are routed through the configured "
                "proxy. Search providers and destination websites can still observe query text "
                "and HTTP request metadata."
            )
        return (
            "Search providers and target websites observe sanitized search queries, HTTP headers, "
            "and client network metadata. Research history and extracted documents remain local."
        )
