"""Application settings loaded from environment variables with secret protection."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _normalize_optional_secret(value: Any) -> SecretStr | None:
    """Convert blank environment secret strings to None so unconfigured keys stay unset."""
    if value is None:
        return None
    if isinstance(value, SecretStr):
        raw = value.get_secret_value().strip()
        return SecretStr(raw) if raw else None
    if isinstance(value, str):
        raw = value.strip()
        return SecretStr(raw) if raw else None
    return value


class Settings(BaseSettings):
    """Runtime settings for the Local-First Web Intelligence Engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Local-First Web Intelligence Engine"
    app_version: str = "0.2.0"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"

    # Local storage & privacy retention
    database_path: Path = Path("./data/web_intelligence.db")
    retention_days: int = Field(default=30, ge=1, le=3650)

    # Provider selection
    search_providers: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["mock"])
    enable_mock_provider: bool = True

    # External search API keys (never logged or returned via API)
    brave_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    exa_api_key: SecretStr | None = None

    # Privacy & network boundaries
    allow_live_network: bool = True
    respect_robots_txt: bool = True
    http_proxy: str | None = None
    https_proxy: str | None = None
    enable_telemetry: bool = False
    disclose_outbound_queries: bool = True

    # Retrieval & security controls
    request_timeout_seconds: float = Field(default=15.0, gt=0.5, le=120.0)
    connect_timeout_seconds: float = Field(default=5.0, gt=0.5, le=60.0)
    max_redirects: int = Field(default=5, ge=0, le=15)
    max_download_bytes: int = Field(default=2_000_000, ge=10_000, le=20_000_000)
    max_concurrent_fetches: int = Field(default=4, ge=1, le=20)
    per_host_min_interval_seconds: float = Field(default=0.5, ge=0.0, le=30.0)
    max_results_per_query: int = Field(default=6, ge=1, le=25)
    max_fetch_urls: int = Field(default=8, ge=1, le=30)
    allowed_ports: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [80, 443])
    user_agent: str = "AdaptiveResearchEngine/0.2 (+local-first-web-intelligence)"

    # Evidence processing defaults
    default_chunk_target_tokens: int = Field(default=180, ge=40, le=2000)
    default_chunk_overlap_sentences: int = Field(default=1, ge=0, le=5)
    near_duplicate_threshold: float = Field(default=0.72, ge=0.3, le=0.99)

    @field_validator("brave_api_key", "tavily_api_key", "exa_api_key", mode="before")
    @classmethod
    def _clean_secret(cls, value: Any) -> SecretStr | None:
        return _normalize_optional_secret(value)

    @field_validator("http_proxy", "https_proxy", mode="before")
    @classmethod
    def _clean_proxy(cls, value: Any) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("search_providers", mode="before")
    @classmethod
    def _parse_providers(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            parsed = [item.strip().lower() for item in value.split(",") if item.strip()]
            return parsed or ["mock"]
        if isinstance(value, list):
            return [str(item).strip().lower() for item in value if str(item).strip()] or ["mock"]
        return ["mock"]

    @field_validator("allowed_ports", mode="before")
    @classmethod
    def _parse_ports(cls, value: Any) -> list[int]:
        if isinstance(value, str):
            return [int(p.strip()) for p in value.split(",") if p.strip()]
        return value

    @field_validator("enable_telemetry")
    @classmethod
    def _disable_telemetry(cls, value: bool) -> bool:
        # Privacy policy: outbound telemetry is strictly disabled in Phase 1.
        return False

    def registered_secrets(self) -> list[str]:
        """Return non-empty secret strings for log redaction filters."""
        secrets: list[str] = []
        for field_val in (self.brave_api_key, self.tavily_api_key, self.exa_api_key):
            if field_val is not None:
                raw = field_val.get_secret_value()
                if raw:
                    secrets.append(raw)
        return secrets

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a redacted configuration summary safe for API exposure."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment,
            "database_path": str(self.database_path),
            "retention_days": self.retention_days,
            "search_providers": self.search_providers,
            "enable_mock_provider": self.enable_mock_provider,
            "configured_credentials": {
                "brave": self.brave_api_key is not None,
                "tavily": self.tavily_api_key is not None,
                "exa": self.exa_api_key is not None,
            },
            "privacy": {
                "allow_live_network": self.allow_live_network,
                "respect_robots_txt": self.respect_robots_txt,
                "proxy_configured": bool(self.https_proxy or self.http_proxy),
                "enable_telemetry": self.enable_telemetry,
                "disclose_outbound_queries": self.disclose_outbound_queries,
                "cloud_llm_required": False,
            },
            "retrieval_limits": {
                "request_timeout_seconds": self.request_timeout_seconds,
                "connect_timeout_seconds": self.connect_timeout_seconds,
                "max_redirects": self.max_redirects,
                "max_download_bytes": self.max_download_bytes,
                "max_concurrent_fetches": self.max_concurrent_fetches,
                "per_host_min_interval_seconds": self.per_host_min_interval_seconds,
                "max_results_per_query": self.max_results_per_query,
                "max_fetch_urls": self.max_fetch_urls,
                "allowed_ports": self.allowed_ports,
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings."""
    return Settings()
