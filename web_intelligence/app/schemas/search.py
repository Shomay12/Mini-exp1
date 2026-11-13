"""Schemas for search queries, normalized search results, and provider diagnostics."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Input payload for POST /api/v1/search."""

    query: str = Field(..., min_length=2, max_length=1000)
    max_results: int = Field(default=6, ge=1, le=25)
    providers: list[str] | None = None


class NormalizedSearchResult(BaseModel):
    """Common search hit returned across all search provider adapters."""

    title: str
    url: str
    canonical_url: str
    domain: str
    snippet: str = ""
    published_at: datetime | None = None
    provider: str
    rank: int = Field(ge=1)
    raw_score: float | None = None


class DisclosedSearchQuery(BaseModel):
    """Explicit disclosure record of an outbound query sent to a search provider."""

    provider: str
    original_subquery: str
    sanitized_query: str
    redactions_applied: list[str] = Field(default_factory=list)
    intent: str = "exploratory"
    result_count: int = 0
    disclosed_at: datetime
    is_external: bool = True


class ProviderDiagnostic(BaseModel):
    """Execution outcome of querying a single provider."""

    provider: str
    success: bool
    result_count: int = 0
    latency_ms: float = 0.0
    error: str | None = None


class SearchResponse(BaseModel):
    """API response for POST /api/v1/search."""

    original_query: str
    sanitized_query: str
    redactions_applied: list[str] = Field(default_factory=list)
    results: list[NormalizedSearchResult] = Field(default_factory=list)
    disclosed_queries: list[DisclosedSearchQuery] = Field(default_factory=list)
    provider_diagnostics: list[ProviderDiagnostic] = Field(default_factory=list)
    total_results: int = 0


class ProviderStatus(BaseModel):
    """Availability status of a registered search provider."""

    name: str
    configured: bool
    enabled: bool
    is_mock: bool = False
    notes: str = ""
