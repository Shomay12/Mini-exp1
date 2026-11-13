"""Schemas for retrieved web documents, source metadata, and provenance."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ExtractionStatus(str, Enum):
    """Outcome of webpage fetching and content extraction."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED_SSRF = "blocked_ssrf"
    BLOCKED_ROBOTS = "blocked_robots"
    TIMEOUT = "timeout"
    TOO_LARGE = "too_large"
    UNSUPPORTED_CONTENT_TYPE = "unsupported_content_type"
    HTTP_ERROR = "http_error"
    SKIPPED = "skipped"


class SourceClassification(str, Enum):
    """Structural classification of source role (distinct from factual accuracy)."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    REFERENCE = "reference"
    UNKNOWN = "unknown"


class DocumentHeading(BaseModel):
    """Heading structure preserved from HTML for passage-aware chunking."""

    level: int = Field(ge=1, le=6)
    text: str
    char_offset: int = Field(default=0, ge=0)


class SourceMetadata(BaseModel):
    """Technical and bibliographic metadata observed on a fetched webpage."""

    language: str | None = None
    description: str | None = None
    publisher: str | None = None
    extractor_used: str | None = None
    http_status: int | None = None
    content_type: str | None = None
    byte_length: int = 0
    headings: list[DocumentHeading] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    redirect_chain: list[str] = Field(default_factory=list)


class SourceProvenance(BaseModel):
    """Inspectable provenance attributes kept separate from relevance and confidence."""

    domain: str
    publisher: str | None = None
    source_classification: SourceClassification = SourceClassification.UNKNOWN
    classification_reason: str = "No structural primary/secondary markers detected."
    author: str | None = None
    published_at: datetime | None = None
    original_url: str
    canonical_url: str


class RetrievedDocument(BaseModel):
    """Normalized webpage document produced after extraction and cleanup."""

    source_id: str
    original_url: str
    canonical_url: str
    final_url: str
    domain: str
    title: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime
    main_content: str = ""
    content_hash: str = ""
    extraction_status: ExtractionStatus = ExtractionStatus.FAILED
    extraction_error: str | None = None
    source_metadata: SourceMetadata = Field(default_factory=SourceMetadata)
    provenance: SourceProvenance | None = None


class SourceRecord(RetrievedDocument):
    """Persisted and deduplicated source record exposed via API and ResearchPackage."""

    is_duplicate: bool = False
    duplicate_of_source_id: str | None = None
    duplicate_reason: str | None = None
    duplicate_similarity: float | None = None
    alias_urls: list[str] = Field(default_factory=list)
    document_relevance_score: float = 0.0
    search_snippet: str = ""
    search_provider: str = ""
