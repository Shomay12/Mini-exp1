"""Schemas for Stage 4 evidence chunks, relevance, claims, conflicts, and coverage."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from web_intelligence.app.schemas.documents import SourceProvenance


class CorroborationStatus(str, Enum):
    """Cross-source corroboration status (never conflated with relevance score)."""

    CORROBORATED = "corroborated"
    SINGLE_SOURCE = "single_source"
    CONTESTED = "contested"
    UNVERIFIED = "unverified"


class RelevanceBreakdown(BaseModel):
    """Transparent multi-signal query relevance metrics (never factual confidence)."""

    bm25_score: float = Field(default=0.0, ge=0.0, le=1.0)
    keyword_match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    heading_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    query_term_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    document_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    rerank_score: float | None = Field(default=None, ge=0.0, le=1.0)
    combined_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_factual_confidence: bool = False


class CorroborationSignals(BaseModel):
    """Cross-source support and contradiction state for an evidence item."""

    corroboration_status: CorroborationStatus = CorroborationStatus.UNVERIFIED
    independent_supporting_sources: int = Field(default=0, ge=0)
    supporting_source_ids: list[str] = Field(default_factory=list)
    supporting_domains: list[str] = Field(default_factory=list)
    conflicting_claim_ids: list[str] = Field(default_factory=list)
    same_underlying_source_cluster: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """Traceable passage unit linked back to its source document and URL."""

    evidence_id: str
    source_id: str
    source_url: str
    source_domain: str
    source_title: str | None = None
    chunk_index: int = Field(ge=0)
    section_heading: str | None = None
    passage_text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    token_estimate: int = Field(ge=0)
    relevance: RelevanceBreakdown
    provenance: SourceProvenance
    corroboration: CorroborationSignals = Field(default_factory=CorroborationSignals)


class ExtractedClaim(BaseModel):
    """Deterministic structured factual assertion linked to a source passage."""

    claim_id: str
    source_id: str
    evidence_id: str
    source_url: str
    source_domain: str
    entity: str
    attribute: str
    statement_text: str
    value_type: str  # "date" | "quantity" | "polarity"
    normalized_value: str
    unit: str | None = None
    polarity: str = "affirmative"  # "affirmative" | "negative"


class ClaimConflict(BaseModel):
    """Recorded discrepancy between claims from independent sources without picking a winner."""

    conflict_id: str
    topic: str
    entity: str
    attribute: str
    conflict_type: str  # "date_mismatch" | "numeric_mismatch" | "polarity_mismatch"
    claim_a: ExtractedClaim
    claim_b: ExtractedClaim
    summary: str
    resolution_note: str = (
        "Conflict preserved for inspection; automated extraction does not arbitrate factual truth."
    )


class SubtopicStatus(str, Enum):
    """Coverage state for an individual subtopic or query facet."""

    COVERED = "covered"
    PARTIAL = "partial"
    MISSING = "missing"


class SubtopicCoverage(BaseModel):
    """Detailed coverage record for a subtopic extracted from the user query."""

    subtopic: str
    status: SubtopicStatus
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    supporting_source_ids: list[str] = Field(default_factory=list)
    independent_domain_count: int = 0
    has_conflict: bool = False
    notes: str = ""


class CoverageStatus(str, Enum):
    """Overall research coverage sufficiency."""

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class CoverageAnalysis(BaseModel):
    """Structured analysis of what is supported, conflicted, single-source, or missing."""

    covered_topics: list[str] = Field(default_factory=list)
    partial_topics: list[str] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    subtopic_details: list[SubtopicCoverage] = Field(default_factory=list)
    well_corroborated_claims: list[str] = Field(default_factory=list)
    single_source_claims: list[str] = Field(default_factory=list)
    conflicted_areas: list[str] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)
    coverage_status: CoverageStatus = CoverageStatus.INSUFFICIENT
    summary: str = ""
