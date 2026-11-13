"""Pydantic schemas for Stage 6 Universal Local LLM execution, prompts, citations, and generation."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from web_intelligence.app.schemas.llm import (
    CapabilityProvenance,
    ChatMessage,
    ConnectionStatus,
    TokenUsage,
)


class PromptTemplateType(str, Enum):
    """Research prompt templates supported by PromptCompiler."""

    QUICK_ANSWER = "quick_answer"
    DETAILED_EXPLANATION = "detailed_explanation"
    RESEARCH_REPORT = "research_report"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TECHNICAL_REPORT = "technical_report"


class GenerationFormat(str, Enum):
    """Output rendering format requested from the answer engine."""

    PROSE = "prose"
    MARKDOWN = "markdown"
    BULLETS = "bullets"
    JSON = "json"


class GenerationLength(str, Enum):
    """Desired answer depth and approximate output reservation profile."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class GenerationMode(str, Enum):
    """Execution mode for single-pass synthesis vs multi-stage long-form report."""

    STANDARD = "standard"
    LONG_FORM_REPORT = "long_form_report"


class GenerationStatus(str, Enum):
    """Lifecycle states of a generation execution run."""

    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ModelTaskRole(str, Enum):
    """Optional role tags enabling future per-task model selection."""

    DEFAULT = "default"
    SYNTHESIS = "synthesis"
    OUTLINE = "outline"
    REPORT_SECTION = "report_section"
    EXTRACTION = "extraction"


class ModelRegistryEntry(BaseModel):
    """Registered local LLM model record across configured inference servers (zero secrets)."""

    entry_id: str = Field(..., min_length=1, max_length=240)
    connection_id: str = Field(..., min_length=1, max_length=120)
    connection_name: str = Field(..., min_length=1, max_length=120)
    provider_id: str = Field(..., min_length=1, max_length=64)
    preset: str | None = None
    server_endpoint: str
    api_endpoint: str | None = None
    model_id: str = Field(..., min_length=1, max_length=200)
    display_name: str = Field(..., min_length=1, max_length=200)
    context_limit: int = Field(..., ge=512, le=2_000_000)
    max_output_limit: int = Field(..., ge=64, le=131_072)
    supports_streaming: bool = True
    supports_json_mode: bool = False
    supports_tool_calling: bool = False
    supports_system_prompt: bool = True
    supports_stop_sequences: bool = True
    capability_provenance: CapabilityProvenance = Field(default_factory=CapabilityProvenance)
    connection_status: ConnectionStatus = ConnectionStatus.CONFIGURED
    last_connected_at: datetime | None = None
    last_error: str | None = None
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = False
    is_discovered: bool = False
    task_roles: list[ModelTaskRole] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelRegistryRegisterRequest(BaseModel):
    """Manual model registration payload when a server omits model discovery."""

    connection_id: str = Field(..., min_length=1, max_length=120)
    model_id: str = Field(..., min_length=1, max_length=200)
    display_name: str | None = Field(default=None, max_length=200)
    context_limit: int | None = Field(default=None, ge=512, le=2_000_000)
    max_output_limit: int | None = Field(default=None, ge=64, le=131_072)
    task_roles: list[ModelTaskRole] = Field(default_factory=list)


class ModelRegistryRoleUpdate(BaseModel):
    """Update task-specific model selection roles on a registered model entry."""

    task_roles: list[ModelTaskRole] = Field(default_factory=list)


class CitationEntry(BaseModel):
    """Resolved inline citation linking a source tag ([S1]) back to evidence and URL."""

    citation_id: str
    stage5_label: str | None = None
    evidence_id: str | None = None
    source_id: str
    title: str
    url: str
    canonical_url: str | None = None
    domain: str
    passage_excerpt: str = ""
    spans: list[tuple[int, int]] = Field(default_factory=list)
    occurrence_count: int = 0


class UnsupportedCitationError(BaseModel):
    """Citation reference emitted by the model that does not exist in source records."""

    raw_tag: str
    span: tuple[int, int]
    reason: str


class UncitedStatementWarning(BaseModel):
    """Factual-looking claim without an accompanying verified citation marker."""

    statement_index: int
    text: str
    span: tuple[int, int]
    reason: str


class CitationValidationReport(BaseModel):
    """Validation report distinguishing citation-syntax integrity from factual verification."""

    passed: bool
    require_citations: bool = True
    known_citation_ids: list[str] = Field(default_factory=list)
    used_citation_ids: list[str] = Field(default_factory=list)
    unused_citation_ids: list[str] = Field(default_factory=list)
    unsupported_citations: list[UnsupportedCitationError] = Field(default_factory=list)
    uncited_statements: list[UncitedStatementWarning] = Field(default_factory=list)
    grounding_note: str = (
        "Citation validation verifies that citation identifiers map to retrieved context "
        "sources and flags uncited claims; a valid source ID does not independently prove factual truth."
    )
    warnings: list[str] = Field(default_factory=list)


class CompiledPrompt(BaseModel):
    """Compiled provider-ready message bundle with strict untrusted-data boundaries."""

    template_type: PromptTemplateType
    output_format: GenerationFormat
    system_prompt: str
    user_prompt: str
    messages: list[ChatMessage]
    citation_map: dict[str, CitationEntry] = Field(default_factory=dict)
    stage5_to_citation_id: dict[str, str] = Field(default_factory=dict)
    estimated_prompt_tokens: int = 0
    reserved_output_tokens: int = 0
    total_required_tokens: int = 0
    context_limit: int = 0
    fits_context_limit: bool = True
    untrusted_data_delimited: bool = True
    sanitized_fence_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class LongFormReportConfig(BaseModel):
    """Controls multi-stage long-form research report planning and synthesis."""

    max_sections: int = Field(default=5, ge=2, le=10)
    section_max_output_tokens: int | None = Field(default=None, ge=64, le=8192)
    include_executive_summary: bool = True
    include_contradiction_section: bool = True
    detect_repetition: bool = True
    detect_contradictions: bool = True


class LongFormSectionPlan(BaseModel):
    """Planned report section with assigned evidence and citation subset."""

    section_id: str
    order: int
    title: str
    objective: str
    assigned_subtopics: list[str] = Field(default_factory=list)
    assigned_evidence_ids: list[str] = Field(default_factory=list)
    assigned_citation_ids: list[str] = Field(default_factory=list)


class LongFormSectionResult(BaseModel):
    """Generated output and diagnostics for one section of a long-form report."""

    section_id: str
    order: int
    title: str
    content: str = ""
    status: GenerationStatus = GenerationStatus.COMPLETED
    finish_reason: str | None = None
    truncated: bool = False
    citations: list[CitationEntry] = Field(default_factory=list)
    citation_validation: CitationValidationReport | None = None
    usage: TokenUsage | None = None
    allocated_input_tokens: int = 0
    duration_ms: int = 0
    warnings: list[str] = Field(default_factory=list)


class LongFormReportResult(BaseModel):
    """Complete multi-stage long-form report execution metadata and sections."""

    objective: str
    outline: list[LongFormSectionPlan] = Field(default_factory=list)
    sections: list[LongFormSectionResult] = Field(default_factory=list)
    repetition_warnings: list[str] = Field(default_factory=list)
    contradiction_warnings: list[str] = Field(default_factory=list)
    synthesis_notes: str = ""


class GenerationContextStats(BaseModel):
    """Snapshot of Stage 5 adaptive context allocation consumed for generation."""

    context_limit: int
    reserved_output_tokens: int
    usable_input_budget: int
    used_input_tokens: int
    evidence_items_included: int
    evidence_items_dropped: int
    compression_applied: bool = False
    validation_passed: bool = True
    validation_warnings: list[str] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    """Request payload for running local LLM answer/report generation over research results."""

    research_id: str = Field(..., min_length=1, max_length=128)
    connection_id: str | None = Field(default=None, max_length=120)
    model_id: str | None = Field(default=None, max_length=200)
    task_role: ModelTaskRole | None = None
    template_type: PromptTemplateType = PromptTemplateType.DETAILED_EXPLANATION
    output_format: GenerationFormat = GenerationFormat.MARKDOWN
    answer_length: GenerationLength = GenerationLength.MEDIUM
    generation_mode: GenerationMode = GenerationMode.STANDARD
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=16, le=65536)
    stop_sequences: list[str] | None = Field(default=None, max_length=16)
    timeout_seconds: float | None = Field(default=None, ge=1.0, le=600.0)
    require_citations: bool = True
    json_schema_description: str | None = Field(default=None, max_length=4000)
    custom_instruction: str | None = Field(default=None, max_length=2000)
    allocation_profile: (
        Literal["quick_answer", "standard_research", "deep_research", "long_form", "custom"]
        | None
    ) = None
    include_history: bool = True
    include_workspace_notes: bool = True
    stream: bool = False
    long_form_config: LongFormReportConfig | None = None


class GenerationStreamRequest(GenerationRequest):
    """Explicit streaming generation request (defaults stream=True)."""

    stream: bool = True


class GenerationResult(BaseModel):
    """Complete or partial/cancelled/failed persisted result of local LLM generation."""

    generation_id: str
    research_id: str
    status: GenerationStatus = GenerationStatus.PENDING
    partial: bool = False
    mode: GenerationMode = GenerationMode.STANDARD
    template_type: PromptTemplateType = PromptTemplateType.DETAILED_EXPLANATION
    output_format: GenerationFormat = GenerationFormat.MARKDOWN
    connection_id: str = ""
    provider_id: str = ""
    server_endpoint: str = ""
    model_id: str = ""
    content: str = ""
    parsed_json: dict[str, Any] | list[Any] | None = None
    finish_reason: str | None = None
    truncated: bool = False
    citations: list[CitationEntry] = Field(default_factory=list)
    source_map: dict[str, CitationEntry] = Field(default_factory=dict)
    citation_validation: CitationValidationReport | None = None
    usage: TokenUsage | None = None
    estimated_prompt_tokens: int = 0
    context_stats: GenerationContextStats | None = None
    long_form: LongFormReportResult | None = None
    error: str | None = None
    error_code: str | None = None
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
