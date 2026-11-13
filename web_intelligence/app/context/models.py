"""Domain models and validated API schemas for Stage 5 Adaptive Context Management."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CoverageAnalysis,
    EvidenceItem,
    ExtractedClaim,
)
from web_intelligence.app.schemas.llm import CapabilitySource, ChatMessage
from web_intelligence.app.schemas.research import ResearchPackage

BUDGET_BUCKET_KEYS = frozenset({"system", "query", "history", "evidence", "metadata"})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ContextConfigurationError(ValueError):
    """Raised when user or system context configuration violates model limits or invariants."""


class ContextLimitExceededError(ContextConfigurationError):
    """Raised when requested budget or context usage exceeds verified/configured limits."""


class AllocationProfile(str, Enum):
    """Adaptive allocation profiles tailored to task structure and output goals."""

    QUICK_ANSWER = "quick_answer"
    STANDARD_RESEARCH = "standard_research"
    DEEP_RESEARCH = "deep_research"
    LONG_FORM = "long_form"
    CUSTOM = "custom"

    # Compatibility alias for downstream generation defaults
    BALANCED = "standard_research"

    @classmethod
    def _missing_(cls, value: object) -> AllocationProfile | None:
        if isinstance(value, str) and value.strip().lower() == "balanced":
            return cls.STANDARD_RESEARCH
        return None


class ExpectedOutputLength(str, Enum):
    """Task-level expected generation length hint."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class CompressionStrategy(str, Enum):
    """Evidence context compression strategies."""

    NONE = "none"
    EXTRACTIVE = "extractive"
    DEDUPLICATION = "deduplication"
    STRUCTURED = "structured"
    MODEL_ASSISTED = "model_assisted"


class TokenEstimate(BaseModel):
    """Transparent token count estimate reporting exact vs conservative approximation."""

    tokens: int = Field(default=0, ge=0)
    is_approximate: bool = True
    strategy: str = "conservative_heuristic"
    tokenizer_available: bool = False
    tokenizer_id: str | None = None
    char_count: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    note: str = ""


class ModelContextCapabilities(BaseModel):
    """Normalized model context capabilities derived from adapter inspection."""

    model_name: str = ""
    provider: str = "unconfigured"
    connection_id: str | None = None
    has_active_model: bool = False
    max_context_tokens: int = Field(default=8192, ge=512)
    configured_context_limit: int | None = Field(default=None, ge=512)
    effective_context_limit: int = Field(default=8192, ge=512)
    max_output_tokens: int = Field(default=1024, ge=64)
    context_limit_source: CapabilitySource = CapabilitySource.UNKNOWN
    output_limit_source: CapabilitySource = CapabilitySource.UNKNOWN
    tokenizer_available: bool = False
    tokenizer_id: str | None = None
    estimation_strategy: str = "conservative_heuristic"
    estimation_is_approximate: bool = True
    supports_structured_output: bool = False
    supports_tool_calling: bool = False
    supports_streaming: bool = False
    warnings: list[str] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBudgetConfig(BaseModel):
    """Persisted user configuration controlling adaptive context allocation."""

    id: str = "default"
    profile: AllocationProfile = AllocationProfile.STANDARD_RESEARCH
    user_context_limit: int | None = Field(default=None, ge=512)
    output_reservation_tokens: int | None = Field(default=None, ge=64)
    safety_margin_tokens: int | None = Field(default=None, ge=0)
    custom_ratios: dict[str, float] | None = None
    expected_output_length: ExpectedOutputLength = ExpectedOutputLength.MEDIUM
    min_evidence_tokens: int = Field(default=256, ge=0)
    max_history_turns: int = Field(default=8, ge=0, le=100)
    compression_strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE
    allow_model_assisted_compression: bool = False
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("custom_ratios")
    @classmethod
    def _validate_custom_ratios(
        cls, value: dict[str, float] | None
    ) -> dict[str, float] | None:
        if value is None:
            return None
        unknown = set(value.keys()) - BUDGET_BUCKET_KEYS
        if unknown:
            raise ValueError(
                f"Unknown custom_ratios keys: {sorted(unknown)}. Allowed: {sorted(BUDGET_BUCKET_KEYS)}."
            )
        for key, weight in value.items():
            if weight < 0:
                raise ValueError(f"Custom ratio for '{key}' cannot be negative.")
        if sum(value.values()) <= 0:
            raise ValueError("custom_ratios weights must sum to greater than zero.")
        return {k: float(v) for k, v in value.items()}


class BudgetAllocation(BaseModel):
    """Concrete token allocation across all seven context components."""

    profile: AllocationProfile
    model_name: str
    effective_context_limit: int = Field(ge=512)
    model_max_context_tokens: int = Field(ge=512)
    system_instructions_tokens: int = Field(ge=0)
    user_query_tokens: int = Field(ge=0)
    conversation_history_tokens: int = Field(ge=0)
    research_evidence_tokens: int = Field(ge=0)
    structured_metadata_tokens: int = Field(ge=0)
    output_reservation_tokens: int = Field(ge=64)
    safety_margin_tokens: int = Field(ge=0)
    input_budget_tokens: int = Field(ge=0)
    total_planned_tokens: int = Field(ge=0)
    policy_rationale: str = ""
    tokenizer_available: bool = False
    estimation_is_approximate: bool = True
    warnings: list[str] = Field(default_factory=list)


class EvidenceScoreBreakdown(BaseModel):
    """Multi-signal deterministic or learned score breakdown for passage selection."""

    relevance: float = Field(ge=0.0, le=1.0)
    topic_coverage: float = Field(ge=0.0, le=1.0)
    source_independence: float = Field(ge=0.0, le=1.0)
    provenance: float = Field(ge=0.0, le=1.0)
    claim_support: float = Field(ge=0.0, le=1.0)
    conflict_priority: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    token_cost_penalty: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0)


class SelectedEvidenceRecord(BaseModel):
    """Evidence passage selected for context inclusion with full source provenance."""

    evidence_id: str
    citation_label: str
    source_id: str
    source_url: str
    source_domain: str
    source_title: str | None = None
    passage_text: str
    original_token_count: int = Field(ge=0)
    allocated_token_count: int = Field(ge=0)
    selection_reason: str
    is_conflict_evidence: bool = False
    conflict_ids: list[str] = Field(default_factory=list)
    covered_subtopics: list[str] = Field(default_factory=list)
    scores: EvidenceScoreBreakdown


class ExcludedEvidenceRecord(BaseModel):
    """Evidence passage excluded from active context with explicit reason for inspection."""

    evidence_id: str
    source_id: str
    source_url: str
    source_domain: str
    source_title: str | None = None
    passage_preview: str
    original_token_count: int = Field(ge=0)
    exclusion_reason: str
    scores: EvidenceScoreBreakdown | None = None


class StructuredFactEntry(BaseModel):
    """Structured claim/fact/date/entity unit retaining full source citations."""

    fact_id: str
    claim_text: str
    entity: str | None = None
    attribute: str | None = None
    value: str | None = None
    dates: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    is_conflicted: bool = False


class CompressedEvidenceUnit(BaseModel):
    """Compressed representation of one or more evidence items preserving provenance."""

    unit_id: str
    strategy: CompressionStrategy
    compressed_text: str
    original_evidence_ids: list[str] = Field(default_factory=list)
    supporting_source_ids: list[str] = Field(default_factory=list)
    supporting_source_urls: list[str] = Field(default_factory=list)
    supporting_domains: list[str] = Field(default_factory=list)
    citation_labels: list[str] = Field(default_factory=list)
    original_tokens: int = Field(ge=0)
    compressed_tokens: int = Field(ge=0)
    provenance_verified: bool = True
    structured_facts: list[StructuredFactEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CompressionResult(BaseModel):
    """Outcome of context compression over selected evidence items."""

    strategy_requested: CompressionStrategy
    strategy_applied: CompressionStrategy
    units: list[CompressedEvidenceUnit] = Field(default_factory=list)
    original_total_tokens: int = Field(default=0, ge=0)
    compressed_total_tokens: int = Field(default=0, ge=0)
    compression_ratio: float = Field(default=1.0, ge=0.0)
    deduplicated_count: int = Field(default=0, ge=0)
    independent_sources_preserved: int = Field(default=0, ge=0)
    provenance_preserved: bool = True
    model_assisted_used: bool = False
    model_assisted_fallback_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    """Single current-turn or session conversation entry."""

    turn_id: str
    research_id: str = "global"
    scope: str = "session"  # "current_turn" | "session"
    role: str  # "user" | "assistant" | "system"
    content: str
    created_at: datetime = Field(default_factory=_utc_now)
    decisions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)
    compacted: bool = False

    @field_validator("role")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        clean = value.strip().lower()
        if clean not in {"user", "assistant", "system"}:
            raise ValueError("role must be 'user', 'assistant', or 'system'.")
        return clean


class CompactedHistorySummary(BaseModel):
    """Deterministic compaction of older history preserving decisions and open questions."""

    summary_text: str
    compacted_turn_ids: list[str] = Field(default_factory=list)
    preserved_decisions: list[str] = Field(default_factory=list)
    preserved_unresolved_questions: list[str] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)


class WorkspaceNote(BaseModel):
    """User-controlled persistent workspace note optionally injected into history."""

    note_id: str
    workspace_id: str = "default"
    title: str
    content: str
    enabled: bool = True
    user_controlled: bool = True
    token_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class HistoryPackage(BaseModel):
    """Combined conversational history and persistent notes respecting token limits."""

    current_turn_items: list[ConversationTurn] = Field(default_factory=list)
    session_turns: list[ConversationTurn] = Field(default_factory=list)
    compacted_summary: CompactedHistorySummary | None = None
    workspace_notes: list[WorkspaceNote] = Field(default_factory=list)
    total_tokens: int = Field(default=0, ge=0)
    compacted: bool = False
    warnings: list[str] = Field(default_factory=list)


class SourceReferenceEntry(BaseModel):
    """Canonical citation table entry linking [E#] labels to verified source URLs."""

    citation_label: str
    source_id: str
    evidence_id: str
    url: str
    canonical_url: str
    domain: str
    title: str | None = None
    classification: str = "unknown"
    published_at: datetime | None = None


class ContextSectionTokenUsage(BaseModel):
    """Measured or conservatively estimated token usage for an individual context section."""

    section: str
    estimated_tokens: int = Field(ge=0)
    allocated_tokens: int = Field(ge=0)
    is_approximate: bool = True


class ContextPackage(BaseModel):
    """Model-independent structured context package ready for adapter conversion and validation."""

    context_id: str
    research_id: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    system_instructions: str
    original_user_question: str
    research_objective: str
    selected_evidence: list[SelectedEvidenceRecord] = Field(default_factory=list)
    excluded_evidence: list[ExcludedEvidenceRecord] = Field(default_factory=list)
    compression: CompressionResult
    supporting_claims: list[ExtractedClaim] = Field(default_factory=list)
    conflicting_claims: list[ClaimConflict] = Field(default_factory=list)
    source_references: list[SourceReferenceEntry] = Field(default_factory=list)
    history: HistoryPackage
    research_coverage: CoverageAnalysis = Field(default_factory=CoverageAnalysis)
    capabilities: ModelContextCapabilities
    budget_allocation: BudgetAllocation
    section_token_usage: list[ContextSectionTokenUsage] = Field(default_factory=list)
    total_input_tokens: int = Field(default=0, ge=0)
    output_reservation_tokens: int = Field(default=0, ge=0)
    total_with_output_tokens: int = Field(default=0, ge=0)
    estimation_is_approximate: bool = True
    estimation_strategy: str = "conservative_heuristic"
    rendered_evidence_block: str = ""
    rendered_user_prompt: str = ""


class ContextValidationIssue(BaseModel):
    """Actionable error or warning produced during pre-inference context validation."""

    code: str
    severity: str = "error"  # "error" | "warning"
    section: str | None = None
    message: str
    remediation: str


class ContextValidationReport(BaseModel):
    """Complete pre-inference validation report for a ContextPackage."""

    is_valid: bool
    estimated_input_tokens: int = Field(ge=0)
    output_reservation_tokens: int = Field(ge=0)
    total_required_tokens: int = Field(ge=0)
    active_context_limit: int = Field(ge=0)
    remaining_tokens: int
    estimation_is_approximate: bool = True
    issues: list[ContextValidationIssue] = Field(default_factory=list)
    checked_rules: list[str] = Field(default_factory=list)


# --- API Request & Response Schemas ---


class ContextCapabilitiesResponse(BaseModel):
    """Response schema for GET /api/v1/context/capabilities."""

    capabilities: ModelContextCapabilities
    available_profiles: list[AllocationProfile]
    compression_strategies: list[CompressionStrategy]
    default_profile: AllocationProfile


class ContextBudgetResponse(BaseModel):
    """Response schema for GET /api/v1/context/budget and PUT /api/v1/context/budget."""

    config: ContextBudgetConfig
    capabilities: ModelContextCapabilities
    effective_allocation: BudgetAllocation


class ContextBudgetUpdateRequest(BaseModel):
    """Validated payload for updating active context budget configuration."""

    profile: AllocationProfile | None = None
    user_context_limit: int | None = Field(default=None, ge=512)
    output_reservation_tokens: int | None = Field(default=None, ge=64)
    safety_margin_tokens: int | None = Field(default=None, ge=0)
    custom_ratios: dict[str, float] | None = None
    expected_output_length: ExpectedOutputLength | None = None
    min_evidence_tokens: int | None = Field(default=None, ge=0)
    max_history_turns: int | None = Field(default=None, ge=0, le=100)
    compression_strategy: CompressionStrategy | None = None
    allow_model_assisted_compression: bool | None = None
    clear_user_context_limit: bool = False
    clear_output_reservation: bool = False
    clear_safety_margin: bool = False

    @field_validator("custom_ratios")
    @classmethod
    def _validate_ratios(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        return ContextBudgetConfig._validate_custom_ratios(value)


class ConversationTurnInput(BaseModel):
    """Input conversation turn for inline preview/validation requests."""

    role: str
    content: str = Field(..., min_length=1)
    scope: str = "session"
    decisions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class ContextPreviewRequest(BaseModel):
    """Request schema for POST /api/v1/context/preview."""

    research_id: str | None = None
    research_package: ResearchPackage | None = None
    question: str | None = None
    research_objective: str | None = None
    system_instructions: str | None = None
    connection_id: str | None = None
    profile: AllocationProfile | None = None
    expected_output_length: ExpectedOutputLength | None = None
    user_context_limit: int | None = Field(default=None, ge=512)
    output_reservation_tokens: int | None = Field(default=None, ge=64)
    safety_margin_tokens: int | None = Field(default=None, ge=0)
    custom_ratios: dict[str, float] | None = None
    compression_strategy: CompressionStrategy | None = None
    allow_model_assisted_compression: bool = False
    conversation_history: list[ConversationTurnInput] = Field(default_factory=list)
    include_workspace_notes: bool = True

    @model_validator(mode="after")
    def _check_target(self) -> ContextPreviewRequest:
        if not self.research_id and self.research_package is None and not self.question:
            raise ValueError(
                "Provide at least one of 'research_id', 'research_package', or 'question'."
            )
        return self


class ContextPreviewResponse(BaseModel):
    """Response schema for POST /api/v1/context/preview and GET /api/v1/context/{research_id}."""

    context_package: ContextPackage
    validation: ContextValidationReport
    adapter_messages: list[ChatMessage] = Field(default_factory=list)


class ContextValidateRequest(BaseModel):
    """Request schema for POST /api/v1/context/validate."""

    context_package: ContextPackage | None = None
    preview_request: ContextPreviewRequest | None = None
    connection_id: str | None = None
    user_context_limit: int | None = Field(default=None, ge=512)

    @model_validator(mode="after")
    def _check_payload(self) -> ContextValidateRequest:
        if self.context_package is None and self.preview_request is None:
            raise ValueError("Provide either 'context_package' or 'preview_request'.")
        return self


class ContextCompressRequest(BaseModel):
    """Request schema for POST /api/v1/context/compress."""

    research_id: str | None = None
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    conflicts: list[ClaimConflict] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE
    target_tokens: int | None = Field(default=None, ge=64)
    allow_model_assisted: bool = False
    connection_id: str | None = None

    @model_validator(mode="after")
    def _check_source(self) -> ContextCompressRequest:
        if not self.research_id and not self.evidence_items:
            raise ValueError("Provide 'research_id' or non-empty 'evidence_items'.")
        return self


class ContextCompressResponse(BaseModel):
    """Response schema for POST /api/v1/context/compress."""

    compression: CompressionResult
    original_evidence: list[EvidenceItem] = Field(default_factory=list)
    preserved_conflict_ids: list[str] = Field(default_factory=list)


class WorkspaceNoteCreate(BaseModel):
    """Payload to create a user-controlled persistent workspace research note."""

    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=20000)
    workspace_id: str = "default"
    enabled: bool = True


class WorkspaceNoteUpdate(BaseModel):
    """Payload to update or toggle a workspace note."""

    title: str | None = Field(default=None, min_length=1, max_length=200)
    content: str | None = Field(default=None, min_length=1, max_length=20000)
    enabled: bool | None = None
