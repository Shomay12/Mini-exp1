"""Domain models, state schemas, and API DTOs for Stage 7 Agentic Research Orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from web_intelligence.app.schemas.documents import SourceRecord
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CoverageAnalysis,
    EvidenceItem,
    ExtractedClaim,
)
from web_intelligence.app.schemas.generation import GenerationResult
from web_intelligence.app.schemas.search import DisclosedSearchQuery, ProviderDiagnostic


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class ResearchDepth(str, Enum):
    """Presets controlling research depth and default bounded execution limits."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class AgentSessionStatus(str, Enum):
    """Explicit lifecycle state machine for autonomous agent research sessions."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Execution status of an individual ResearchTask node in the DAG."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"


class EvidenceSupportStatus(str, Enum):
    """Phase 4 distinction of evidence strength (never equates relevance with truth)."""

    CONFIRMED = "confirmed"
    CORROBORATED = "corroborated"
    WEAK = "weak"
    CONFLICTING = "conflicting"
    UNVERIFIED = "unverified"


class GapCategory(str, Enum):
    """Taxonomy of research gaps evaluated after each iteration."""

    UNANSWERED_QUESTION = "unanswered_question"
    MISSING_CATEGORY = "missing_category"
    SINGLE_SOURCE_CLAIM = "single_source_claim"
    CONFLICTING_RESULTS = "conflicting_results"
    OUTDATED_INFORMATION = "outdated_information"
    WEAK_SUPPORT = "weak_support"
    MISSING_PRIMARY_SOURCE = "missing_primary_source"


class StoppingReason(str, Enum):
    """Explicit auditable reasons the autonomous research loop halted."""

    OBJECTIVES_COVERED = "objectives_covered"
    DIMINISHING_RETURNS = "diminishing_returns"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    TIME_BUDGET_EXHAUSTED = "time_budget_exhausted"
    SEARCH_BUDGET_REACHED = "search_budget_reached"
    FETCH_BUDGET_REACHED = "fetch_budget_reached"
    INFERENCE_BUDGET_REACHED = "inference_budget_reached"
    TOKEN_BUDGET_REACHED = "token_budget_reached"
    TOOLS_UNAVAILABLE = "tools_unavailable"
    USER_CANCELLED = "user_cancelled"
    MODEL_UNAVAILABLE = "model_unavailable"
    PAUSED_BY_USER = "paused_by_user"
    WAITING_USER_APPROVAL = "waiting_user_approval"
    EXECUTION_ERROR = "execution_error"


class TraceEventType(str, Enum):
    """Structured event classes captured in the operational execution trace."""

    SESSION_STARTED = "session_started"
    PLAN_CREATED = "plan_created"
    PLAN_UPDATED = "plan_updated"
    TASK_CREATED = "task_created"
    TASK_SKIPPED = "task_skipped"
    TOOL_SELECTED = "tool_selected"
    TOOL_EXECUTED = "tool_executed"
    TOOL_REJECTED = "tool_rejected"
    SOURCE_RETAINED = "source_retained"
    SOURCE_REJECTED = "source_rejected"
    EVIDENCE_EVALUATED = "evidence_evaluated"
    GAP_IDENTIFIED = "gap_identified"
    ITERATION_STARTED = "iteration_started"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    SESSION_PAUSED = "session_paused"
    SESSION_RESUMED = "session_resumed"
    STOPPING_DECISION = "stopping_decision"
    SYNTHESIS_COMPLETED = "synthesis_completed"
    ERROR = "error"


class ToolPermission(str, Enum):
    """Capability & safety classification attached to registered tools."""

    READ_LOCAL = "read_local"
    LOCAL_INFERENCE = "local_inference"
    EXTERNAL_SEARCH = "external_search"
    EXTERNAL_FETCH = "external_fetch"


class ResearchTask(BaseModel):
    """Structured task node managed by TaskManager and ResearchPlanner."""

    task_id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=3, max_length=600)
    objective: str = Field(..., min_length=3, max_length=800)
    expected_evidence: str = Field(default="Primary documentation or empirical findings")
    entities_keywords: list[str] = Field(default_factory=list, max_length=20)
    suggested_tools: list[str] = Field(default_factory=lambda: ["web_search", "url_retrieve"])
    priority: int = Field(default=3, ge=1, le=5)
    dependencies: list[str] = Field(default_factory=list, max_length=12)
    completion_criteria: str = Field(
        default="At least one relevant retrieved passage or corroborated claim with provenance."
    )
    estimated_cost: float = Field(default=1.0, ge=0.0, le=100.0)
    depth: int = Field(default=0, ge=0, le=10)
    iteration_created: int = Field(default=0, ge=0)
    parent_gap_id: str | None = None
    origin: Literal["initial_plan", "gap_followup", "user_edit"] = "initial_plan"
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    assigned_tools_used: list[str] = Field(default_factory=list)
    search_queries_used: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    result_summary: str | None = None
    error: str | None = None
    fingerprint: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class ResearchPlanSpec(BaseModel):
    """Validated structured research plan produced by AgentResearchPlanner."""

    plan_id: str
    version: int = Field(default=1, ge=1)
    objective: str
    normalized_objective: str
    planned_by: str = "deterministic_planner"
    model_id: str | None = None
    planning_notes: str = ""
    tasks: list[ResearchTask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ToolDescriptor(BaseModel):
    """Metadata and safety contract for a tool in ToolRegistry."""

    name: str = Field(..., min_length=1, max_length=64)
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    permissions: list[ToolPermission] = Field(default_factory=list)
    timeout_seconds: float = Field(default=20.0, gt=0.1, le=300.0)
    estimated_cost: float = Field(default=1.0, ge=0.0, le=100.0)
    requires_external_network: bool = False
    requires_approval: bool = False
    enabled: bool = True


class ToolProposal(BaseModel):
    """Validated action proposed by ToolSelector for a task."""

    proposal_id: str
    task_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    selected_by: str = "deterministic_selector"
    requires_approval: bool = False


class ToolExecutionResult(BaseModel):
    """Normalized execution envelope returned by ToolRegistry.execute()."""

    tool_name: str
    task_id: str
    success: bool
    sanitized_inputs: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    error_code: str | None = None
    duration_ms: int = 0
    estimated_tokens_consumed: int = 0
    used_external_network: bool = False
    executed_at: datetime = Field(default_factory=utc_now)


class SourceDecisionRecord(BaseModel):
    """Traceable outcome explaining why an observed source was retained or rejected."""

    source_id: str
    url: str
    domain: str
    retained: bool
    reason: str
    relevance_score: float = 0.0
    is_duplicate: bool = False


class NormalizedObservation(BaseModel):
    """Structured observation normalized from one or more tool executions for a task."""

    task_id: str
    iteration: int
    tool_names: list[str] = Field(default_factory=list)
    disclosed_queries: list[DisclosedSearchQuery] = Field(default_factory=list)
    provider_diagnostics: list[ProviderDiagnostic] = Field(default_factory=list)
    new_sources: list[SourceRecord] = Field(default_factory=list)
    new_evidence: list[EvidenceItem] = Field(default_factory=list)
    new_claims: list[ExtractedClaim] = Field(default_factory=list)
    new_conflicts: list[ClaimConflict] = Field(default_factory=list)
    source_decisions: list[SourceDecisionRecord] = Field(default_factory=list)
    snippet_only_hits: int = 0
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)


class TaskEvidenceAssessment(BaseModel):
    """Per-task evidence evaluation across provenance, independence, recency, and quality."""

    task_id: str
    question: str
    support_status: EvidenceSupportStatus = EvidenceSupportStatus.UNVERIFIED
    answered: bool = False
    relevance_score: float = 0.0
    directness_score: float = 0.0
    primary_source_count: int = 0
    secondary_source_count: int = 0
    reference_source_count: int = 0
    independent_domain_count: int = 0
    mirror_or_duplicate_count: int = 0
    snippet_only_count: int = 0
    dated_source_count: int = 0
    newest_source_date: datetime | None = None
    methodological_cues_found: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    unsupported_claim_count: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class EvidenceEvaluationReport(BaseModel):
    """Session-wide Phase 4 evidence quality evaluation."""

    iteration: int
    overall_support_status: EvidenceSupportStatus = EvidenceSupportStatus.UNVERIFIED
    objective_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    total_sources_evaluated: int = 0
    unique_primary_domains: int = 0
    total_accepted_evidence: int = 0
    total_rejected_sources: int = 0
    confirmed_tasks: int = 0
    corroborated_tasks: int = 0
    weak_tasks: int = 0
    conflicting_tasks: int = 0
    unverified_tasks: int = 0
    conflict_count: int = 0
    task_assessments: list[TaskEvidenceAssessment] = Field(default_factory=list)
    coverage_analysis: CoverageAnalysis | None = None
    summary: str = ""


class ResearchGap(BaseModel):
    """Actionable knowledge or provenance gap detected by GapAnalyzer."""

    gap_id: str
    category: GapCategory
    severity: Literal["high", "medium", "low"] = "medium"
    description: str
    related_task_id: str | None = None
    related_claim_ids: list[str] = Field(default_factory=list)
    related_conflict_id: str | None = None
    suggested_followup_question: str
    suggested_objective: str
    expected_evidence: str = "Independent or primary source evidence resolving this gap"
    suggested_tools: list[str] = Field(default_factory=lambda: ["web_search", "url_retrieve"])
    priority: int = Field(default=4, ge=1, le=5)
    resolved: bool = False
    addressed_by_task_id: str | None = None


class GapAnalysisReport(BaseModel):
    """Iteration gap analysis output guiding optional follow-up planning."""

    iteration: int
    gaps: list[ResearchGap] = Field(default_factory=list)
    high_priority_count: int = 0
    medium_priority_count: int = 0
    low_priority_count: int = 0
    summary: str = ""


class ExecutionBudgetConfig(BaseModel):
    """Conservative, user-configurable hard bounds enforced independently of the LLM."""

    max_iterations: int = Field(default=3, ge=1, le=12)
    max_search_calls: int = Field(default=8, ge=1, le=60)
    max_page_retrievals: int = Field(default=12, ge=1, le=100)
    max_concurrent_tasks: int = Field(default=2, ge=1, le=8)
    max_execution_seconds: float = Field(default=120.0, ge=5.0, le=1800.0)
    max_inference_calls: int = Field(default=16, ge=0, le=120)
    max_estimated_tokens: int = Field(default=64_000, ge=1000, le=1_000_000)
    max_research_depth: int = Field(default=2, ge=0, le=5)
    tool_timeout_seconds: float = Field(default=25.0, ge=1.0, le=180.0)
    max_followups_per_iteration: int = Field(default=3, ge=0, le=8)
    require_approval_for_external_search: bool = False
    require_approval_for_remote_urls: bool = False


class ExecutionBudgetUsage(BaseModel):
    """Live cumulative consumption counters tracked against ExecutionBudgetConfig."""

    iterations_completed: int = 0
    search_calls: int = 0
    page_retrievals: int = 0
    inference_calls: int = 0
    estimated_tokens: int = 0
    elapsed_seconds: float = 0.0
    max_depth_reached: int = 0
    tool_calls_total: int = 0
    tool_failures_total: int = 0


class ResearchTraceEntry(BaseModel):
    """Chronological, auditable operational trace entry (no private CoT)."""

    trace_id: str
    research_id: str
    sequence: int = Field(ge=1)
    iteration: int = Field(default=0, ge=0)
    phase: str
    event_type: TraceEventType
    task_id: str | None = None
    tool_name: str | None = None
    sanitized_inputs: dict[str, Any] = Field(default_factory=dict)
    outcome: str = ""
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class PendingApprovalRequest(BaseModel):
    """Explicit user authorization gate for sensitive or external tool invocations."""

    approval_id: str
    research_id: str
    task_id: str
    tool_name: str
    reason: str
    external_network: bool = True
    proposed_inputs: dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "approved", "rejected"] = "pending"
    decision_note: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class StoppingDecision(BaseModel):
    """Multi-factor stopping verdict returned by StoppingPolicy."""

    should_stop: bool
    reason: StoppingReason | None = None
    explanation: str
    objective_coverage_ratio: float = 0.0
    high_priority_gaps_remaining: int = 0
    budget_exhausted_dimensions: list[str] = Field(default_factory=list)
    unresolved_items_summary: list[str] = Field(default_factory=list)


class ResearchMetrics(BaseModel):
    """Operational telemetry surfaced for session summary and UI progress."""

    tasks_total: int = 0
    tasks_completed: int = 0
    tasks_pending: int = 0
    tasks_failed: int = 0
    tasks_skipped: int = 0
    sources_collected: int = 0
    sources_retained: int = 0
    sources_rejected: int = 0
    evidence_accepted: int = 0
    evidence_rejected: int = 0
    claims_extracted: int = 0
    conflicts_detected: int = 0
    gaps_identified: int = 0
    gaps_unresolved: int = 0
    trace_entries_count: int = 0


class ResearchSession(BaseModel):
    """Complete persistent state of an autonomous Stage 7 research session."""

    research_id: str
    original_query: str
    research_objective: str
    research_depth: ResearchDepth = ResearchDepth.STANDARD
    status: AgentSessionStatus = AgentSessionStatus.PENDING
    current_iteration: int = 0
    current_phase: str = "pending"
    current_task_id: str | None = None
    research_plan: ResearchPlanSpec | None = None
    completed_tasks: list[ResearchTask] = Field(default_factory=list)
    pending_tasks: list[ResearchTask] = Field(default_factory=list)
    failed_tasks: list[ResearchTask] = Field(default_factory=list)
    collected_sources: list[SourceRecord] = Field(default_factory=list)
    evidence_records: list[EvidenceItem] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    conflicts: list[ClaimConflict] = Field(default_factory=list)
    disclosed_queries: list[DisclosedSearchQuery] = Field(default_factory=list)
    provider_diagnostics: list[ProviderDiagnostic] = Field(default_factory=list)
    source_decisions: list[SourceDecisionRecord] = Field(default_factory=list)
    identified_gaps: list[ResearchGap] = Field(default_factory=list)
    execution_budget: ExecutionBudgetConfig = Field(default_factory=ExecutionBudgetConfig)
    budget_usage: ExecutionBudgetUsage = Field(default_factory=ExecutionBudgetUsage)
    research_metrics: ResearchMetrics = Field(default_factory=ResearchMetrics)
    evidence_evaluation: EvidenceEvaluationReport | None = None
    stopping_decision: StoppingDecision | None = None
    pending_approvals: list[PendingApprovalRequest] = Field(default_factory=list)
    research_package_id: str | None = None
    generation_result: GenerationResult | None = None
    providers: list[str] | None = None
    active_connection_id: str | None = None
    active_model_id: str | None = None
    use_local_llm: bool = True
    allow_external_search: bool = True
    require_user_approval_for_external: bool = False
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class AgentResearchRequest(BaseModel):
    """API input payload for POST /api/v1/agent/research."""

    query: str = Field(..., min_length=3, max_length=2000)
    research_objective: str | None = Field(default=None, max_length=2000)
    research_depth: ResearchDepth = ResearchDepth.STANDARD
    connection_id: str | None = Field(default=None, max_length=120)
    model_id: str | None = Field(default=None, max_length=200)
    use_local_llm: bool = True
    providers: list[str] | None = None
    allow_external_search: bool = True
    require_user_approval_for_external: bool = False
    execution_budget: ExecutionBudgetConfig | None = None
    max_results_per_query: int = Field(default=5, ge=1, le=20)
    chunk_target_tokens: int = Field(default=180, ge=40, le=1000)
    temporal_focus: bool = False

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        clean = value.strip()
        if len(clean) < 3:
            raise ValueError("Research query must be at least 3 non-whitespace characters.")
        return clean


class AgentPlanTaskEdit(BaseModel):
    """Editable fields on a pending ResearchTask via plan inspection/update."""

    task_id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=3, max_length=600)
    objective: str = Field(..., min_length=3, max_length=800)
    expected_evidence: str = Field(default="Primary documentation or empirical findings")
    entities_keywords: list[str] = Field(default_factory=list, max_length=20)
    suggested_tools: list[str] = Field(default_factory=lambda: ["web_search", "url_retrieve"])
    priority: int = Field(default=3, ge=1, le=5)
    dependencies: list[str] = Field(default_factory=list, max_length=12)
    completion_criteria: str = Field(
        default="At least one relevant retrieved passage or corroborated claim."
    )


class AgentPlanEditRequest(BaseModel):
    """Validated user plan update payload (cannot alter already completed/running tasks)."""

    tasks: list[AgentPlanTaskEdit] = Field(..., min_length=1, max_length=20)


class AgentApprovalDecisionRequest(BaseModel):
    """User authorization verdict on a pending external/sensitive action."""

    approval_id: str | None = None
    approved: bool = True
    decision_note: str | None = Field(default=None, max_length=400)
    resume_execution: bool = True


class AgentSessionSummary(BaseModel):
    """Compact list item returned by GET /api/v1/agent/research."""

    research_id: str
    original_query: str
    research_objective: str
    research_depth: ResearchDepth
    status: AgentSessionStatus
    current_iteration: int
    current_phase: str
    tasks_completed: int = 0
    tasks_total: int = 0
    sources_retained: int = 0
    conflicts_detected: int = 0
    gaps_unresolved: int = 0
    active_model_id: str | None = None
    created_at: datetime
    updated_at: datetime


class AgentPlanResponse(BaseModel):
    """Response for GET/PUT /api/v1/agent/research/{research_id}/plan."""

    research_id: str
    status: AgentSessionStatus
    plan: ResearchPlanSpec | None = None


class AgentTasksResponse(BaseModel):
    """Response for GET /api/v1/agent/research/{research_id}/tasks."""

    research_id: str
    status: AgentSessionStatus
    current_task_id: str | None = None
    completed_tasks: list[ResearchTask] = Field(default_factory=list)
    pending_tasks: list[ResearchTask] = Field(default_factory=list)
    failed_tasks: list[ResearchTask] = Field(default_factory=list)


class AgentTraceResponse(BaseModel):
    """Response for GET /api/v1/agent/research/{research_id}/trace."""

    research_id: str
    status: AgentSessionStatus
    entries: list[ResearchTraceEntry] = Field(default_factory=list)


class AgentGapsResponse(BaseModel):
    """Response for GET /api/v1/agent/research/{research_id}/gaps."""

    research_id: str
    status: AgentSessionStatus
    gaps: list[ResearchGap] = Field(default_factory=list)
    unresolved_count: int = 0
    resolved_count: int = 0
    evidence_evaluation: EvidenceEvaluationReport | None = None
    stopping_decision: StoppingDecision | None = None
