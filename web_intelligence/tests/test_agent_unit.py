"""Comprehensive unit tests for Stage 7 Agentic Research Orchestration components."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from web_intelligence.app.agent.evidence_evaluator import EvidenceQualityEvaluator
from web_intelligence.app.agent.execution_budget import (
    ExecutionBudgetTracker,
    default_budget_for_depth,
)
from web_intelligence.app.agent.gap_analyzer import GapAnalyzer
from web_intelligence.app.agent.models import (
    AgentPlanEditRequest,
    AgentPlanTaskEdit,
    AgentSessionStatus,
    EvidenceEvaluationReport,
    EvidenceSupportStatus,
    ExecutionBudgetConfig,
    ExecutionBudgetUsage,
    GapAnalysisReport,
    GapCategory,
    ResearchDepth,
    ResearchGap,
    ResearchPlanSpec,
    ResearchSession,
    ResearchTask,
    SourceDecisionRecord,
    StoppingReason,
    TaskStatus,
    TraceEventType,
)
from web_intelligence.app.agent.research_memory import ResearchWorkingMemory
from web_intelligence.app.agent.research_planner import AgentResearchPlanner
from web_intelligence.app.agent.research_state import (
    InvalidStateTransitionError,
    ResearchSessionStateManager,
    is_duplicate_task,
)
from web_intelligence.app.agent.research_trace import ResearchTraceRecorder
from web_intelligence.app.agent.stopping_policy import StoppingPolicy
from web_intelligence.app.agent.task_manager import PlanValidationError, TaskManager
from web_intelligence.app.agent.tool_registry import (
    FORBIDDEN_TOOL_NAMES,
    ToolApprovalRequiredError,
    ToolRegistry,
)
from web_intelligence.app.agent.tool_selector import ToolSelector
from web_intelligence.app.core.config import Settings
from web_intelligence.app.llm.base import UniversalLocalModelAdapter
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.providers.fixtures import (
    FixturePage,
    build_fixture_transport,
    fixture_dns_resolver,
)
from web_intelligence.app.providers.router import SearchProviderRouter
from web_intelligence.app.retrieval.search_engine import WebRetrievalEngine
from web_intelligence.app.schemas.documents import (
    SourceClassification,
    SourceProvenance,
    SourceRecord,
)
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CorroborationSignals,
    CorroborationStatus,
    EvidenceItem,
    ExtractedClaim,
    RelevanceBreakdown,
)
from web_intelligence.app.schemas.llm import ModelCapabilities
from web_intelligence.app.storage.database import DatabaseManager
from web_intelligence.app.storage.repositories import ResearchRepository


class _FakeAdapter(UniversalLocalModelAdapter):
    provider_name = "fake_local"

    def __init__(self, responses: list[Any], *, supports_json: bool = True) -> None:
        self._responses = list(responses)
        self.calls: int = 0
        self._caps = ModelCapabilities(
            model_name="fake-4b",
            backend="fake",
            max_context_tokens=8192,
            recommended_output_tokens=1024,
            supports_json_mode=supports_json,
        )

    def capabilities(self) -> ModelCapabilities:
        return self._caps

    async def validate_connection(self):  # type: ignore[override]
        return True

    async def test_connection(self):  # type: ignore[override]
        return True, "ok", ["fake-4b"]

    async def inspect_model_capabilities(self, model_name: str | None = None):  # type: ignore[override]
        return self._caps

    async def list_models(self):  # type: ignore[override]
        return ["fake-4b"]

    async def chat(self, request):  # type: ignore[override]
        raise NotImplementedError

    async def stream_chat(self, request):  # type: ignore[override]
        if False:
            yield ""

    async def generate_json(self, request):  # type: ignore[override]
        self.calls += 1
        item = self._responses.pop(0) if self._responses else {}
        if isinstance(item, Exception):
            raise item
        return item


async def _make_registry(tmp_path: Path, *, fixtures: list[FixturePage] | None = None):
    settings = Settings(
        database_path=tmp_path / "unit.db",
        search_providers=["mock"],
        enable_mock_provider=True,
        allow_external_search=False,
    )
    db = DatabaseManager(settings.database_path)
    await db.initialize()
    repo = ResearchRepository(db)
    transport_fn = (lambda: build_fixture_transport(fixtures)) if fixtures is not None else build_fixture_transport
    gateway = PrivacyGateway(settings, transport_factory=transport_fn)
    router = SearchProviderRouter(settings, gateway)
    if fixtures is not None:
        router._registry["mock"].set_fixtures(fixtures)  # type: ignore[attr-defined]
    retrieval = WebRetrievalEngine(settings, gateway, router=router, dns_resolver=fixture_dns_resolver)
    reg = ToolRegistry(router=router, retrieval_engine=retrieval, repository=repo)
    return reg, repo, gateway


def _make_evidence(
    *,
    ev_id: str,
    src_id: str,
    domain: str,
    text: str,
    classification: SourceClassification = SourceClassification.SECONDARY,
    corroboration_status: CorroborationStatus = CorroborationStatus.SINGLE_SOURCE,
    independent_sources: int = 1,
    relevance: float = 0.85,
    published_at: datetime | None = None,
    conflict_ids: list[str] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=ev_id,
        source_id=src_id,
        source_url=f"https://{domain}/article",
        source_domain=domain,
        source_title="Sample Article",
        chunk_index=0,
        section_heading="Overview",
        passage_text=text,
        char_start=0,
        char_end=len(text),
        token_estimate=max(5, len(text.split())),
        relevance=RelevanceBreakdown(combined_score=relevance, bm25_score=relevance),
        provenance=SourceProvenance(
            domain=domain,
            publisher=domain,
            source_classification=classification,
            classification_reason="test",
            published_at=published_at,
            original_url=f"https://{domain}/article",
            canonical_url=f"https://{domain}/article",
        ),
        corroboration=CorroborationSignals(
            corroboration_status=corroboration_status,
            independent_supporting_sources=independent_sources,
            conflicting_claim_ids=conflict_ids or [],
        ),
    )


@pytest.mark.asyncio
async def test_planner_llm_validation_retry_forbidden_tools_and_fallback(tmp_path: Path) -> None:
    registry, _, _ = await _make_registry(tmp_path)
    planner = AgentResearchPlanner(registry)
    session = ResearchSession(
        research_id="ag-plan-1",
        original_query="Compare SQLite FTS5 vs dense embeddings for small local models",
        research_objective="Evaluate retrieval approaches for compact local AI systems.",
        research_depth=ResearchDepth.STANDARD,
    )
    budget = ExecutionBudgetTracker(default_budget_for_depth(ResearchDepth.STANDARD))

    # 1. Invalid response on first try (uses forbidden tool 'python'), valid JSON on retry
    adapter = _FakeAdapter(
        [
            {
                "tasks": [
                    {
                        "task_id": "t1",
                        "question": "Run benchmark script",
                        "suggested_tools": ["python"],
                    }
                ]
            },
            {
                "planning_notes": "Two clean non-overlapping investigation facets.",
                "tasks": [
                    {
                        "task_id": "task-1",
                        "question": "Evaluate SQLite FTS5 lexical retrieval metrics",
                        "objective": "Gather BM25 & FTS5 benchmarks.",
                        "suggested_tools": ["web_search", "url_retrieve"],
                        "priority": 5,
                        "dependencies": [],
                    },
                    {
                        "task_id": "task-2",
                        "question": "Compare dense embeddings vs hybrid retrieval",
                        "objective": "Verify recall on entity/date lookups.",
                        "suggested_tools": ["web_search"],
                        "priority": 4,
                        "dependencies": ["task-1"],
                    },
                ],
            },
        ]
    )
    plan, _note = await planner.create_initial_plan(session=session, budget=budget, adapter=adapter)
    assert adapter.calls == 2
    assert plan.planned_by == "local_llm_planner"
    assert [t.task_id for t in plan.tasks] == ["task-1", "task-2"]
    assert plan.tasks[1].dependencies == ["task-1"]

    # 2. Both retries invalid -> clean deterministic fallback without crash
    bad_adapter = _FakeAdapter([ValueError("corrupt json"), {"tasks": []}])
    fallback_plan, note = await planner.create_initial_plan(
        session=session,
        budget=ExecutionBudgetTracker(default_budget_for_depth(ResearchDepth.STANDARD)),
        adapter=bad_adapter,
    )
    assert fallback_plan.planned_by == "deterministic_planner"
    assert len(fallback_plan.tasks) >= 2
    assert "deterministic" in note.lower()


def test_task_manager_dag_cycles_dependencies_duplicates_and_edits() -> None:
    t1 = ResearchTask(
        task_id="task-1",
        question="Identify compact open models under 8GB RAM",
        objective="Find specs",
        priority=5,
        status=TaskStatus.COMPLETED,
    )
    t2 = ResearchTask(
        task_id="task-2",
        question="Benchmark compact model throughput on M2",
        objective="Measure throughput",
        priority=4,
        dependencies=["task-1"],
    )
    t3 = ResearchTask(
        task_id="task-3",
        question="Analyze thermal throttling risks",
        objective="Check sustained load",
        priority=3,
        dependencies=["task-2"],
    )
    session = ResearchSession(
        research_id="ag-dag",
        original_query="Test query",
        research_objective="Test objective",
        research_plan=ResearchPlanSpec(
            plan_id="p1",
            objective="Test objective",
            normalized_objective="test objective",
            tasks=[t1, t2, t3],
        ),
    )

    # t2 becomes READY because t1 completed; t3 waits on t2
    batch = TaskManager.select_next_batch(session, max_tasks=3)
    assert [t.task_id for t in batch] == ["task-2"]
    assert t3.status == TaskStatus.PENDING

    # Cycle rejection
    c1 = ResearchTask(task_id="a", question="Question alpha", objective="Obj", dependencies=["b"])
    c2 = ResearchTask(task_id="b", question="Question beta", objective="Obj", dependencies=["a"])
    with pytest.raises(PlanValidationError, match="Cyclic"):
        TaskManager.validate_dag([c1, c2])

    # Duplicate follow-up blocker
    dup = ResearchTask(
        task_id="task-99",
        question="Identify compact open models under 8GB RAM",
        objective="Duplicate",
    )
    assert is_duplicate_task(dup, [t1, t2, t3]) is not None
    added, skipped = TaskManager.merge_followup_tasks(session, [dup])
    assert added == []
    assert len(skipped) == 1

    # Safe user edit preserves COMPLETED task-1 intact and updates pending t3
    updated = TaskManager.apply_user_plan_edit(
        session,
        AgentPlanEditRequest(
            tasks=[
                AgentPlanTaskEdit(
                    task_id="task-1",
                    question="Attempted overwrite of completed task",
                    objective="Should stay immutable",
                    priority=1,
                ),
                AgentPlanTaskEdit(
                    task_id="task-2",
                    question="Benchmark Aurora throughput on laptop hardware",
                    objective="Updated question",
                    priority=5,
                    dependencies=["task-1"],
                ),
            ]
        ),
        valid_tool_names={"web_search", "url_retrieve"},
    )
    preserved_t1 = next(t for t in updated.tasks if t.task_id == "task-1")
    assert preserved_t1.status == TaskStatus.COMPLETED
    assert preserved_t1.question == "Identify compact open models under 8GB RAM"


@pytest.mark.asyncio
async def test_tool_registry_security_privacy_approval_timeout_and_selector(tmp_path: Path) -> None:
    registry, _, _ = await _make_registry(tmp_path)

    # Forbidden tool names can never be registered or selected
    for bad in ("python", "bash", "shell", "browser"):
        assert bad in FORBIDDEN_TOOL_NAMES
        ok, _reason = registry.is_tool_available(
            bad,
            ResearchSession(
                research_id="s1", original_query="q", research_objective="obj"
            ),
        )
        assert not ok

    # Mock-only external search allowed when provider=["mock"] even with allow_external_search=False;
    # non-mock external provider is blocked cleanly!
    private_session = ResearchSession(
        research_id="s-priv",
        original_query="local model memory footprint",
        research_objective="Evaluate local memory footprint",
        providers=["mock"],
        allow_external_search=False,
    )
    sanitized_res = await registry.execute(
        tool_name="web_search",
        task_id="t1",
        arguments={
            "query": "[workspace_note] project_codename_apollo aurora model memory",
            "max_results": 3,
        },
        session=private_session,
        blocked_private_terms=["project_codename_apollo"],
    )
    assert sanitized_res.success
    assert "project_codename_apollo" not in sanitized_res.sanitized_inputs["query"].lower()
    assert "workspace_note" not in sanitized_res.sanitized_inputs["query"].lower()

    # Non-mock session with require_user_approval_for_external raises explicit gate
    gated_session = ResearchSession(
        research_id="s-gate",
        original_query="local model memory",
        research_objective="Obj",
        providers=["brave"],
        allow_external_search=True,
        require_user_approval_for_external=True,
    )
    with pytest.raises(ToolApprovalRequiredError):
        await registry.execute(
            tool_name="web_search",
            task_id="t1",
            arguments={"query": "aurora model"},
            session=gated_session,
        )

    # Extra unexpected JSON arguments are strictly forbidden (extra="forbid")
    bad_args_res = await registry.execute(
        tool_name="evidence_search",
        task_id="t1",
        arguments={"query": "aurora", "unexpected_shell": "rm -rf /"},
        session=private_session,
    )
    assert not bad_args_res.success
    assert bad_args_res.error_code == "invalid_tool_input"

    # Selector rejects LLM proposal for forbidden tool and falls back safely
    selector = ToolSelector(registry)
    llm_bad_tool = _FakeAdapter(
        [{"tools": [{"tool_name": "bash", "arguments": {"cmd": "ls"}, "rationale": "x"}]}]
    )
    proposals, note = await selector.select_tools_for_task(
        ResearchTask(task_id="t1", question="aurora memory", objective="Obj"),
        private_session,
        adapter=llm_bad_tool,
    )
    assert all(p.tool_name not in FORBIDDEN_TOOL_NAMES for p in proposals)
    assert "Rejected prohibited" in note

    # Timeout enforcement check via slow mock handler
    async def _slow_handler(_args, _sess):
        await asyncio.sleep(0.5)
        return {}

    from web_intelligence.app.agent.models import ToolDescriptor, ToolPermission
    from web_intelligence.app.agent.tool_registry import _StrictToolInput

    class _SlowIn(_StrictToolInput):
        x: str = "a"

    registry.register(
        descriptor=ToolDescriptor(
            name="slow_test_tool",
            description="Test timeout enforcement",
            input_schema=_SlowIn.model_json_schema(),
            output_schema={},
            permissions=[ToolPermission.READ_LOCAL],
            timeout_seconds=0.2,
        ),
        input_model=_SlowIn,
        handler=_slow_handler,
    )
    to_res = await registry.execute(
        tool_name="slow_test_tool",
        task_id="t1",
        arguments={"x": "a"},
        session=private_session,
        timeout_override=0.05,
    )
    assert not to_res.success
    assert to_res.error_code == "tool_timeout"


def test_evidence_evaluator_and_seven_gap_categories() -> None:
    evaluator = EvidenceQualityEvaluator()
    gap_analyzer = GapAnalyzer(stale_days=180)

    # Task 1: single secondary blog with high relevance must NOT become CONFIRMED/CORROBORATED -> WEAK!
    ev_weak = _make_evidence(
        ev_id="ev-weak",
        src_id="src-blog",
        domain="random-blog.example.com",
        text="A blog review suggests compact local models use about 4 GB RAM.",
        classification=SourceClassification.SECONDARY,
        corroboration_status=CorroborationStatus.SINGLE_SOURCE,
        independent_sources=1,
        relevance=0.96,
        published_at=datetime.now(timezone.utc) - timedelta(days=500),
    )
    # Task 2: snippet-only, 0 retained passages -> UNVERIFIED!
    # Task 3: conflicting pair -> CONFLICTING!
    ev_conf_a = _make_evidence(
        ev_id="ev-ca",
        src_id="src-a",
        domain="docs.openmodels.org",
        text="Benchmark evaluation measured Aurora-SLM-4B at 48 tok/s on M2.",
        classification=SourceClassification.PRIMARY,
        corroboration_status=CorroborationStatus.CONTESTED,
        conflict_ids=["conf-1"],
        published_at=datetime.now(timezone.utc) - timedelta(days=500),
    )
    ev_conf_b = _make_evidence(
        ev_id="ev-cb",
        src_id="src-b",
        domain="bench.example.org",
        text="Hardware experiment measured Aurora-SLM-4B at 34 tok/s due to thermal throttling.",
        classification=SourceClassification.SECONDARY,
        corroboration_status=CorroborationStatus.CONTESTED,
        conflict_ids=["conf-1"],
        published_at=datetime.now(timezone.utc) - timedelta(days=500),
    )
    claim_a = ExtractedClaim(
        claim_id="cl-a",
        source_id="src-a",
        evidence_id="ev-ca",
        source_url="https://docs.openmodels.org/a",
        source_domain="docs.openmodels.org",
        entity="aurora-slm-4b",
        attribute="throughput",
        statement_text="48 tok/s",
        value_type="quantity",
        normalized_value="48",
        unit="tokens/sec",
    )
    claim_b = ExtractedClaim(
        claim_id="cl-b",
        source_id="src-b",
        evidence_id="ev-cb",
        source_url="https://bench.example.org/b",
        source_domain="bench.example.org",
        entity="aurora-slm-4b",
        attribute="throughput",
        statement_text="34 tok/s",
        value_type="quantity",
        normalized_value="34",
        unit="tokens/sec",
    )
    claim_weak = ExtractedClaim(
        claim_id="cl-w",
        source_id="src-blog",
        evidence_id="ev-weak",
        source_url="https://random-blog.example.com/article",
        source_domain="random-blog.example.com",
        entity="compact",
        attribute="memory_ram",
        statement_text="4 GB RAM",
        value_type="quantity",
        normalized_value="4",
        unit="gb",
    )
    conflict = ClaimConflict(
        conflict_id="conf-1",
        topic="aurora-slm-4b throughput",
        entity="aurora-slm-4b",
        attribute="throughput",
        conflict_type="numeric_mismatch",
        claim_a=claim_a,
        claim_b=claim_b,
        summary="Throughput mismatch 48 vs 34 tok/s",
    )

    session = ResearchSession(
        research_id="ag-eval",
        original_query="latest Aurora-SLM-4B benchmarks and privacy tradeoffs",
        research_objective="Evaluate latest compact models and privacy",
        current_iteration=1,
        research_plan=ResearchPlanSpec(
            plan_id="p1",
            objective="Evaluate",
            normalized_objective="evaluate",
            tasks=[
                ResearchTask(
                    task_id="task-1",
                    question="compact models memory review",
                    objective="obj1",
                    status=TaskStatus.COMPLETED,
                    evidence_ids=["ev-weak"],
                ),
                ResearchTask(
                    task_id="task-2",
                    question="privacy sanitization rules offline boundaries",
                    objective="obj2",
                    status=TaskStatus.COMPLETED,
                    evidence_ids=[],
                ),
                ResearchTask(
                    task_id="task-3",
                    question="aurora-slm-4b throughput benchmark evaluation",
                    objective="obj3",
                    status=TaskStatus.COMPLETED,
                    evidence_ids=["ev-ca", "ev-cb"],
                ),
            ],
        ),
        collected_sources=[
            SourceRecord(
                source_id="src-blog",
                original_url="https://random-blog.example.com/article",
                final_url="https://random-blog.example.com/article",
                canonical_url="https://random-blog.example.com/article",
                domain="random-blog.example.com",
                main_content="body",
                retrieved_at=datetime.now(timezone.utc),
                published_at=datetime.now(timezone.utc) - timedelta(days=500),
            ),
            SourceRecord(
                source_id="src-mirror",
                original_url="https://mirror.example.net/article",
                final_url="https://mirror.example.net/article",
                canonical_url="https://mirror.example.net/article",
                domain="mirror.example.net",
                main_content="body",
                retrieved_at=datetime.now(timezone.utc),
                is_duplicate=True,
                duplicate_of_source_id="src-blog",
                duplicate_reason="near_duplicate",
            ),
            SourceRecord(
                source_id="src-a",
                original_url="https://docs.openmodels.org/a",
                final_url="https://docs.openmodels.org/a",
                canonical_url="https://docs.openmodels.org/a",
                domain="docs.openmodels.org",
                main_content="body",
                retrieved_at=datetime.now(timezone.utc),
                published_at=datetime.now(timezone.utc) - timedelta(days=500),
            ),
            SourceRecord(
                source_id="src-b",
                original_url="https://bench.example.org/b",
                final_url="https://bench.example.org/b",
                canonical_url="https://bench.example.org/b",
                domain="bench.example.org",
                main_content="body",
                retrieved_at=datetime.now(timezone.utc),
                published_at=datetime.now(timezone.utc) - timedelta(days=500),
            ),
        ],
        evidence_records=[ev_weak, ev_conf_a, ev_conf_b],
        claims=[claim_weak, claim_a, claim_b],
        conflicts=[conflict],
        source_decisions=[
            SourceDecisionRecord(source_id="src-blog", url="u1", domain="d1", retained=True, reason="ok"),
            SourceDecisionRecord(source_id="src-mirror", url="u2", domain="d2", retained=False, reason="dup", is_duplicate=True),
            SourceDecisionRecord(source_id="src-a", url="u3", domain="d3", retained=True, reason="ok"),
            SourceDecisionRecord(source_id="src-b", url="u4", domain="d4", retained=True, reason="ok"),
        ],
    )

    report = evaluator.evaluate_session(session, snippet_only_by_task={"task-2": 3})
    by_tid = {a.task_id: a for a in report.task_assessments}
    assert by_tid["task-1"].support_status == EvidenceSupportStatus.WEAK
    assert by_tid["task-2"].support_status == EvidenceSupportStatus.UNVERIFIED
    assert by_tid["task-2"].snippet_only_count == 3
    assert by_tid["task-3"].support_status == EvidenceSupportStatus.CONFLICTING
    assert "hardware" in by_tid["task-3"].methodological_cues_found

    gaps_rep = gap_analyzer.analyze(session, report, temporal_focus=True)
    found_categories = {g.category for g in gaps_rep.gaps}
    assert GapCategory.UNANSWERED_QUESTION in found_categories
    assert GapCategory.MISSING_PRIMARY_SOURCE in found_categories
    assert GapCategory.CONFLICTING_RESULTS in found_categories
    assert GapCategory.SINGLE_SOURCE_CLAIM in found_categories
    assert GapCategory.OUTDATED_INFORMATION in found_categories


def test_stopping_policy_budget_state_machine_and_trace_redaction() -> None:
    policy = StoppingPolicy()
    base_session = ResearchSession(
        research_id="ag-stop",
        original_query="Test",
        research_objective="Obj",
        research_plan=ResearchPlanSpec(
            plan_id="p1",
            objective="Obj",
            normalized_objective="obj",
            tasks=[
                ResearchTask(task_id="t1", question="Question one?", objective="Obj1", status=TaskStatus.COMPLETED)
            ],
        ),
    )

    # Diminishing returns stop
    dec_dim = policy.evaluate(
        session=base_session,
        budget=ExecutionBudgetTracker(ExecutionBudgetConfig(max_iterations=5)),
        evaluation=EvidenceEvaluationReport(
            iteration=2,
            objective_coverage_ratio=0.5,
            total_accepted_evidence=2,
        ),
        gaps=GapAnalysisReport(iteration=2, gaps=[]),
        iteration_evidence_deltas=[2, 0],
        iteration_coverage_deltas=[0.5, 0.0],
    )
    assert dec_dim.should_stop and dec_dim.reason == StoppingReason.DIMINISHING_RETURNS

    # Max search budget stop
    dec_search = policy.evaluate(
        session=base_session,
        budget=ExecutionBudgetTracker(
            ExecutionBudgetConfig(max_search_calls=2),
            ExecutionBudgetUsage(search_calls=2),
        ),
        evaluation=None,
        gaps=None,
        iteration_evidence_deltas=[1],
        iteration_coverage_deltas=[0.2],
    )
    assert dec_search.should_stop and dec_search.reason == StoppingReason.SEARCH_BUDGET_REACHED

    # Objectives covered stop
    dec_ok = policy.evaluate(
        session=base_session,
        budget=ExecutionBudgetTracker(ExecutionBudgetConfig(max_iterations=4)),
        evaluation=EvidenceEvaluationReport(
            iteration=2,
            objective_coverage_ratio=0.90,
            unique_primary_domains=2,
            total_accepted_evidence=4,
        ),
        gaps=GapAnalysisReport(iteration=2, gaps=[], high_priority_count=0),
        iteration_evidence_deltas=[2, 2],
        iteration_coverage_deltas=[0.4, 0.5],
    )
    assert dec_ok.should_stop and dec_ok.reason == StoppingReason.OBJECTIVES_COVERED

    # State machine transition guard
    with pytest.raises(InvalidStateTransitionError):
        ResearchSessionStateManager.transition(base_session, AgentSessionStatus.COMPLETED)

    # Trace redaction hides api_key/bearer secrets
    recorder = ResearchTraceRecorder("ag-stop")
    entry = recorder.record(
        iteration=1,
        phase="test",
        event_type=TraceEventType.TOOL_EXECUTED,
        reason="Checked endpoint",
        sanitized_inputs={"api_key": "super-secret-value", "query": "Bearer sk-12345"},
    )
    assert entry.sanitized_inputs["api_key"] == "[REDACTED]"
    assert "sk-12345" not in str(entry.sanitized_inputs)
