"""Central Stage 7 AgentResearchOrchestrator coordinating bounded iterative research and Stage 6 synthesis."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from web_intelligence.app.agent.evidence_evaluator import EvidenceQualityEvaluator
from web_intelligence.app.agent.execution_budget import (
    ExecutionBudgetTracker,
    default_budget_for_depth,
)
from web_intelligence.app.agent.execution_engine import AgentExecutionEngine
from web_intelligence.app.agent.gap_analyzer import GapAnalyzer
from web_intelligence.app.agent.models import (
    AgentApprovalDecisionRequest,
    AgentPlanEditRequest,
    AgentResearchRequest,
    AgentSessionStatus,
    ResearchDepth,
    ResearchPlanSpec,
    ResearchSession,
    ResearchTask,
    ResearchTraceEntry,
    StoppingDecision,
    StoppingReason,
    TaskStatus,
    ToolDescriptor,
    TraceEventType,
    utc_now,
)
from web_intelligence.app.agent.observation_processor import ObservationProcessor
from web_intelligence.app.agent.research_memory import ResearchWorkingMemory
from web_intelligence.app.agent.research_planner import AgentResearchPlanner
from web_intelligence.app.agent.research_state import ResearchSessionStateManager
from web_intelligence.app.agent.research_trace import ResearchTraceRecorder
from web_intelligence.app.agent.stopping_policy import StoppingPolicy
from web_intelligence.app.agent.task_manager import TaskManager
from web_intelligence.app.agent.tool_registry import ToolRegistry
from web_intelligence.app.agent.tool_selector import ToolSelector
from web_intelligence.app.context.context_builder import StructuredContextBuilder
from web_intelligence.app.context.engine import ContextManagementEngine
from web_intelligence.app.context.models import (
    AllocationProfile,
    ContextBudgetConfig,
    ContextPreviewRequest,
    ExpectedOutputLength,
)
from web_intelligence.app.context.token_estimator import resolve_token_estimator
from web_intelligence.app.core.logging import get_logger
from web_intelligence.app.generation.citations import (
    CitationValidator,
    merge_citations_preserving_order,
)
from web_intelligence.app.generation.executor import GenerationError, InferenceExecutor
from web_intelligence.app.generation.long_form import LongFormReportGenerator
from web_intelligence.app.generation.prompt_compiler import PromptCompiler
from web_intelligence.app.llm.base import UniversalLocalModelAdapter
from web_intelligence.app.llm.manager import LLMConnectionManager
from web_intelligence.app.llm.security import redact_llm_error
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.providers.router import SearchProviderRouter
from web_intelligence.app.query.decomposition import decompose_query
from web_intelligence.app.research.context_builder import ContextBuilder as Stage4PackageBuilder
from web_intelligence.app.research.research_state import ResearchState
from web_intelligence.app.retrieval.search_engine import WebRetrievalEngine
from web_intelligence.app.schemas.evidence import CoverageAnalysis
from web_intelligence.app.schemas.generation import (
    CitationEntry,
    GenerationFormat,
    GenerationLength,
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    LongFormReportConfig,
    PromptTemplateType,
)
from web_intelligence.app.schemas.research import (
    QueryPlan,
    ResearchPackage,
    ResearchStatus,
)
from web_intelligence.app.storage.repositories import ResearchRepository

logger = get_logger("agent.orchestrator")


class AgentSessionNotFoundError(LookupError):
    """Raised when a requested agent research_id is missing."""


@dataclass
class _SessionControlHandle:
    """In-memory runtime control signals for an active background research session."""

    task: asyncio.Task[ResearchSession] | None = None
    pause_event: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    approved_actions: set[tuple[str, str]] = field(default_factory=set)
    iteration_evidence_deltas: list[int] = field(default_factory=list)
    iteration_coverage_deltas: list[float] = field(default_factory=list)


class AgentResearchOrchestrator:
    """Cohesive coordinator driving iterative PLAN -> EXECUTE -> OBSERVE -> EVALUATE -> GAPS -> SYNTHESIS."""

    def __init__(
        self,
        *,
        repository: ResearchRepository,
        gateway: PrivacyGateway,
        router: SearchProviderRouter,
        retrieval: WebRetrievalEngine | None = None,
        context_engine: ContextManagementEngine | None = None,
        llm_manager: LLMConnectionManager | None = None,
        inference_executor: InferenceExecutor | None = None,
        long_form_generator: LongFormReportGenerator | None = None,
        prompt_compiler: PromptCompiler | None = None,
        citation_validator: CitationValidator | None = None,
        tool_registry: ToolRegistry | None = None,
        planner: AgentResearchPlanner | None = None,
        execution_engine: AgentExecutionEngine | None = None,
        evidence_evaluator: EvidenceQualityEvaluator | None = None,
        gap_analyzer: GapAnalyzer | None = None,
        stopping_policy: StoppingPolicy | None = None,
    ) -> None:
        self._repository = repository
        self._gateway = gateway
        self._router = router
        self._retrieval = retrieval or WebRetrievalEngine(
            settings=gateway._settings, gateway=gateway, router=router
        )
        self._context_engine = context_engine
        self._llm_manager = llm_manager
        self._executor = inference_executor
        self._long_form = long_form_generator
        self._prompt_compiler = prompt_compiler or PromptCompiler()
        self._citation_validator = citation_validator or CitationValidator()
        self._memory = ResearchWorkingMemory()
        self._stage4_builder = Stage4PackageBuilder()

        self._registry = tool_registry or ToolRegistry(
            router=router,
            retrieval_engine=self._retrieval,
            repository=repository,
            context_engine=context_engine,
            llm_manager=llm_manager,
            memory=self._memory,
        )
        self._selector = ToolSelector(self._registry, memory=self._memory)
        self._observer = ObservationProcessor()
        self._exec_engine = execution_engine or AgentExecutionEngine(
            registry=self._registry,
            selector=self._selector,
            observation_processor=self._observer,
            memory=self._memory,
        )
        self._planner = planner or AgentResearchPlanner(self._registry)
        self._evaluator = evidence_evaluator or EvidenceQualityEvaluator()
        self._gap_analyzer = gap_analyzer or GapAnalyzer()
        self._stopping = stopping_policy or StoppingPolicy()

        self._handles: dict[str, _SessionControlHandle] = {}
        self._request_opts: dict[str, dict[str, Any]] = {}

    def list_tools(self) -> list[ToolDescriptor]:
        """Return registered ToolRegistry descriptors for API inspection."""
        return self._registry.list_tools()

    async def start_research(
        self,
        request: AgentResearchRequest,
        *,
        background: bool = True,
    ) -> ResearchSession:
        """Create and persist a new agent research session and start execution."""
        rid = f"ag-{uuid.uuid4().hex[:12]}"
        budget_cfg = request.execution_budget or default_budget_for_depth(
            request.research_depth,
            require_approval_for_external=request.require_user_approval_for_external,
        )
        if request.require_user_approval_for_external:
            budget_cfg = budget_cfg.model_copy(
                update={"require_approval_for_external_search": True}
            )

        objective = (
            request.research_objective.strip()
            if request.research_objective and request.research_objective.strip()
            else f"Conduct evidence-grounded investigation of: {request.query}"
        )

        # Best-effort connection discovery without blocking or crashing offline runs
        conn_id = request.connection_id
        model_id = request.model_id
        if request.use_local_llm and (conn_id is None or model_id is None):
            conn_id, model_id = await self._probe_default_model(conn_id, model_id)

        session = ResearchSession(
            research_id=rid,
            original_query=request.query,
            research_objective=objective,
            research_depth=request.research_depth,
            status=AgentSessionStatus.PENDING,
            current_phase="pending",
            execution_budget=budget_cfg,
            providers=request.providers,
            active_connection_id=conn_id,
            active_model_id=model_id,
            use_local_llm=request.use_local_llm,
            allow_external_search=request.allow_external_search,
            require_user_approval_for_external=request.require_user_approval_for_external,
        )
        await self._repository.save_agent_session(session)

        handle = _SessionControlHandle()
        self._handles[rid] = handle
        self._request_opts[rid] = {
            "max_results_per_query": request.max_results_per_query,
            "chunk_target_tokens": request.chunk_target_tokens,
            "temporal_focus": request.temporal_focus,
        }

        if background:
            task = asyncio.create_task(self.run_session(rid), name=f"agent-research-{rid}")
            handle.task = task
            return session

        return await self.run_session(rid)

    async def run_session(self, research_id: str) -> ResearchSession:
        """Execute or resume the bounded multi-iteration autonomous research loop."""
        session = await self._repository.get_agent_session(research_id)
        if session is None:
            raise AgentSessionNotFoundError(f"Agent research session '{research_id}' not found.")

        handle = self._handles.setdefault(research_id, _SessionControlHandle())
        existing_trace = await self._repository.list_agent_trace_entries(research_id)
        trace = ResearchTraceRecorder(
            research_id,
            initial_sequence=existing_trace[-1].sequence if existing_trace else 0,
        )
        budget = ExecutionBudgetTracker(session.execution_budget, session.budget_usage)
        opts = self._request_opts.get(
            research_id,
            {"max_results_per_query": 5, "chunk_target_tokens": 180, "temporal_focus": False},
        )
        blocked_terms = await self._load_private_note_phrases()
        loop_started = time.monotonic()

        try:
            adapter = await self._resolve_adapter(session)

            if handle.cancel_event.is_set() or session.status == AgentSessionStatus.CANCELLED:
                return await self._finalize_with_stop(
                    session=session,
                    trace=trace,
                    budget=budget,
                    reason=StoppingReason.USER_CANCELLED,
                    explanation="Session cancelled by user request.",
                    adapter=adapter,
                )

            if session.status == AgentSessionStatus.PENDING and not existing_trace:
                trace.record(
                    iteration=0,
                    phase="initialization",
                    event_type=TraceEventType.SESSION_STARTED,
                    reason=(
                        f"Started autonomous research ({session.research_depth.value}, "
                        f"use_local_llm={session.use_local_llm}, external_search={session.allow_external_search})."
                    ),
                    details={"budget": session.execution_budget.model_dump(mode="json")},
                )
            else:
                trace.record(
                    iteration=session.current_iteration,
                    phase="resume",
                    event_type=TraceEventType.SESSION_RESUMED,
                    reason=(
                        f"Resuming session from persisted state '{session.status.value}' at "
                        f"iteration {session.current_iteration} with {len(session.completed_tasks)} completed task(s)."
                    ),
                )

            # 1. Create initial research plan if absent
            if session.research_plan is None:
                ResearchSessionStateManager.transition(
                    session, AgentSessionStatus.PLANNING, phase="planning"
                )
                plan, plan_note = await self._planner.create_initial_plan(
                    session=session, budget=budget, adapter=adapter
                )
                session.research_plan = plan
                ResearchSessionStateManager.sync_task_buckets(session)
                trace.record(
                    iteration=0,
                    phase="planning",
                    event_type=TraceEventType.PLAN_CREATED,
                    reason=plan_note,
                    outcome=f"{len(plan.tasks)} task(s) created",
                    details={"plan_id": plan.plan_id, "planned_by": plan.planned_by},
                )
                for task in plan.tasks:
                    trace.record(
                        iteration=0,
                        phase="planning",
                        event_type=TraceEventType.TASK_CREATED,
                        task_id=task.task_id,
                        reason=task.objective,
                        details={"priority": task.priority, "dependencies": task.dependencies},
                    )
                await self._checkpoint(session, trace, budget)

            # 2. If resuming after user approval, run newly approved waiting actions first
            ran_approved_actions = False
            approved_prev_ev_count = len(session.evidence_records)
            approved_prev_cov_ratio = (
                session.evidence_evaluation.objective_coverage_ratio
                if session.evidence_evaluation
                else 0.0
            )
            approved_snippet_map: dict[str, int] = {}
            if session.pending_approvals and session.research_plan is not None:
                approved_items = [a for a in session.pending_approvals if a.status == "approved"]
                for apr in approved_items:
                    target_task = next(
                        (t for t in session.research_plan.tasks if t.task_id == apr.task_id),
                        None,
                    )
                    if target_task is None:
                        continue
                    handle.approved_actions.add((apr.task_id, apr.tool_name))
                    if session.status != AgentSessionStatus.EXECUTING:
                        ResearchSessionStateManager.transition(
                            session, AgentSessionStatus.EXECUTING, phase="executing_approved"
                        )
                    outcome = await self._exec_engine.execute_approved_action(
                        approval=apr,
                        task=target_task,
                        session=session,
                        budget=budget,
                        trace=trace,
                        max_results_per_query=opts["max_results_per_query"],
                        chunk_target_tokens=opts["chunk_target_tokens"],
                        blocked_private_terms=blocked_terms,
                    )
                    ran_approved_actions = True
                    if outcome.observation is not None:
                        approved_snippet_map[target_task.task_id] = outcome.observation.snippet_only_hits
                    apr.status = "approved"
                    session.pending_approvals = [
                        p
                        for p in session.pending_approvals
                        if p.approval_id != apr.approval_id
                    ]
                    if outcome.approval_required is not None:
                        ResearchSessionStateManager.transition(
                            session, AgentSessionStatus.WAITING_APPROVAL, phase="waiting_approval"
                        )
                        ResearchSessionStateManager.sync_task_buckets(session)
                        await self._checkpoint(session, trace, budget)
                        return session
                ResearchSessionStateManager.sync_task_buckets(session)
                await self._checkpoint(session, trace, budget)

            # Check if paused right away or waiting unresolved approval
            if any(p.status == "pending" for p in session.pending_approvals):
                ResearchSessionStateManager.transition(
                    session, AgentSessionStatus.WAITING_APPROVAL, phase="waiting_approval"
                )
                await self._checkpoint(session, trace, budget)
                return session

            if handle.cancel_event.is_set() or session.status == AgentSessionStatus.CANCELLED:
                return await self._finalize_with_stop(
                    session=session,
                    trace=trace,
                    budget=budget,
                    reason=StoppingReason.USER_CANCELLED,
                    explanation="Cancelled prior to iteration execution.",
                    adapter=adapter,
                )

            if handle.pause_event.is_set():
                ResearchSessionStateManager.transition(
                    session, AgentSessionStatus.PAUSED, phase="paused"
                )
                await self._checkpoint(session, trace, budget)
                return session

            # Finalize interrupted iteration if newly approved tasks just finished executing
            skip_main_loop = False
            if ran_approved_actions and session.current_iteration > 0:
                budget.record_elapsed_seconds(time.monotonic() - loop_started)
                loop_started = time.monotonic()
                skip_main_loop = await self._evaluate_gaps_and_stopping(
                    iter_num=session.current_iteration,
                    session=session,
                    trace=trace,
                    budget=budget,
                    handle=handle,
                    adapter=adapter,
                    opts=opts,
                    prev_evidence_count=approved_prev_ev_count,
                    prev_coverage_ratio=approved_prev_cov_ratio,
                    snippet_only_map=approved_snippet_map,
                )

            # 3. Main bounded research loop
            while not skip_main_loop:
                budget.record_elapsed_seconds(time.monotonic() - loop_started)
                loop_started = time.monotonic()

                next_iter = session.current_iteration + 1
                can_iter, iter_msg = budget.can_start_iteration(next_iter)
                if not can_iter:
                    decision = self._stopping.evaluate(
                        session=session,
                        budget=budget,
                        evaluation=session.evidence_evaluation,
                        gaps=None,
                        iteration_evidence_deltas=handle.iteration_evidence_deltas,
                        iteration_coverage_deltas=handle.iteration_coverage_deltas,
                    )
                    trace.record(
                        iteration=session.current_iteration,
                        phase="stopping",
                        event_type=TraceEventType.STOPPING_DECISION,
                        reason=iter_msg or decision.explanation,
                        outcome=decision.reason.value if decision.reason else "stop",
                    )
                    session.stopping_decision = decision
                    break

                session.current_iteration = next_iter
                trace.record(
                    iteration=next_iter,
                    phase="iteration_start",
                    event_type=TraceEventType.ITERATION_STARTED,
                    reason=f"Starting research iteration {next_iter}/{budget.config.max_iterations}.",
                )

                # Select next runnable batch of tasks (respects DAG dependencies and priority)
                if session.status != AgentSessionStatus.EXECUTING:
                    ResearchSessionStateManager.transition(
                        session, AgentSessionStatus.EXECUTING, phase="executing"
                    )

                prev_evidence_count = len(session.evidence_records)
                prev_coverage_ratio = (
                    session.evidence_evaluation.objective_coverage_ratio
                    if session.evidence_evaluation
                    else 0.0
                )

                batch = TaskManager.select_next_batch(
                    session, max_tasks=budget.config.max_concurrent_tasks
                )
                snippet_only_map: dict[str, int] = {}

                if batch:
                    batch_out = await self._exec_engine.execute_batch(
                        tasks=batch,
                        session=session,
                        budget=budget,
                        trace=trace,
                        adapter=adapter,
                        pause_event=handle.pause_event,
                        cancel_event=handle.cancel_event,
                        approved_proposals=handle.approved_actions,
                        max_results_per_query=opts["max_results_per_query"],
                        chunk_target_tokens=opts["chunk_target_tokens"],
                        blocked_private_terms=blocked_terms,
                    )
                    snippet_only_map.update(batch_out.snippet_only_by_task)
                    session.current_task_id = None
                    ResearchSessionStateManager.sync_task_buckets(session)

                    if batch_out.approval_requests:
                        ResearchSessionStateManager.transition(
                            session,
                            AgentSessionStatus.WAITING_APPROVAL,
                            phase="waiting_approval",
                        )
                        eval_snap = self._evaluator.evaluate_session(
                            session, snippet_only_by_task=snippet_only_map
                        )
                        session.evidence_evaluation = eval_snap
                        session.stopping_decision = self._stopping.evaluate(
                            session=session,
                            budget=budget,
                            evaluation=eval_snap,
                            gaps=None,
                            iteration_evidence_deltas=handle.iteration_evidence_deltas,
                            iteration_coverage_deltas=handle.iteration_coverage_deltas,
                            waiting_approval=True,
                        )
                        await self._checkpoint(session, trace, budget)
                        return session

                    if batch_out.cancelled or handle.cancel_event.is_set():
                        return await self._finalize_with_stop(
                            session=session,
                            trace=trace,
                            budget=budget,
                            reason=StoppingReason.USER_CANCELLED,
                            explanation="Execution interrupted by user cancellation.",
                            adapter=adapter,
                        )

                    if batch_out.paused or handle.pause_event.is_set():
                        ResearchSessionStateManager.transition(
                            session, AgentSessionStatus.PAUSED, phase="paused"
                        )
                        session.stopping_decision = self._stopping.evaluate(
                            session=session,
                            budget=budget,
                            evaluation=session.evidence_evaluation,
                            gaps=None,
                            iteration_evidence_deltas=handle.iteration_evidence_deltas,
                            iteration_coverage_deltas=handle.iteration_coverage_deltas,
                            pause_requested=True,
                        )
                        trace.record(
                            iteration=next_iter,
                            phase="pause",
                            event_type=TraceEventType.SESSION_PAUSED,
                            reason="Session paused cleanly at task boundary checkpoint.",
                        )
                        await self._checkpoint(session, trace, budget)
                        return session

                should_stop = await self._evaluate_gaps_and_stopping(
                    iter_num=next_iter,
                    session=session,
                    trace=trace,
                    budget=budget,
                    handle=handle,
                    adapter=adapter,
                    opts=opts,
                    prev_evidence_count=prev_evidence_count,
                    prev_coverage_ratio=prev_coverage_ratio,
                    snippet_only_map=snippet_only_map,
                )
                if should_stop:
                    break

            # 4. Final synthesis handoff to Stage 6
            return await self._synthesize_and_complete(
                session=session,
                trace=trace,
                budget=budget,
                adapter=adapter,
                cancelled=handle.cancel_event.is_set(),
            )

        except Exception as exc:
            safe_err = redact_llm_error(exc)
            logger.exception(
                "Agent session failed",
                extra={"extra_data": {"research_id": research_id}},
            )
            session.error_message = safe_err
            trace.record(
                iteration=session.current_iteration,
                phase="error",
                event_type=TraceEventType.ERROR,
                reason=safe_err,
                outcome="failed",
            )
            session.stopping_decision = StoppingDecision(
                should_stop=True,
                reason=StoppingReason.EXECUTION_ERROR,
                explanation=safe_err,
                objective_coverage_ratio=(
                    session.evidence_evaluation.objective_coverage_ratio
                    if session.evidence_evaluation
                    else 0.0
                ),
            )
            ResearchSessionStateManager.transition(
                session, AgentSessionStatus.FAILED, phase="failed"
            )
            await self._checkpoint(session, trace, budget)
            return session

    async def pause_session(self, research_id: str) -> ResearchSession:
        """Request clean checkpoint pause for an active session (or confirm already paused)."""
        session = await self._require_session(research_id)
        if session.status in {
            AgentSessionStatus.COMPLETED,
            AgentSessionStatus.CANCELLED,
            AgentSessionStatus.FAILED,
        }:
            return session
        handle = self._handles.setdefault(research_id, _SessionControlHandle())
        handle.pause_event.set()
        if handle.task is None or handle.task.done():
            ResearchSessionStateManager.transition(
                session, AgentSessionStatus.PAUSED, phase="paused"
            )
            await self._repository.save_agent_session(session)
        return session

    async def resume_session(
        self,
        research_id: str,
        *,
        background: bool = True,
    ) -> ResearchSession:
        """Resume a paused or interrupted session without repeating completed tasks."""
        session = await self._require_session(research_id)
        if session.status == AgentSessionStatus.COMPLETED:
            return session
        handle = self._handles.setdefault(research_id, _SessionControlHandle())
        handle.pause_event.clear()
        handle.cancel_event.clear()
        if handle.task is not None and not handle.task.done():
            return session
        if background:
            handle.task = asyncio.create_task(
                self.run_session(research_id), name=f"agent-resume-{research_id}"
            )
            return session
        return await self.run_session(research_id)

    async def cancel_session(self, research_id: str) -> ResearchSession:
        """Cancel an active or paused session cleanly and persist terminal state."""
        session = await self._require_session(research_id)
        if session.status in {
            AgentSessionStatus.COMPLETED,
            AgentSessionStatus.CANCELLED,
            AgentSessionStatus.FAILED,
        }:
            return session
        handle = self._handles.setdefault(research_id, _SessionControlHandle())
        handle.cancel_event.set()
        if (
            handle.task is None
            or handle.task.done()
            or session.status
            in {
                AgentSessionStatus.PENDING,
                AgentSessionStatus.PAUSED,
                AgentSessionStatus.WAITING_APPROVAL,
            }
        ):
            ResearchSessionStateManager.transition(
                session, AgentSessionStatus.CANCELLED, phase="cancelled"
            )
            session.stopping_decision = StoppingDecision(
                should_stop=True,
                reason=StoppingReason.USER_CANCELLED,
                explanation="Session cancelled by user request.",
            )
            await self._repository.save_agent_session(session)
        return session

    async def handle_approval(
        self,
        research_id: str,
        decision: AgentApprovalDecisionRequest,
        *,
        background: bool = True,
    ) -> ResearchSession:
        """Approve or reject pending external tool actions and optionally resume execution."""
        session = await self._require_session(research_id)
        if not session.pending_approvals:
            return session

        trace_existing = await self._repository.list_agent_trace_entries(research_id)
        trace = ResearchTraceRecorder(
            research_id,
            initial_sequence=trace_existing[-1].sequence if trace_existing else 0,
        )

        matched = False
        for apr in session.pending_approvals:
            if decision.approval_id and apr.approval_id != decision.approval_id:
                continue
            if apr.status != "pending":
                continue
            matched = True
            apr.status = "approved" if decision.approved else "rejected"
            apr.decision_note = decision.decision_note
            apr.resolved_at = utc_now()
            trace.record(
                iteration=session.current_iteration,
                phase="approval_gate",
                event_type=TraceEventType.APPROVAL_RESOLVED,
                task_id=apr.task_id,
                tool_name=apr.tool_name,
                reason=(
                    f"User {'approved' if decision.approved else 'rejected'} tool '{apr.tool_name}'"
                    + (f": {decision.decision_note}" if decision.decision_note else ".")
                ),
                outcome=apr.status,
                details={"approval_id": apr.approval_id},
            )
            if not decision.approved and session.research_plan:
                for t in session.research_plan.tasks:
                    if t.task_id == apr.task_id and t.status == TaskStatus.WAITING_APPROVAL:
                        t.status = TaskStatus.SKIPPED
                        t.result_summary = "Skipped because user rejected external action."
                        t.updated_at = utc_now()

        if matched and not decision.approved:
            session.pending_approvals = [
                p for p in session.pending_approvals if p.status == "pending"
            ]
            ResearchSessionStateManager.sync_task_buckets(session)

        await self._repository.append_agent_trace_entries(trace.entries)
        ResearchSessionStateManager.recompute_metrics(
            session, trace_entries_count=trace.sequence
        )
        await self._repository.save_agent_session(session)

        if decision.resume_execution:
            return await self.resume_session(research_id, background=background)
        return session

    async def update_plan(
        self,
        research_id: str,
        edit: AgentPlanEditRequest,
    ) -> ResearchPlanSpec:
        """Validate and apply user modifications to pending plan tasks."""
        session = await self._require_session(research_id)
        valid_tools = {d.name for d in self._registry.list_tools()}
        updated_plan = TaskManager.apply_user_plan_edit(
            session, edit, valid_tool_names=valid_tools
        )
        existing_trace = await self._repository.list_agent_trace_entries(research_id)
        trace = ResearchTraceRecorder(
            research_id,
            initial_sequence=existing_trace[-1].sequence if existing_trace else 0,
        )
        trace.record(
            iteration=session.current_iteration,
            phase="plan_edit",
            event_type=TraceEventType.PLAN_UPDATED,
            reason=f"User edited research plan to version {updated_plan.version}.",
            outcome=f"{len(updated_plan.tasks)} task(s) total",
        )
        await self._repository.append_agent_trace_entries(trace.entries)
        ResearchSessionStateManager.recompute_metrics(
            session, trace_entries_count=trace.sequence
        )
        await self._repository.save_agent_session(session)
        return updated_plan

    async def shutdown(self) -> None:
        """Gracefully cancel and await background agent worker tasks on server shutdown."""
        pending: list[asyncio.Task[ResearchSession]] = []
        for handle in self._handles.values():
            handle.cancel_event.set()
            if handle.task is not None and not handle.task.done():
                handle.task.cancel()
                pending.append(handle.task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _finalize_with_stop(
        self,
        *,
        session: ResearchSession,
        trace: ResearchTraceRecorder,
        budget: ExecutionBudgetTracker,
        reason: StoppingReason,
        explanation: str,
        adapter: UniversalLocalModelAdapter | None,
    ) -> ResearchSession:
        session.stopping_decision = StoppingDecision(
            should_stop=True,
            reason=reason,
            explanation=explanation,
            objective_coverage_ratio=(
                session.evidence_evaluation.objective_coverage_ratio
                if session.evidence_evaluation
                else 0.0
            ),
        )
        return await self._synthesize_and_complete(
            session=session,
            trace=trace,
            budget=budget,
            adapter=adapter,
            cancelled=(reason == StoppingReason.USER_CANCELLED),
        )

    async def _evaluate_gaps_and_stopping(
        self,
        *,
        iter_num: int,
        session: ResearchSession,
        trace: ResearchTraceRecorder,
        budget: ExecutionBudgetTracker,
        handle: _SessionControlHandle,
        adapter: UniversalLocalModelAdapter | None,
        opts: dict[str, Any],
        prev_evidence_count: int,
        prev_coverage_ratio: float,
        snippet_only_map: dict[str, int],
    ) -> bool:
        """Evaluate newly collected evidence, update gaps/follow-ups, checkpoint, and decide stopping."""
        ResearchSessionStateManager.transition(
            session, AgentSessionStatus.EVALUATING, phase="evaluating"
        )
        prev_gap_ids = {g.gap_id for g in session.identified_gaps}
        evaluation = self._evaluator.evaluate_session(
            session, snippet_only_by_task=snippet_only_map
        )
        session.evidence_evaluation = evaluation
        trace.record(
            iteration=iter_num,
            phase="evaluating",
            event_type=TraceEventType.EVIDENCE_EVALUATED,
            reason=evaluation.summary,
            outcome=evaluation.overall_support_status.value,
            details={
                "coverage_ratio": evaluation.objective_coverage_ratio,
                "unique_primary_domains": evaluation.unique_primary_domains,
                "conflicts": evaluation.conflict_count,
            },
        )

        gap_report = self._gap_analyzer.analyze(
            session,
            evaluation,
            temporal_focus=opts["temporal_focus"],
        )
        for g in session.identified_gaps:
            if g.gap_id not in prev_gap_ids:
                trace.record(
                    iteration=iter_num,
                    phase="gap_analysis",
                    event_type=TraceEventType.GAP_IDENTIFIED,
                    task_id=g.related_task_id,
                    reason=g.description,
                    outcome=g.category.value,
                    details={"gap_id": g.gap_id, "severity": g.severity},
                )

        ResearchSessionStateManager.mark_addressed_gaps(session)
        if budget.usage.iterations_completed < iter_num:
            budget.mark_iteration_completed()

        ev_delta = len(session.evidence_records) - prev_evidence_count
        cov_delta = round(evaluation.objective_coverage_ratio - prev_coverage_ratio, 4)
        handle.iteration_evidence_deltas.append(ev_delta)
        handle.iteration_coverage_deltas.append(cov_delta)

        remaining_ready = [
            t
            for t in (session.research_plan.tasks if session.research_plan else [])
            if t.status in {TaskStatus.PENDING, TaskStatus.READY}
        ]
        if (
            not remaining_ready
            and gap_report.gaps
            and iter_num < budget.config.max_iterations
            and budget.primary_exhaustion_reason() is None
        ):
            ResearchSessionStateManager.transition(
                session, AgentSessionStatus.PLANNING, phase="planning_followups"
            )
            new_tasks, follow_note = await self._planner.create_followup_tasks(
                session=session,
                gaps=gap_report,
                budget=budget,
                adapter=adapter,
            )
            added, skipped_dups = TaskManager.merge_followup_tasks(session, new_tasks)
            for dup_cand, existing_match in skipped_dups:
                trace.record(
                    iteration=iter_num,
                    phase="planning_followups",
                    event_type=TraceEventType.TASK_SKIPPED,
                    task_id=dup_cand.task_id,
                    reason=(
                        f"Prevented duplicate follow-up matching existing task "
                        f"'{existing_match.task_id}'."
                    ),
                )
            if added:
                trace.record(
                    iteration=iter_num,
                    phase="planning_followups",
                    event_type=TraceEventType.PLAN_UPDATED,
                    reason=follow_note,
                    outcome=f"Added {len(added)} follow-up task(s)",
                )
                for nt in added:
                    trace.record(
                        iteration=iter_num,
                        phase="planning_followups",
                        event_type=TraceEventType.TASK_CREATED,
                        task_id=nt.task_id,
                        reason=nt.objective,
                        details={"parent_gap_id": nt.parent_gap_id, "depth": nt.depth},
                    )

        any_permitted_tool = any(
            self._registry.is_tool_available(d.name, session)[0]
            for d in self._registry.list_tools()
            if d.name != "local_llm_inference"
        )
        stop_decision = self._stopping.evaluate(
            session=session,
            budget=budget,
            evaluation=evaluation,
            gaps=gap_report,
            iteration_evidence_deltas=handle.iteration_evidence_deltas,
            iteration_coverage_deltas=handle.iteration_coverage_deltas,
            tools_available=any_permitted_tool,
            cancel_requested=handle.cancel_event.is_set(),
            pause_requested=handle.pause_event.is_set(),
        )
        session.stopping_decision = stop_decision
        trace.record(
            iteration=iter_num,
            phase="stopping_policy",
            event_type=TraceEventType.STOPPING_DECISION,
            reason=stop_decision.explanation,
            outcome=(
                stop_decision.reason.value
                if stop_decision.should_stop and stop_decision.reason
                else "continue"
            ),
            details={
                "should_stop": stop_decision.should_stop,
                "coverage": stop_decision.objective_coverage_ratio,
                "high_gaps": stop_decision.high_priority_gaps_remaining,
            },
        )
        await self._checkpoint(session, trace, budget)
        return stop_decision.should_stop

    async def _synthesize_and_complete(
        self,
        *,
        session: ResearchSession,
        trace: ResearchTraceRecorder,
        budget: ExecutionBudgetTracker,
        adapter: UniversalLocalModelAdapter | None,
        cancelled: bool = False,
    ) -> ResearchSession:
        if session.status != AgentSessionStatus.CANCELLED:
            ResearchSessionStateManager.transition(
                session, AgentSessionStatus.SYNTHESIZING, phase="synthesizing"
            )
        if session.evidence_evaluation is None:
            session.evidence_evaluation = self._evaluator.evaluate_session(session)

        package = self._build_stage4_package(session)
        await self._repository.save_research_package(package)
        session.research_package_id = package.research_id

        gen_result = await self._run_stage6_synthesis(
            session=session,
            package=package,
            budget=budget,
            adapter=adapter,
        )
        session.generation_result = gen_result
        await self._repository.save_generation_result(gen_result)

        trace.record(
            iteration=session.current_iteration,
            phase="synthesizing",
            event_type=TraceEventType.SYNTHESIS_COMPLETED,
            reason=(
                f"Produced cited Stage 6 report ({gen_result.model_id}, "
                f"{len(gen_result.citations)} cited source(s))."
            ),
            outcome=gen_result.status.value,
            details={
                "generation_id": gen_result.generation_id,
                "mode": gen_result.mode.value,
                "valid_citations": (
                    len(gen_result.citation_validation.used_citation_ids)
                    if gen_result.citation_validation
                    else len(gen_result.citations)
                ),
            },
        )
        target_status = (
            AgentSessionStatus.CANCELLED if cancelled else AgentSessionStatus.COMPLETED
        )
        if session.status != target_status:
            ResearchSessionStateManager.transition(
                session, target_status, phase=target_status.value
            )
        else:
            session.current_phase = target_status.value
        await self._checkpoint(session, trace, budget)
        return session

    def _build_stage4_package(self, session: ResearchSession) -> ResearchPackage:
        norm, intent, topics, keywords, subtopics, planned = decompose_query(
            session.original_query
        )
        proc = QueryPlan(
            original_query=session.original_query,
            normalized_query=norm,
            intent=intent,
            topics=topics,
            keywords=keywords,
            subtopics=subtopics,
            planned_queries=planned,
        )
        coverage = (
            session.evidence_evaluation.coverage_analysis
            if session.evidence_evaluation and session.evidence_evaluation.coverage_analysis
            else CoverageAnalysis()
        )
        state = ResearchState(
            research_id=f"pkg-{session.research_id}",
            original_query=session.original_query,
            status=ResearchStatus.COMPLETED,
            completed_stages=["planning", "retrieval", "evidence", "agent_orchestration"],
            queries_planned=len(session.disclosed_queries),
            urls_found=len(session.collected_sources),
            documents_fetched=len(session.collected_sources),
            documents_processed=sum(1 for s in session.collected_sources if not s.is_duplicate),
            duplicates_removed=sum(1 for s in session.collected_sources if s.is_duplicate),
            evidence_items_count=len(session.evidence_records),
        )
        return self._stage4_builder.build(
            state=state,
            query_plan=proc,
            search_queries=list(session.disclosed_queries),
            sources=list(session.collected_sources),
            evidence=list(session.evidence_records),
            claims=list(session.claims),
            conflicts=list(session.conflicts),
            coverage=coverage,
            provider_diagnostics=list(session.provider_diagnostics),
            privacy_policy=self._gateway.policy,
        )

    async def _run_stage6_synthesis(
        self,
        *,
        session: ResearchSession,
        package: ResearchPackage,
        budget: ExecutionBudgetTracker,
        adapter: UniversalLocalModelAdapter | None,
    ) -> GenerationResult:
        """Delegate synthesis to Stage 5 + Stage 6 when a local LLM is ready, else deterministic cited fallback."""
        # Build Stage 5 ContextPackage first (guarantees bounded context + global [S#] citation numbering)
        profile = (
            AllocationProfile.DEEP_RESEARCH
            if session.research_depth == ResearchDepth.DEEP
            else AllocationProfile.STANDARD_RESEARCH
        )
        if self._context_engine is not None:
            preview = await self._context_engine.preview_context(
                ContextPreviewRequest(
                    research_package=package,
                    question=session.original_query,
                    research_objective=session.research_objective,
                    connection_id=session.active_connection_id,
                    profile=profile,
                )
            )
            ctx_pkg = preview.context_package
            ctx_caps = ctx_pkg.capabilities
            ctx_val = preview.validation
        else:
            from web_intelligence.app.context.token_estimator import normalize_model_context_capabilities

            ctx_caps = normalize_model_context_capabilities(None)
            ctx_pkg = await StructuredContextBuilder().build_async(
                question=session.original_query,
                research_id=package.research_id,
                research_objective=session.research_objective,
                evidence_items=package.evidence,
                sources=package.sources,
                claims=package.claims,
                conflicts=package.conflicts,
                coverage=package.coverage,
                capabilities=ctx_caps,
                budget_config=ContextBudgetConfig(profile=profile),
                estimator=resolve_token_estimator(None),
            )
            ctx_val = None

        gap_notes = "\n".join(
            f"- [{g.severity.upper()}] {g.description}"
            for g in session.identified_gaps
            if not g.resolved
        )
        custom_instr = (
            "Synthesize an exhaustive, transparent research report covering each task outcome, "
            "explicitly acknowledging any conflicting claims without picking an unverified side, "
            "and listing remaining unresolved research gaps."
            + (f"\nUnresolved gaps to report truthfully:\n{gap_notes}" if gap_notes else "")
        )
        compiled = self._prompt_compiler.compile(
            ctx_pkg,
            template_type=PromptTemplateType.RESEARCH_REPORT,
            output_format=GenerationFormat.MARKDOWN,
            answer_length=(
                GenerationLength.LONG
                if session.research_depth == ResearchDepth.DEEP
                else GenerationLength.MEDIUM
            ),
            require_citations=bool(package.evidence),
            custom_instruction=custom_instr,
            token_estimator=resolve_token_estimator(adapter),
        )

        # Try Stage 6 local LLM generation when connected & within budget
        if (
            session.use_local_llm
            and self._executor is not None
            and adapter is not None
            and package.evidence
            and budget.usage.inference_calls < budget.config.max_inference_calls
        ):
            try:
                target = await self._executor.resolve_target(
                    connection_id=session.active_connection_id,
                    model_id=session.active_model_id,
                )
                gen_id = f"gen-{uuid.uuid4().hex[:12]}"
                req_mode = (
                    GenerationMode.LONG_FORM_REPORT
                    if self._long_form is not None
                    and (len(package.evidence) >= 4 or session.research_depth != ResearchDepth.QUICK)
                    else GenerationMode.STANDARD
                )
                req = GenerationRequest(
                    research_id=package.research_id,
                    connection_id=target.connection_id,
                    model_id=target.model_id,
                    generation_mode=req_mode,
                    template_type=PromptTemplateType.RESEARCH_REPORT,
                    output_format=GenerationFormat.MARKDOWN,
                    answer_length=GenerationLength.MEDIUM,
                    require_citations=True,
                    custom_instruction=custom_instr,
                    long_form_config=LongFormReportConfig(max_sections=5),
                )

                if (
                    req.generation_mode == GenerationMode.LONG_FORM_REPORT
                    and self._long_form is not None
                ):
                    final_res: GenerationResult | None = None
                    async for ev in self._long_form.stream_generate(
                        generation_id=gen_id,
                        request=req,
                        package=package,
                        target=target,
                        capabilities=ctx_caps,
                        global_citation_map=compiled.citation_map,
                    ):
                        if ev.get("event") == "section_complete":
                            budget.record_planner_or_synthesis_inference(estimated_tokens=300)
                        elif ev.get("event") == "result":
                            final_res = ev["result"]
                    if final_res is not None:
                        return final_res

                # Single-pass Stage 6 inference fallback
                budget.record_planner_or_synthesis_inference(
                    estimated_tokens=compiled.estimated_prompt_tokens
                )
                llm_resp, warn_list = await self._executor.execute(
                    target=target,
                    context_package=ctx_pkg,
                    compiled_prompt=compiled,
                    context_validation=ctx_val,
                )
                citations, val_rep = self._citation_validator.validate(
                    llm_resp.content,
                    compiled.citation_map,
                    stage5_to_citation_id=compiled.stage5_to_citation_id,
                    require_citations=True,
                )
                merged_c, src_map = merge_citations_preserving_order([citations])
                return GenerationResult(
                    generation_id=gen_id,
                    research_id=package.research_id,
                    status=GenerationStatus.COMPLETED,
                    mode=GenerationMode.STANDARD,
                    template_type=PromptTemplateType.RESEARCH_REPORT,
                    output_format=GenerationFormat.MARKDOWN,
                    connection_id=target.connection_id,
                    provider_id=target.provider_id,
                    server_endpoint=target.server_endpoint,
                    model_id=target.model_id,
                    content=llm_resp.content.strip(),
                    finish_reason=llm_resp.finish_reason,
                    truncated=llm_resp.finish_reason == "length",
                    citations=merged_c,
                    source_map=src_map,
                    citation_validation=val_rep,
                    usage=llm_resp.usage,
                    estimated_prompt_tokens=compiled.estimated_prompt_tokens,
                    warnings=[*compiled.warnings, *warn_list],
                    completed_at=utc_now(),
                )
            except (GenerationError, Exception) as exc:
                logger.warning(
                    "Stage 6 LLM synthesis unavailable; producing deterministic cited synthesis",
                    extra={"extra_data": {"error": redact_llm_error(exc)}},
                )

        # Deterministic cited fallback guaranteeing trustworthy grounded output offline
        return self._build_deterministic_synthesis(
            session=session,
            package=package,
            citation_map=compiled.citation_map,
            stage5_to_cid=compiled.stage5_to_citation_id,
        )

    def _build_deterministic_synthesis(
        self,
        *,
        session: ResearchSession,
        package: ResearchPackage,
        citation_map: dict[str, CitationEntry],
        stage5_to_cid: dict[str, str],
    ) -> GenerationResult:
        source_to_cid = {entry.source_id: cid for cid, entry in citation_map.items()}
        ev_by_id = {ev.evidence_id: ev for ev in package.evidence}
        tasks = list(session.research_plan.tasks) if session.research_plan else []
        assess_map = (
            {a.task_id: a for a in session.evidence_evaluation.task_assessments}
            if session.evidence_evaluation
            else {}
        )

        lines: list[str] = [
            f"# Research Report: {session.original_query}",
            "",
            "## Executive Overview",
            (
                f"Completed {session.current_iteration} autonomous research iteration(s) across "
                f"{len(session.collected_sources)} inspected source(s) ({len(package.evidence)} "
                f"evidence passage(s) retained, {len(package.conflicts)} conflict(s) detected)."
            ),
        ]
        if session.stopping_decision:
            lines.append(
                f"- **Stopping Status**: `{session.stopping_decision.reason.value if session.stopping_decision.reason else 'completed'}` — {session.stopping_decision.explanation}"
            )
        lines.append("")

        lines.append("## Task Findings")
        if not tasks:
            lines.append("No research tasks were executed.")
        for task in tasks:
            assess = assess_map.get(task.task_id)
            status_tag = assess.support_status.value if assess else task.status.value
            lines.append(f"### {task.task_id}: {task.question} ({status_tag})")
            task_ev = [ev_by_id[eid] for eid in task.evidence_ids if eid in ev_by_id]
            if not task_ev and package.evidence:
                # Match evidence by task assessment if linked semantically
                if assess and assess.evidence_ids:
                    task_ev = [ev_by_id[eid] for eid in assess.evidence_ids if eid in ev_by_id]
            if task_ev:
                for ev in task_ev[:3]:
                    cid = source_to_cid.get(ev.source_id)
                    cite = f" [{cid}]" if cid else ""
                    snippet = " ".join(ev.passage_text.split())
                    if len(snippet) > 280:
                        snippet = snippet[:277].rstrip() + "..."
                    lines.append(f"- {snippet}{cite}")
            else:
                lines.append(
                    "- No verified full-text passages resolved this task within the current run."
                )
            lines.append("")

        if package.conflicts:
            lines.append("## Conflicting Evidence")
            for conf in package.conflicts:
                cid_a = source_to_cid.get(conf.claim_a.source_id)
                cid_b = source_to_cid.get(conf.claim_b.source_id)
                c_a_ref = f" [{cid_a}]" if cid_a else ""
                c_b_ref = f" [{cid_b}]" if cid_b else ""
                lines.append(
                    f"- **{conf.topic}**: {conf.claim_a.source_domain} reports "
                    f"`{conf.claim_a.normalized_value}`{c_a_ref} whereas "
                    f"{conf.claim_b.source_domain} reports `{conf.claim_b.normalized_value}`{c_b_ref}."
                )
            lines.append("")

        unresolved = [g for g in session.identified_gaps if not g.resolved]
        lines.append("## Unresolved Gaps & Limitations")
        if unresolved:
            for gap in unresolved:
                lines.append(f"- **[{gap.severity.upper()} / {gap.category.value}]** {gap.description}")
        else:
            lines.append("- No remaining unresolved gaps detected.")

        content = "\n".join(lines).strip()
        citations, val_rep = self._citation_validator.validate(
            content,
            citation_map,
            stage5_to_citation_id=stage5_to_cid,
            require_citations=bool(citation_map),
        )
        merged_c, src_map = merge_citations_preserving_order([citations])
        return GenerationResult(
            generation_id=f"gen-{uuid.uuid4().hex[:12]}",
            research_id=package.research_id,
            status=GenerationStatus.COMPLETED,
            mode=GenerationMode.STANDARD,
            template_type=PromptTemplateType.RESEARCH_REPORT,
            output_format=GenerationFormat.MARKDOWN,
            connection_id="deterministic",
            provider_id="deterministic",
            server_endpoint="local://deterministic-fallback",
            model_id="deterministic_synthesis",
            content=content,
            finish_reason="stop",
            citations=merged_c,
            source_map=src_map,
            citation_validation=val_rep,
            warnings=(
                ["Local LLM unavailable or bypassed; produced deterministic cited fallback report."]
                if session.use_local_llm
                else []
            ),
            completed_at=utc_now(),
        )

    async def _checkpoint(
        self,
        session: ResearchSession,
        trace: ResearchTraceRecorder,
        budget: ExecutionBudgetTracker,
    ) -> None:
        """Persist buffered trace entries and updated ResearchSession state atomically."""
        if trace.entries:
            await self._repository.append_agent_trace_entries(trace.entries)
            trace.entries.clear()
        session.budget_usage = budget.usage
        ResearchSessionStateManager.sync_task_buckets(session)
        ResearchSessionStateManager.recompute_metrics(
            session, trace_entries_count=trace.sequence
        )
        await self._repository.save_agent_session(session)

    async def _require_session(self, research_id: str) -> ResearchSession:
        session = await self._repository.get_agent_session(research_id)
        if session is None:
            raise AgentSessionNotFoundError(f"Agent research session '{research_id}' not found.")
        return session

    async def _resolve_adapter(
        self, session: ResearchSession
    ) -> UniversalLocalModelAdapter | None:
        if not session.use_local_llm or self._llm_manager is None:
            return None
        try:
            cfg, adapter = await self._llm_manager.get_adapter(session.active_connection_id)
            session.active_connection_id = cfg.id
            if not session.active_model_id and cfg.model:
                session.active_model_id = cfg.model
            return adapter
        except Exception:
            return None

    async def _probe_default_model(
        self,
        connection_id: str | None,
        model_id: str | None,
    ) -> tuple[str | None, str | None]:
        if self._llm_manager is None:
            return connection_id, model_id
        try:
            cfg, _ = await self._llm_manager.get_adapter(connection_id)
            return cfg.id, model_id or (cfg.model or None)
        except Exception:
            return connection_id, model_id

    async def _load_private_note_phrases(self) -> list[str]:
        try:
            notes = await self._repository.list_workspace_notes("default")
            phrases: list[str] = []
            for note in notes:
                if not note.enabled:
                    continue
                if len(note.title.strip()) >= 6:
                    phrases.append(note.title.strip())
                for line in note.content.splitlines():
                    line_clean = line.strip()
                    if len(line_clean) >= 8:
                        phrases.append(line_clean[:120])
            return phrases[:20]
        except Exception:
            return []
