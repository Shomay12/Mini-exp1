"""Bounded task execution engine with retries, timeouts, concurrency caps, and approval gates."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from web_intelligence.app.agent.execution_budget import ExecutionBudgetTracker
from web_intelligence.app.agent.models import (
    NormalizedObservation,
    PendingApprovalRequest,
    ResearchSession,
    ResearchTask,
    TaskStatus,
    ToolExecutionResult,
    TraceEventType,
    utc_now,
)
from web_intelligence.app.agent.observation_processor import ObservationProcessor
from web_intelligence.app.agent.research_memory import ResearchWorkingMemory
from web_intelligence.app.agent.research_trace import ResearchTraceRecorder
from web_intelligence.app.agent.tool_registry import ToolApprovalRequiredError, ToolRegistry
from web_intelligence.app.agent.tool_selector import ToolSelector
from web_intelligence.app.llm.base import UniversalLocalModelAdapter
from web_intelligence.app.schemas.search import NormalizedSearchResult


@dataclass
class TaskExecutionOutcome:
    """Structured result of executing one ResearchTask."""

    task: ResearchTask
    observation: NormalizedObservation | None = None
    approval_required: PendingApprovalRequest | None = None
    paused: bool = False
    cancelled: bool = False
    budget_exhausted: bool = False
    error: str | None = None


@dataclass
class BatchExecutionOutcome:
    """Aggregated outcome for a concurrent or sequential batch of tasks."""

    task_outcomes: list[TaskExecutionOutcome] = field(default_factory=list)
    approval_requests: list[PendingApprovalRequest] = field(default_factory=list)
    paused: bool = False
    cancelled: bool = False
    budget_exhausted: bool = False
    snippet_only_by_task: dict[str, int] = field(default_factory=dict)


class AgentExecutionEngine:
    """Executes scheduled ResearchTasks through ToolSelector -> ToolRegistry -> ObservationProcessor."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        selector: ToolSelector,
        observation_processor: ObservationProcessor,
        memory: ResearchWorkingMemory | None = None,
    ) -> None:
        self._registry = registry
        self._selector = selector
        self._observer = observation_processor
        self._memory = memory or ResearchWorkingMemory()

    async def execute_batch(
        self,
        *,
        tasks: list[ResearchTask],
        session: ResearchSession,
        budget: ExecutionBudgetTracker,
        trace: ResearchTraceRecorder,
        adapter: UniversalLocalModelAdapter | None = None,
        pause_event: asyncio.Event | None = None,
        cancel_event: asyncio.Event | None = None,
        approved_proposals: set[tuple[str, str]] | None = None,
        max_results_per_query: int = 5,
        chunk_target_tokens: int = 180,
        blocked_private_terms: list[str] | None = None,
    ) -> BatchExecutionOutcome:
        """Execute a batch of ready tasks bounded by max_concurrent_tasks."""
        if not tasks:
            return BatchExecutionOutcome()

        semaphore = asyncio.Semaphore(max(1, budget.config.max_concurrent_tasks))
        state_lock = asyncio.Lock()

        async def _run_one(task: ResearchTask) -> TaskExecutionOutcome:
            async with semaphore:
                return await self._execute_single_task(
                    task=task,
                    session=session,
                    budget=budget,
                    trace=trace,
                    state_lock=state_lock,
                    adapter=adapter,
                    pause_event=pause_event,
                    cancel_event=cancel_event,
                    approved_proposals=approved_proposals,
                    max_results_per_query=max_results_per_query,
                    chunk_target_tokens=chunk_target_tokens,
                    blocked_private_terms=blocked_private_terms,
                )

        outcomes = await asyncio.gather(*(_run_one(t) for t in tasks))
        batch_out = BatchExecutionOutcome(task_outcomes=list(outcomes))
        for item in outcomes:
            if item.approval_required is not None:
                batch_out.approval_requests.append(item.approval_required)
            if item.paused:
                batch_out.paused = True
            if item.cancelled:
                batch_out.cancelled = True
            if item.budget_exhausted:
                batch_out.budget_exhausted = True
            if item.observation is not None:
                batch_out.snippet_only_by_task[item.task.task_id] = item.observation.snippet_only_hits
        return batch_out

    async def execute_approved_action(
        self,
        *,
        approval: PendingApprovalRequest,
        task: ResearchTask,
        session: ResearchSession,
        budget: ExecutionBudgetTracker,
        trace: ResearchTraceRecorder,
        max_results_per_query: int = 5,
        chunk_target_tokens: int = 180,
        blocked_private_terms: list[str] | None = None,
    ) -> TaskExecutionOutcome:
        """Execute an explicitly user-approved external tool action and normalize its observation."""
        can_run, cap_reason = budget.can_execute_tool(approval.tool_name, task_depth=task.depth)
        if not can_run:
            task.status = TaskStatus.PENDING
            return TaskExecutionOutcome(task=task, budget_exhausted=True, error=cap_reason)

        task.status = TaskStatus.RUNNING
        task.attempts += 1
        task.updated_at = utc_now()
        results: list[ToolExecutionResult] = []

        res = await self._registry.execute(
            tool_name=approval.tool_name,
            task_id=task.task_id,
            arguments=approval.proposed_inputs,
            session=session,
            bypass_approval_check=True,
            blocked_private_terms=blocked_private_terms,
        )
        results.append(res)
        self._record_budget_and_trace(
            res=res,
            task=task,
            session=session,
            budget=budget,
            trace=trace,
            reason=f"Executed user-approved '{approval.tool_name}' action ({approval.approval_id}).",
        )

        # Follow-on URL retrieval if web_search succeeded & remote URLs don't separately require fresh approval
        if res.success and res.tool_name == "web_search":
            chained = await self._maybe_fetch_search_hits(
                search_res=res,
                task=task,
                session=session,
                budget=budget,
                trace=trace,
                state_lock=None,
                bypass_approval=not session.execution_budget.require_approval_for_remote_urls
                and not session.require_user_approval_for_external,
            )
            if isinstance(chained, PendingApprovalRequest):
                task.status = TaskStatus.WAITING_APPROVAL
                task.updated_at = utc_now()
                session.pending_approvals.append(chained)
                obs = await self._observer.process_task_results(
                    task=task,
                    session=session,
                    results=results,
                    chunk_target_tokens=chunk_target_tokens,
                )
                return TaskExecutionOutcome(
                    task=task,
                    observation=obs,
                    approval_required=chained,
                )
            if chained is not None:
                results.append(chained)

        obs = await self._observer.process_task_results(
            task=task,
            session=session,
            results=results,
            chunk_target_tokens=chunk_target_tokens,
        )
        self._trace_source_decisions(session, trace, task, obs)
        task.status = TaskStatus.COMPLETED
        task.result_summary = obs.summary
        task.completed_at = utc_now()
        task.updated_at = task.completed_at
        return TaskExecutionOutcome(task=task, observation=obs)

    async def _execute_single_task(
        self,
        *,
        task: ResearchTask,
        session: ResearchSession,
        budget: ExecutionBudgetTracker,
        trace: ResearchTraceRecorder,
        state_lock: asyncio.Lock,
        adapter: UniversalLocalModelAdapter | None,
        pause_event: asyncio.Event | None,
        cancel_event: asyncio.Event | None,
        approved_proposals: set[tuple[str, str]] | None,
        max_results_per_query: int,
        chunk_target_tokens: int,
        blocked_private_terms: list[str] | None,
    ) -> TaskExecutionOutcome:
        if cancel_event is not None and cancel_event.is_set():
            task.status = TaskStatus.CANCELLED
            task.updated_at = utc_now()
            return TaskExecutionOutcome(task=task, cancelled=True)
        if pause_event is not None and pause_event.is_set():
            task.status = TaskStatus.PENDING
            task.updated_at = utc_now()
            return TaskExecutionOutcome(task=task, paused=True)

        async with state_lock:
            session.current_task_id = task.task_id
            task.status = TaskStatus.RUNNING
            task.attempts += 1
            task.updated_at = utc_now()

        # Select tools (can consult connected LLM cleanly, else deterministic baseline)
        proposals, selector_note = await self._selector.select_tools_for_task(
            task,
            session,
            adapter=(
                adapter
                if adapter is not None
                and session.use_local_llm
                and budget.usage.inference_calls < budget.config.max_inference_calls
                else None
            ),
            max_results_per_query=max_results_per_query,
        )
        if "local_llm_validated" in selector_note:
            budget.record_planner_or_synthesis_inference(estimated_tokens=180)

        async with state_lock:
            for prop in proposals:
                trace.record(
                    iteration=session.current_iteration,
                    phase="tool_selection",
                    event_type=TraceEventType.TOOL_SELECTED,
                    task_id=task.task_id,
                    tool_name=prop.tool_name,
                    sanitized_inputs=prop.arguments,
                    reason=prop.rationale,
                    details={"selected_by": prop.selected_by, "note": selector_note},
                )

        if not proposals:
            async with state_lock:
                task.status = TaskStatus.FAILED
                task.error = selector_note or "No permitted tools selected for task."
                task.updated_at = utc_now()
            return TaskExecutionOutcome(task=task, error=task.error)

        results: list[ToolExecutionResult] = []
        for prop in proposals:
            if cancel_event is not None and cancel_event.is_set():
                async with state_lock:
                    task.status = TaskStatus.CANCELLED
                    task.updated_at = utc_now()
                return TaskExecutionOutcome(task=task, cancelled=True)
            if pause_event is not None and pause_event.is_set():
                async with state_lock:
                    task.status = TaskStatus.PENDING
                    task.updated_at = utc_now()
                return TaskExecutionOutcome(task=task, paused=True)

            async with state_lock:
                can_run, cap_msg = budget.can_execute_tool(prop.tool_name, task_depth=task.depth)
                if not can_run:
                    trace.record(
                        iteration=session.current_iteration,
                        phase="tool_execution",
                        event_type=TraceEventType.TOOL_REJECTED,
                        task_id=task.task_id,
                        tool_name=prop.tool_name,
                        reason=cap_msg or "Execution budget limit reached.",
                    )
                    task.status = TaskStatus.PENDING
                    task.updated_at = utc_now()
                    return TaskExecutionOutcome(task=task, budget_exhausted=True, error=cap_msg)

            bypass = bool(
                approved_proposals and (task.task_id, prop.tool_name) in approved_proposals
            )
            exec_res = await self._execute_with_retry_or_approval(
                tool_name=prop.tool_name,
                task=task,
                arguments=prop.arguments,
                session=session,
                budget=budget,
                trace=trace,
                state_lock=state_lock,
                bypass_approval=bypass,
                blocked_private_terms=blocked_private_terms,
            )
            if isinstance(exec_res, PendingApprovalRequest):
                async with state_lock:
                    task.status = TaskStatus.WAITING_APPROVAL
                    task.updated_at = utc_now()
                    session.pending_approvals.append(exec_res)
                    trace.record(
                        iteration=session.current_iteration,
                        phase="approval_gate",
                        event_type=TraceEventType.APPROVAL_REQUESTED,
                        task_id=task.task_id,
                        tool_name=exec_res.tool_name,
                        sanitized_inputs=exec_res.proposed_inputs,
                        reason=exec_res.reason,
                        details={"approval_id": exec_res.approval_id},
                    )
                return TaskExecutionOutcome(task=task, approval_required=exec_res)

            results.append(exec_res)

            # Chain url_retrieve after web_search hits for full-text primary extraction
            if exec_res.success and exec_res.tool_name == "web_search":
                chained = await self._maybe_fetch_search_hits(
                    search_res=exec_res,
                    task=task,
                    session=session,
                    budget=budget,
                    trace=trace,
                    state_lock=state_lock,
                    bypass_approval=bool(
                        approved_proposals and (task.task_id, "url_retrieve") in approved_proposals
                    ),
                )
                if isinstance(chained, PendingApprovalRequest):
                    async with state_lock:
                        task.status = TaskStatus.WAITING_APPROVAL
                        task.updated_at = utc_now()
                        session.pending_approvals.append(chained)
                        obs = await self._observer.process_task_results(
                            task=task,
                            session=session,
                            results=results,
                            chunk_target_tokens=chunk_target_tokens,
                        )
                        trace.record(
                            iteration=session.current_iteration,
                            phase="approval_gate",
                            event_type=TraceEventType.APPROVAL_REQUESTED,
                            task_id=task.task_id,
                            tool_name=chained.tool_name,
                            sanitized_inputs=chained.proposed_inputs,
                            reason=chained.reason,
                            details={"approval_id": chained.approval_id},
                        )
                    return TaskExecutionOutcome(
                        task=task,
                        observation=obs,
                        approval_required=chained,
                    )
                if chained is not None:
                    results.append(chained)

        # Normalize observations safely under lock
        async with state_lock:
            obs = await self._observer.process_task_results(
                task=task,
                session=session,
                results=results,
                chunk_target_tokens=chunk_target_tokens,
            )
            self._trace_source_decisions(session, trace, task, obs)

            any_success = any(r.success for r in results)
            if any_success or obs.new_evidence or obs.snippet_only_hits > 0:
                task.status = TaskStatus.COMPLETED
                task.result_summary = obs.summary
                task.completed_at = utc_now()
            else:
                task.status = TaskStatus.FAILED
                err_txt = "; ".join(obs.warnings) or "All tool attempts failed."
                task.error = err_txt
                task.result_summary = err_txt
            task.updated_at = utc_now()

        return TaskExecutionOutcome(task=task, observation=obs, error=task.error)

    async def _execute_with_retry_or_approval(
        self,
        *,
        tool_name: str,
        task: ResearchTask,
        arguments: dict[str, Any],
        session: ResearchSession,
        budget: ExecutionBudgetTracker,
        trace: ResearchTraceRecorder,
        state_lock: asyncio.Lock | None,
        bypass_approval: bool,
        blocked_private_terms: list[str] | None,
    ) -> ToolExecutionResult | PendingApprovalRequest:
        max_tries = 2 if tool_name in {"web_search", "url_retrieve"} else 1
        last_result: ToolExecutionResult | None = None

        for attempt in range(max_tries):
            try:
                res = await self._registry.execute(
                    tool_name=tool_name,
                    task_id=task.task_id,
                    arguments=arguments,
                    session=session,
                    bypass_approval_check=bypass_approval,
                    blocked_private_terms=blocked_private_terms,
                )
            except ToolApprovalRequiredError as gate_exc:
                return PendingApprovalRequest(
                    approval_id=f"apr-{uuid.uuid4().hex[:8]}",
                    research_id=session.research_id,
                    task_id=task.task_id,
                    tool_name=gate_exc.tool_name,
                    reason=gate_exc.reason,
                    external_network=True,
                    proposed_inputs=gate_exc.sanitized_inputs,
                )

            if state_lock is not None:
                async with state_lock:
                    self._record_budget_and_trace(
                        res=res,
                        task=task,
                        session=session,
                        budget=budget,
                        trace=trace,
                        reason=f"Tool '{tool_name}' attempt {attempt + 1}/{max_tries}.",
                    )
            else:
                self._record_budget_and_trace(
                    res=res,
                    task=task,
                    session=session,
                    budget=budget,
                    trace=trace,
                    reason=f"Tool '{tool_name}' attempt {attempt + 1}/{max_tries}.",
                )

            last_result = res
            if res.success or res.error_code in {
                "tool_unavailable",
                "invalid_tool_input",
                "privacy_sanitized_empty",
            }:
                return res
            # Check remaining budget before retrying transient network/timeout error
            if state_lock is not None:
                async with state_lock:
                    if not budget.can_execute_tool(tool_name, task_depth=task.depth)[0]:
                        return res
            elif not budget.can_execute_tool(tool_name, task_depth=task.depth)[0]:
                return res

        assert last_result is not None
        return last_result

    async def _maybe_fetch_search_hits(
        self,
        *,
        search_res: ToolExecutionResult,
        task: ResearchTask,
        session: ResearchSession,
        budget: ExecutionBudgetTracker,
        trace: ResearchTraceRecorder,
        state_lock: asyncio.Lock | None,
        bypass_approval: bool,
    ) -> ToolExecutionResult | PendingApprovalRequest | None:
        raw_hits = search_res.data.get("hits", [])
        if not raw_hits:
            return None

        known_urls = self._memory.known_canonical_urls(session)
        candidates: list[NormalizedSearchResult] = []
        for item in raw_hits:
            hit = NormalizedSearchResult.model_validate(item)
            if hit.url in known_urls or (hit.canonical_url and hit.canonical_url in known_urls):
                continue
            candidates.append(hit)

        if not candidates:
            return None

        if state_lock is not None:
            async with state_lock:
                can_fetch, reason = budget.can_execute_tool("url_retrieve", task_depth=task.depth)
                rem = budget.remaining_fetches()
                if not can_fetch or rem <= 0:
                    trace.record(
                        iteration=session.current_iteration,
                        phase="tool_execution",
                        event_type=TraceEventType.TOOL_REJECTED,
                        task_id=task.task_id,
                        tool_name="url_retrieve",
                        reason=reason or "Page retrieval budget exhausted.",
                    )
                    return None
        else:
            can_fetch, reason = budget.can_execute_tool("url_retrieve", task_depth=task.depth)
            rem = budget.remaining_fetches()
            if not can_fetch or rem <= 0:
                return None

        selected = candidates[: min(rem, 4)]
        args = {
            "urls": [h.url for h in selected],
            "snippets_by_url": {h.url: h.snippet for h in selected},
            "titles_by_url": {h.url: h.title for h in selected},
            "providers_by_url": {h.url: h.provider for h in selected},
            "max_urls": len(selected),
        }
        return await self._execute_with_retry_or_approval(
            tool_name="url_retrieve",
            task=task,
            arguments=args,
            session=session,
            budget=budget,
            trace=trace,
            state_lock=state_lock,
            bypass_approval=bypass_approval,
            blocked_private_terms=None,
        )

    @staticmethod
    def _record_budget_and_trace(
        *,
        res: ToolExecutionResult,
        task: ResearchTask,
        session: ResearchSession,
        budget: ExecutionBudgetTracker,
        trace: ResearchTraceRecorder,
        reason: str,
    ) -> None:
        if res.tool_name not in task.assigned_tools_used:
            task.assigned_tools_used.append(res.tool_name)
        pages_n = int(res.data.get("pages_fetched", 0)) if res.tool_name == "url_retrieve" else 0
        budget.record_tool_usage(
            res.tool_name,
            success=res.success,
            search_calls=1 if res.tool_name == "web_search" else 0,
            page_retrievals=max(1, pages_n) if res.tool_name == "url_retrieve" else 0,
            inference_calls=1 if res.tool_name == "local_llm_inference" else 0,
            estimated_tokens=res.estimated_tokens_consumed,
            task_depth=task.depth,
        )
        outcome_txt = "success" if res.success else f"failed ({res.error_code or 'error'})"
        trace.record(
            iteration=session.current_iteration,
            phase="tool_execution",
            event_type=TraceEventType.TOOL_EXECUTED if res.success else TraceEventType.TOOL_REJECTED,
            task_id=task.task_id,
            tool_name=res.tool_name,
            sanitized_inputs=res.sanitized_inputs,
            outcome=outcome_txt,
            reason=reason if res.success else (res.error or reason),
            details={"duration_ms": res.duration_ms, "used_external_network": res.used_external_network},
        )

    @staticmethod
    def _trace_source_decisions(
        session: ResearchSession,
        trace: ResearchTraceRecorder,
        task: ResearchTask,
        obs: NormalizedObservation,
    ) -> None:
        for dec in obs.source_decisions:
            trace.record(
                iteration=session.current_iteration,
                phase="observation",
                event_type=(
                    TraceEventType.SOURCE_RETAINED
                    if dec.retained
                    else TraceEventType.SOURCE_REJECTED
                ),
                task_id=task.task_id,
                sanitized_inputs={"url": dec.url, "domain": dec.domain},
                outcome="retained" if dec.retained else "rejected",
                reason=dec.reason,
                details={
                    "source_id": dec.source_id,
                    "relevance_score": dec.relevance_score,
                    "is_duplicate": dec.is_duplicate,
                },
            )
