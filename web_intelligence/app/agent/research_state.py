"""Explicit session state transitions, task synchronization, and duplicate-work guards."""

from __future__ import annotations

import re

from web_intelligence.app.agent.models import (
    AgentSessionStatus,
    ResearchGap,
    ResearchMetrics,
    ResearchSession,
    ResearchTask,
    TaskStatus,
    utc_now,
)

_NORM_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "the", "a", "an", "and", "or", "for", "of", "to", "in", "on", "with", "what",
    "how", "why", "which", "are", "is", "do", "does", "whether", "about",
})

ALLOWED_STATUS_TRANSITIONS: dict[AgentSessionStatus, set[AgentSessionStatus]] = {
    AgentSessionStatus.PENDING: {
        AgentSessionStatus.PLANNING,
        AgentSessionStatus.CANCELLED,
        AgentSessionStatus.FAILED,
    },
    AgentSessionStatus.PLANNING: {
        AgentSessionStatus.EXECUTING,
        AgentSessionStatus.PAUSED,
        AgentSessionStatus.WAITING_APPROVAL,
        AgentSessionStatus.SYNTHESIZING,
        AgentSessionStatus.CANCELLED,
        AgentSessionStatus.FAILED,
    },
    AgentSessionStatus.EXECUTING: {
        AgentSessionStatus.EVALUATING,
        AgentSessionStatus.PAUSED,
        AgentSessionStatus.WAITING_APPROVAL,
        AgentSessionStatus.SYNTHESIZING,
        AgentSessionStatus.CANCELLED,
        AgentSessionStatus.FAILED,
    },
    AgentSessionStatus.EVALUATING: {
        AgentSessionStatus.PLANNING,
        AgentSessionStatus.EXECUTING,
        AgentSessionStatus.PAUSED,
        AgentSessionStatus.SYNTHESIZING,
        AgentSessionStatus.CANCELLED,
        AgentSessionStatus.FAILED,
    },
    AgentSessionStatus.PAUSED: {
        AgentSessionStatus.PLANNING,
        AgentSessionStatus.EXECUTING,
        AgentSessionStatus.EVALUATING,
        AgentSessionStatus.SYNTHESIZING,
        AgentSessionStatus.CANCELLED,
        AgentSessionStatus.FAILED,
    },
    AgentSessionStatus.WAITING_APPROVAL: {
        AgentSessionStatus.EXECUTING,
        AgentSessionStatus.EVALUATING,
        AgentSessionStatus.PAUSED,
        AgentSessionStatus.SYNTHESIZING,
        AgentSessionStatus.CANCELLED,
        AgentSessionStatus.FAILED,
    },
    AgentSessionStatus.SYNTHESIZING: {
        AgentSessionStatus.COMPLETED,
        AgentSessionStatus.CANCELLED,
        AgentSessionStatus.FAILED,
    },
    AgentSessionStatus.COMPLETED: {
        # Allow reopening only via explicit follow-up/resume when new pending tasks are added
        AgentSessionStatus.PLANNING,
        AgentSessionStatus.EXECUTING,
    },
    AgentSessionStatus.CANCELLED: {
        AgentSessionStatus.EXECUTING,
        AgentSessionStatus.PLANNING,
    },
    AgentSessionStatus.FAILED: {
        AgentSessionStatus.EXECUTING,
        AgentSessionStatus.PLANNING,
    },
}


class InvalidStateTransitionError(ValueError):
    """Raised when an invalid AgentSessionStatus transition is attempted."""


def compute_task_fingerprint(question: str, keywords: list[str] | None = None) -> str:
    """Compute a stable normalized token fingerprint for duplicate-task detection."""
    tokens = [
        t
        for t in _NORM_RE.findall(f"{question} {' '.join(keywords or [])}".lower())
        if len(t) >= 2 and t not in _STOP
    ]
    unique_sorted = sorted(set(tokens))
    return " ".join(unique_sorted[:16])


def is_duplicate_task(
    candidate: ResearchTask,
    existing_tasks: list[ResearchTask],
    *,
    similarity_threshold: float = 0.78,
) -> ResearchTask | None:
    """Return matching existing task if candidate repeats completed or active work."""
    cand_fp = candidate.fingerprint or compute_task_fingerprint(
        candidate.question, candidate.entities_keywords
    )
    cand_tokens = set(cand_fp.split())
    if not cand_tokens:
        return None

    for existing in existing_tasks:
        if existing.status in {TaskStatus.CANCELLED}:
            continue
        if existing.task_id == candidate.task_id:
            return existing
        ex_fp = existing.fingerprint or compute_task_fingerprint(
            existing.question, existing.entities_keywords
        )
        if cand_fp == ex_fp:
            return existing
        ex_tokens = set(ex_fp.split())
        if not ex_tokens:
            continue
        union = cand_tokens | ex_tokens
        jaccard = len(cand_tokens & ex_tokens) / len(union) if union else 0.0
        if jaccard >= similarity_threshold:
            return existing
    return None


class ResearchSessionStateManager:
    """Coordinates validated session transitions, task partitions, and metrics."""

    @staticmethod
    def transition(
        session: ResearchSession,
        target: AgentSessionStatus,
        *,
        phase: str | None = None,
    ) -> ResearchSession:
        """Transition `session.status` along the explicit allowed state graph."""
        if session.status != target:
            allowed = ALLOWED_STATUS_TRANSITIONS.get(session.status, set())
            if target not in allowed:
                raise InvalidStateTransitionError(
                    f"Illegal session transition from '{session.status.value}' to '{target.value}'."
                )
            session.status = target
        if phase is not None:
            session.current_phase = phase
        session.updated_at = utc_now()
        if target == AgentSessionStatus.COMPLETED and session.completed_at is None:
            session.completed_at = session.updated_at
        return session

    @staticmethod
    def sync_task_buckets(session: ResearchSession) -> ResearchSession:
        """Reconcile `completed_tasks`, `pending_tasks`, and `failed_tasks` from `research_plan`."""
        tasks = list(session.research_plan.tasks) if session.research_plan else []
        completed: list[ResearchTask] = []
        pending: list[ResearchTask] = []
        failed: list[ResearchTask] = []

        for task in tasks:
            if not task.fingerprint:
                task.fingerprint = compute_task_fingerprint(task.question, task.entities_keywords)
            if task.status in {TaskStatus.COMPLETED, TaskStatus.SKIPPED}:
                completed.append(task)
            elif task.status == TaskStatus.FAILED:
                failed.append(task)
            elif task.status != TaskStatus.CANCELLED:
                pending.append(task)

        session.completed_tasks = completed
        session.pending_tasks = pending
        session.failed_tasks = failed
        return ResearchSessionStateManager.recompute_metrics(session)

    @staticmethod
    def recompute_metrics(
        session: ResearchSession,
        *,
        trace_entries_count: int | None = None,
    ) -> ResearchSession:
        """Recompute ResearchMetrics accurately from current persisted session state."""
        tasks = list(session.research_plan.tasks) if session.research_plan else []
        completed_n = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        skipped_n = sum(1 for t in tasks if t.status == TaskStatus.SKIPPED)
        failed_n = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        pending_n = sum(
            1
            for t in tasks
            if t.status in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.WAITING_APPROVAL}
        )
        retained_n = sum(1 for d in session.source_decisions if d.retained)
        rejected_n = sum(1 for d in session.source_decisions if not d.retained)
        # Fallback if decisions weren't logged individually
        if not session.source_decisions and session.collected_sources:
            retained_n = sum(1 for s in session.collected_sources if not s.is_duplicate)
            rejected_n = sum(1 for s in session.collected_sources if s.is_duplicate)

        unresolved_gaps = sum(1 for g in session.identified_gaps if not g.resolved)
        prev_trace_count = session.research_metrics.trace_entries_count
        session.research_metrics = ResearchMetrics(
            tasks_total=len(tasks),
            tasks_completed=completed_n,
            tasks_pending=pending_n,
            tasks_failed=failed_n,
            tasks_skipped=skipped_n,
            sources_collected=len(session.collected_sources),
            sources_retained=retained_n,
            sources_rejected=rejected_n,
            evidence_accepted=len(session.evidence_records),
            evidence_rejected=rejected_n,
            claims_extracted=len(session.claims),
            conflicts_detected=len(session.conflicts),
            gaps_identified=len(session.identified_gaps),
            gaps_unresolved=unresolved_gaps,
            trace_entries_count=(
                trace_entries_count if trace_entries_count is not None else prev_trace_count
            ),
        )
        session.updated_at = utc_now()
        return session

    @staticmethod
    def mark_addressed_gaps(session: ResearchSession) -> list[ResearchGap]:
        """Mark gaps as resolved when their assigned follow-up task completed with evidence."""
        if not session.research_plan:
            return session.identified_gaps
        task_map = {t.task_id: t for t in session.research_plan.tasks}
        for gap in session.identified_gaps:
            if gap.addressed_by_task_id and gap.addressed_by_task_id in task_map:
                t = task_map[gap.addressed_by_task_id]
                if t.status == TaskStatus.COMPLETED and t.evidence_ids:
                    gap.resolved = True
        return session.identified_gaps
