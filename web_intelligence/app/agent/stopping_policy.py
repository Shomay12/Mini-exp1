"""Multi-factor StoppingPolicy evaluating coverage, quality, open gaps, budgets, and user interrupts."""

from __future__ import annotations

from web_intelligence.app.agent.execution_budget import ExecutionBudgetTracker
from web_intelligence.app.agent.models import (
    EvidenceEvaluationReport,
    GapAnalysisReport,
    ResearchSession,
    StoppingDecision,
    StoppingReason,
    TaskStatus,
)


class StoppingPolicy:
    """Evaluates whether the autonomous research loop should proceed or halt with full transparency."""

    def __init__(self, *, sufficiency_threshold: float = 0.72) -> None:
        self._sufficiency_threshold = sufficiency_threshold

    def evaluate(
        self,
        *,
        session: ResearchSession,
        budget: ExecutionBudgetTracker,
        evaluation: EvidenceEvaluationReport | None,
        gaps: GapAnalysisReport | None,
        iteration_evidence_deltas: list[int],
        iteration_coverage_deltas: list[float],
        tools_available: bool = True,
        cancel_requested: bool = False,
        pause_requested: bool = False,
        waiting_approval: bool = False,
        model_hard_unavailable: bool = False,
    ) -> StoppingDecision:
        """Apply prioritized stopping rules combining user control, budgets, diminishing returns, and coverage."""
        cov_ratio = evaluation.objective_coverage_ratio if evaluation else 0.0
        high_gaps = gaps.high_priority_count if gaps else 0
        exhausted = budget.exhausted_dimensions()
        unresolved_summary = self._build_unresolved_summary(session, gaps)

        # 1. User / interactive control signals
        if cancel_requested:
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.USER_CANCELLED,
                explanation="Research stopped by user cancellation request.",
                objective_coverage_ratio=cov_ratio,
                high_priority_gaps_remaining=high_gaps,
                budget_exhausted_dimensions=exhausted,
                unresolved_items_summary=unresolved_summary,
            )
        if waiting_approval:
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.WAITING_USER_APPROVAL,
                explanation="Execution paused pending explicit user approval for an external tool action.",
                objective_coverage_ratio=cov_ratio,
                high_priority_gaps_remaining=high_gaps,
                budget_exhausted_dimensions=exhausted,
                unresolved_items_summary=unresolved_summary,
            )
        if pause_requested:
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.PAUSED_BY_USER,
                explanation="Execution paused cleanly at checkpoint by user request.",
                objective_coverage_ratio=cov_ratio,
                high_priority_gaps_remaining=high_gaps,
                budget_exhausted_dimensions=exhausted,
                unresolved_items_summary=unresolved_summary,
            )
        if model_hard_unavailable:
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.MODEL_UNAVAILABLE,
                explanation="Required local LLM became unavailable during strict model-only operation.",
                objective_coverage_ratio=cov_ratio,
                high_priority_gaps_remaining=high_gaps,
                budget_exhausted_dimensions=exhausted,
                unresolved_items_summary=unresolved_summary,
            )

        # 2. Hard budget exhaustion checks (independent of LLM claims)
        hard_reason = budget.primary_exhaustion_reason()
        if hard_reason is not None:
            return StoppingDecision(
                should_stop=True,
                reason=hard_reason,
                explanation=(
                    f"Hard execution budget limit reached ({', '.join(exhausted) or hard_reason.value})."
                ),
                objective_coverage_ratio=cov_ratio,
                high_priority_gaps_remaining=high_gaps,
                budget_exhausted_dimensions=exhausted,
                unresolved_items_summary=unresolved_summary,
            )

        # 3. Tools unavailable to make further progress
        pending_tasks = [
            t
            for t in (session.research_plan.tasks if session.research_plan else [])
            if t.status in {TaskStatus.PENDING, TaskStatus.READY}
        ]
        if not tools_available and pending_tasks:
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.TOOLS_UNAVAILABLE,
                explanation="No permitted tools are available to execute remaining research tasks.",
                objective_coverage_ratio=cov_ratio,
                high_priority_gaps_remaining=high_gaps,
                budget_exhausted_dimensions=exhausted,
                unresolved_items_summary=unresolved_summary,
            )

        # 4. Diminishing returns after at least 2 iterations or zero recent progress with no fresh gain
        if (
            len(iteration_evidence_deltas) >= 2
            and iteration_evidence_deltas[-1] == 0
            and abs(iteration_coverage_deltas[-1]) < 0.02
            and not pending_tasks
        ):
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.DIMINISHING_RETURNS,
                explanation=(
                    "Recent iteration yielded zero new accepted evidence passages and negligible "
                    f"coverage gain ({iteration_coverage_deltas[-1]:+.2f}); halting rather than looping."
                ),
                objective_coverage_ratio=cov_ratio,
                high_priority_gaps_remaining=high_gaps,
                budget_exhausted_dimensions=exhausted,
                unresolved_items_summary=unresolved_summary,
            )

        # 5. Sufficient objective coverage with no remaining runnable tasks and resolved high-priority gaps
        if (
            evaluation is not None
            and not pending_tasks
            and high_gaps == 0
            and cov_ratio >= self._sufficiency_threshold
            and evaluation.total_accepted_evidence > 0
        ):
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.OBJECTIVES_COVERED,
                explanation=(
                    f"Research objectives covered ({cov_ratio:.0%}) across "
                    f"{evaluation.unique_primary_domains} independent domain(s) with no remaining "
                    "high-priority gaps."
                ),
                objective_coverage_ratio=cov_ratio,
                high_priority_gaps_remaining=high_gaps,
                budget_exhausted_dimensions=exhausted,
                unresolved_items_summary=unresolved_summary,
            )

        # If all current tasks finished and zero follow-up gaps remain actionable, stop gracefully
        if not pending_tasks and (gaps is None or len(gaps.gaps) == 0):
            return StoppingDecision(
                should_stop=True,
                reason=(
                    StoppingReason.OBJECTIVES_COVERED
                    if (evaluation and evaluation.total_accepted_evidence > 0)
                    else StoppingReason.DIMINISHING_RETURNS
                ),
                explanation="All planned research tasks completed and no actionable follow-up gaps remain.",
                objective_coverage_ratio=cov_ratio,
                high_priority_gaps_remaining=0,
                budget_exhausted_dimensions=exhausted,
                unresolved_items_summary=unresolved_summary,
            )

        return StoppingDecision(
            should_stop=False,
            reason=None,
            explanation=(
                f"Continuing research: coverage={cov_ratio:.0%}, pending_tasks={len(pending_tasks)}, "
                f"high_priority_gaps={high_gaps}."
            ),
            objective_coverage_ratio=cov_ratio,
            high_priority_gaps_remaining=high_gaps,
            budget_exhausted_dimensions=exhausted,
            unresolved_items_summary=unresolved_summary,
        )

    @staticmethod
    def _build_unresolved_summary(
        session: ResearchSession,
        gaps: GapAnalysisReport | None,
    ) -> list[str]:
        items: list[str] = []
        if gaps:
            for g in gaps.gaps[:6]:
                items.append(f"[{g.severity.upper()} / {g.category.value}] {g.description}")
        elif session.identified_gaps:
            for g in session.identified_gaps:
                if not g.resolved:
                    items.append(f"[{g.severity.upper()} / {g.category.value}] {g.description}")
        return items[:8]
