"""Independent execution budget enforcement for bounded Stage 7 agent research."""

from __future__ import annotations

from web_intelligence.app.agent.models import (
    ExecutionBudgetConfig,
    ExecutionBudgetUsage,
    ResearchDepth,
    StoppingReason,
)


def default_budget_for_depth(
    depth: ResearchDepth,
    *,
    require_approval_for_external: bool = False,
) -> ExecutionBudgetConfig:
    """Return conservative, bounded default execution limits scaled by requested depth."""
    if depth == ResearchDepth.QUICK:
        return ExecutionBudgetConfig(
            max_iterations=2,
            max_search_calls=4,
            max_page_retrievals=6,
            max_concurrent_tasks=2,
            max_execution_seconds=60.0,
            max_inference_calls=8,
            max_estimated_tokens=32_000,
            max_research_depth=1,
            tool_timeout_seconds=20.0,
            max_followups_per_iteration=2,
            require_approval_for_external_search=require_approval_for_external,
        )
    if depth == ResearchDepth.DEEP:
        return ExecutionBudgetConfig(
            max_iterations=5,
            max_search_calls=14,
            max_page_retrievals=24,
            max_concurrent_tasks=3,
            max_execution_seconds=240.0,
            max_inference_calls=30,
            max_estimated_tokens=120_000,
            max_research_depth=3,
            tool_timeout_seconds=30.0,
            max_followups_per_iteration=4,
            require_approval_for_external_search=require_approval_for_external,
        )
    return ExecutionBudgetConfig(
        max_iterations=3,
        max_search_calls=8,
        max_page_retrievals=12,
        max_concurrent_tasks=2,
        max_execution_seconds=120.0,
        max_inference_calls=16,
        max_estimated_tokens=64_000,
        max_research_depth=2,
        tool_timeout_seconds=25.0,
        max_followups_per_iteration=3,
        require_approval_for_external_search=require_approval_for_external,
    )


class ExecutionBudgetTracker:
    """Validates and consumes hard execution quotas independently of LLM proposals."""

    def __init__(
        self,
        config: ExecutionBudgetConfig,
        usage: ExecutionBudgetUsage | None = None,
    ) -> None:
        self.config = config
        self.usage = usage or ExecutionBudgetUsage()

    def exhausted_dimensions(self) -> list[str]:
        """List every budget dimension that has reached or exceeded its hard ceiling."""
        exhausted: list[str] = []
        if self.usage.iterations_completed >= self.config.max_iterations:
            exhausted.append("max_iterations")
        if self.usage.search_calls >= self.config.max_search_calls:
            exhausted.append("max_search_calls")
        if self.usage.page_retrievals >= self.config.max_page_retrievals:
            exhausted.append("max_page_retrievals")
        if self.usage.elapsed_seconds >= self.config.max_execution_seconds:
            exhausted.append("max_execution_seconds")
        if self.usage.inference_calls >= self.config.max_inference_calls and self.config.max_inference_calls > 0:
            exhausted.append("max_inference_calls")
        if self.usage.estimated_tokens >= self.config.max_estimated_tokens:
            exhausted.append("max_estimated_tokens")
        return exhausted

    def primary_exhaustion_reason(self) -> StoppingReason | None:
        """Return highest-priority StoppingReason when a hard execution cap is hit."""
        if self.usage.elapsed_seconds >= self.config.max_execution_seconds:
            return StoppingReason.TIME_BUDGET_EXHAUSTED
        if self.usage.iterations_completed >= self.config.max_iterations:
            return StoppingReason.MAX_ITERATIONS_REACHED
        if self.usage.estimated_tokens >= self.config.max_estimated_tokens:
            return StoppingReason.TOKEN_BUDGET_EXHAUSTED
        if self.usage.search_calls >= self.config.max_search_calls:
            return StoppingReason.SEARCH_BUDGET_REACHED
        if self.usage.page_retrievals >= self.config.max_page_retrievals:
            return StoppingReason.FETCH_BUDGET_REACHED
        if (
            self.config.max_inference_calls > 0
            and self.usage.inference_calls >= self.config.max_inference_calls
        ):
            return StoppingReason.INFERENCE_BUDGET_REACHED
        return None

    def can_start_iteration(self, next_iteration: int) -> tuple[bool, str | None]:
        """Return whether another research iteration is allowed under time/iteration caps."""
        if self.usage.elapsed_seconds >= self.config.max_execution_seconds:
            return False, (
                f"Execution time limit reached ({self.usage.elapsed_seconds:.1f}s / "
                f"{self.config.max_execution_seconds:.1f}s)."
            )
        if next_iteration > self.config.max_iterations:
            return False, (
                f"Maximum iteration limit reached ({self.usage.iterations_completed} / "
                f"{self.config.max_iterations})."
            )
        if self.usage.estimated_tokens >= self.config.max_estimated_tokens:
            return False, (
                f"Estimated token budget exhausted ({self.usage.estimated_tokens} / "
                f"{self.config.max_estimated_tokens})."
            )
        return True, None

    def can_execute_tool(self, tool_name: str, *, task_depth: int = 0) -> tuple[bool, str | None]:
        """Enforce per-tool hard ceilings before executing a proposed action."""
        if self.usage.elapsed_seconds >= self.config.max_execution_seconds:
            return False, "Execution time budget exhausted."
        if task_depth > self.config.max_research_depth:
            return False, f"Task depth {task_depth} exceeds max_research_depth ({self.config.max_research_depth})."
        if self.usage.estimated_tokens >= self.config.max_estimated_tokens:
            return False, "Estimated token budget exhausted."
        if tool_name == "web_search" and self.usage.search_calls >= self.config.max_search_calls:
            return False, f"Search call limit reached ({self.config.max_search_calls})."
        if tool_name == "url_retrieve" and self.usage.page_retrievals >= self.config.max_page_retrievals:
            return False, f"Page retrieval limit reached ({self.config.max_page_retrievals})."
        if tool_name == "local_llm_inference":
            if self.config.max_inference_calls <= 0:
                return False, "Local LLM inference calls disabled by budget config (max_inference_calls=0)."
            if self.usage.inference_calls >= self.config.max_inference_calls:
                return False, f"Inference call limit reached ({self.config.max_inference_calls})."
        return True, None

    def remaining_fetches(self) -> int:
        return max(0, self.config.max_page_retrievals - self.usage.page_retrievals)

    def remaining_searches(self) -> int:
        return max(0, self.config.max_search_calls - self.usage.search_calls)

    def record_tool_usage(
        self,
        tool_name: str,
        *,
        success: bool,
        search_calls: int = 0,
        page_retrievals: int = 0,
        inference_calls: int = 0,
        estimated_tokens: int = 0,
        task_depth: int = 0,
    ) -> None:
        """Increment cumulative consumption counters after a tool execution."""
        self.usage.tool_calls_total += 1
        if not success:
            self.usage.tool_failures_total += 1
        self.usage.search_calls += max(0, search_calls)
        self.usage.page_retrievals += max(0, page_retrievals)
        self.usage.inference_calls += max(0, inference_calls)
        if tool_name == "web_search" and search_calls == 0:
            self.usage.search_calls += 1
        elif tool_name == "url_retrieve" and page_retrievals == 0:
            self.usage.page_retrievals += 1
        elif tool_name == "local_llm_inference" and inference_calls == 0:
            self.usage.inference_calls += 1
        self.usage.estimated_tokens += max(0, estimated_tokens)
        if task_depth > self.usage.max_depth_reached:
            self.usage.max_depth_reached = task_depth

    def record_planner_or_synthesis_inference(self, *, estimated_tokens: int = 0) -> bool:
        """Consume one LLM inference call when within cap; returns False if over cap."""
        if self.config.max_inference_calls > 0 and self.usage.inference_calls >= self.config.max_inference_calls:
            return False
        self.usage.inference_calls += 1
        self.usage.estimated_tokens += max(0, estimated_tokens)
        return True

    def record_elapsed_seconds(self, delta_seconds: float) -> None:
        self.usage.estimated_tokens = max(0, self.usage.estimated_tokens)
        self.usage.elapsed_seconds = round(self.usage.elapsed_seconds + max(0.0, delta_seconds), 3)

    def mark_iteration_completed(self) -> None:
        self.usage.iterations_completed += 1
