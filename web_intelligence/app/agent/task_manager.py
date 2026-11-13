"""DAG dependency validation, priority scheduling, duplicate prevention, and safe user plan edits."""

from __future__ import annotations

from web_intelligence.app.agent.models import (
    AgentPlanEditRequest,
    ResearchPlanSpec,
    ResearchSession,
    ResearchTask,
    TaskStatus,
    utc_now,
)
from web_intelligence.app.agent.research_state import (
    ResearchSessionStateManager,
    compute_task_fingerprint,
    is_duplicate_task,
)


class PlanValidationError(ValueError):
    """Raised when a task graph fails dependency, cycle, or edit safety rules."""


class TaskManager:
    """Manages task readiness, DAG dependencies, follow-up merging, and validated plan edits."""

    @staticmethod
    def validate_dag(tasks: list[ResearchTask]) -> None:
        """Verify every dependency references a valid task_id and contains zero cycles."""
        by_id = {t.task_id: t for t in tasks}
        if len(by_id) != len(tasks):
            raise PlanValidationError("Duplicate task_id detected in research plan.")

        for task in tasks:
            for dep in task.dependencies:
                if dep == task.task_id:
                    raise PlanValidationError(
                        f"Task '{task.task_id}' cannot depend on itself."
                    )
                if dep not in by_id:
                    raise PlanValidationError(
                        f"Task '{task.task_id}' references unknown dependency '{dep}'."
                    )

        # 3-color DFS cycle check (0=unvisited, 1=visiting, 2=visited)
        color: dict[str, int] = {t.task_id: 0 for t in tasks}

        def dfs(node_id: str) -> None:
            color[node_id] = 1
            for dep_id in by_id[node_id].dependencies:
                state = color[dep_id]
                if state == 1:
                    raise PlanValidationError(
                        f"Cyclic task dependency detected involving '{node_id}' -> '{dep_id}'."
                    )
                if state == 0:
                    dfs(dep_id)
            color[node_id] = 2

        for t in tasks:
            if color[t.task_id] == 0:
                dfs(t.task_id)

    @staticmethod
    def select_next_batch(
        session: ResearchSession,
        *,
        max_tasks: int = 2,
    ) -> list[ResearchTask]:
        """Mark tasks whose dependencies finished as READY and pick highest-priority batch."""
        if not session.research_plan:
            return []

        max_depth = session.execution_budget.max_research_depth
        task_map = {t.task_id: t for t in session.research_plan.tasks}
        ready: list[ResearchTask] = []

        for task in session.research_plan.tasks:
            if task.status not in {TaskStatus.PENDING, TaskStatus.READY}:
                continue
            if task.depth > max_depth:
                task.status = TaskStatus.SKIPPED
                task.result_summary = (
                    f"Skipped: task depth {task.depth} exceeds max_research_depth={max_depth}."
                )
                task.updated_at = utc_now()
                continue

            deps = [task_map.get(d) for d in task.dependencies]
            if any(d is None or d.status in {TaskStatus.FAILED, TaskStatus.CANCELLED} for d in deps):
                task.status = TaskStatus.SKIPPED
                task.result_summary = "Skipped because an upstream dependency failed or was cancelled."
                task.updated_at = utc_now()
                continue

            if all(d is not None and d.status in {TaskStatus.COMPLETED, TaskStatus.SKIPPED} for d in deps):
                task.status = TaskStatus.READY
                task.updated_at = utc_now()
                ready.append(task)

        ready.sort(key=lambda t: (-t.priority, t.depth, t.created_at))
        ResearchSessionStateManager.sync_task_buckets(session)
        return ready[: max(1, max_tasks)]

    @staticmethod
    def merge_followup_tasks(
        session: ResearchSession,
        candidates: list[ResearchTask],
    ) -> tuple[list[ResearchTask], list[tuple[ResearchTask, ResearchTask]]]:
        """Append non-duplicate follow-up tasks to the plan while blocking duplicate work."""
        if not session.research_plan:
            return [], []

        existing = list(session.research_plan.tasks)
        existing_ids = {t.task_id for t in existing}
        added: list[ResearchTask] = []
        skipped: list[tuple[ResearchTask, ResearchTask]] = []

        for cand in candidates:
            if not cand.fingerprint:
                cand.fingerprint = compute_task_fingerprint(cand.question, cand.entities_keywords)
            dup = is_duplicate_task(cand, existing)
            if dup is not None:
                skipped.append((cand, dup))
                continue
            # Ensure unique task_id if collision occurs
            tid = cand.task_id
            suffix = 2
            while tid in existing_ids:
                tid = f"{cand.task_id}-{suffix}"
                suffix += 1
            cand.task_id = tid
            cand.dependencies = [d for d in cand.dependencies if d in existing_ids]
            existing_ids.add(tid)
            existing.append(cand)
            added.append(cand)

        if added:
            session.research_plan.tasks = existing
            session.research_plan.version += 1
            session.research_plan.updated_at = utc_now()
            TaskManager.validate_dag(session.research_plan.tasks)
            ResearchSessionStateManager.sync_task_buckets(session)

        return added, skipped

    @staticmethod
    def apply_user_plan_edit(
        session: ResearchSession,
        edit: AgentPlanEditRequest,
        *,
        valid_tool_names: set[str],
    ) -> ResearchPlanSpec:
        """Apply user-modified task list safely without mutating completed or running tasks."""
        if session.research_plan is None:
            raise PlanValidationError("Session has no initialized research plan yet.")

        old_by_id = {t.task_id: t for t in session.research_plan.tasks}
        locked_tasks = [
            t
            for t in session.research_plan.tasks
            if t.status in {TaskStatus.COMPLETED, TaskStatus.RUNNING, TaskStatus.WAITING_APPROVAL}
        ]
        locked_ids = {t.task_id for t in locked_tasks}

        updated_tasks: list[ResearchTask] = list(locked_tasks)
        for item in edit.tasks:
            if item.task_id in locked_ids:
                # Preserve immutable completed/in-flight task execution state
                continue
            existing = old_by_id.get(item.task_id)
            clean_tools = [t for t in item.suggested_tools if t in valid_tool_names]
            if not clean_tools:
                clean_tools = ["web_search", "url_retrieve"]
            task = ResearchTask(
                task_id=item.task_id,
                question=item.question.strip(),
                objective=item.objective.strip(),
                expected_evidence=item.expected_evidence.strip(),
                entities_keywords=[k.strip() for k in item.entities_keywords if k.strip()],
                suggested_tools=clean_tools,
                priority=item.priority,
                dependencies=list(item.dependencies),
                completion_criteria=item.completion_criteria.strip(),
                depth=existing.depth if existing else 0,
                iteration_created=existing.iteration_created if existing else session.current_iteration,
                parent_gap_id=existing.parent_gap_id if existing else None,
                origin=existing.origin if existing else "user_edit",
                status=TaskStatus.PENDING,
                fingerprint=compute_task_fingerprint(item.question, item.entities_keywords),
            )
            updated_tasks.append(task)

        TaskManager.validate_dag(updated_tasks)
        session.research_plan.tasks = updated_tasks
        session.research_plan.version += 1
        session.research_plan.updated_at = utc_now()
        ResearchSessionStateManager.sync_task_buckets(session)
        return session.research_plan
