"""AgentResearchPlanner decomposing objectives and follow-up gaps into validated task DAGs."""

from __future__ import annotations

import uuid
from typing import Any

from web_intelligence.app.agent.execution_budget import ExecutionBudgetTracker
from web_intelligence.app.agent.models import (
    GapAnalysisReport,
    ResearchDepth,
    ResearchGap,
    ResearchPlanSpec,
    ResearchSession,
    ResearchTask,
    TaskStatus,
)
from web_intelligence.app.agent.research_state import (
    compute_task_fingerprint,
    is_duplicate_task,
)
from web_intelligence.app.agent.task_manager import PlanValidationError, TaskManager
from web_intelligence.app.agent.tool_registry import FORBIDDEN_TOOL_NAMES, ToolRegistry
from web_intelligence.app.llm.base import UniversalLocalModelAdapter
from web_intelligence.app.query.decomposition import decompose_query
from web_intelligence.app.query.processor import extract_keywords, normalize_query
from web_intelligence.app.schemas.llm import ChatMessage, LLMInferenceRequest


class AgentResearchPlanner:
    """Produces structured ResearchPlans via connected local LLM + schema validation or deterministic rules."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._registry = tool_registry

    async def create_initial_plan(
        self,
        *,
        session: ResearchSession,
        budget: ExecutionBudgetTracker,
        adapter: UniversalLocalModelAdapter | None = None,
    ) -> tuple[ResearchPlanSpec, str]:
        """Generate and validate the initial structured research DAG for `session`."""
        permitted = self._permitted_tool_names(session)
        planning_notes: list[str] = []

        if (
            adapter is not None
            and session.use_local_llm
            and budget.config.max_inference_calls > 0
            and budget.usage.inference_calls < budget.config.max_inference_calls
        ):
            llm_plan, llm_err = await self._plan_with_llm(
                session=session,
                adapter=adapter,
                permitted_tools=permitted,
                budget=budget,
            )
            if llm_plan is not None:
                return llm_plan, "Initial research plan generated and schema-validated via local LLM."
            if llm_err:
                planning_notes.append(llm_err)

        det_plan = self._build_deterministic_plan(session, permitted_tools=permitted)
        note = "; ".join(planning_notes) if planning_notes else (
            "Initial research plan created using deterministic multi-facet task decomposition."
        )
        det_plan.planning_notes = note
        return det_plan, note

    async def create_followup_tasks(
        self,
        *,
        session: ResearchSession,
        gaps: GapAnalysisReport,
        budget: ExecutionBudgetTracker,
        adapter: UniversalLocalModelAdapter | None = None,
    ) -> tuple[list[ResearchTask], str]:
        """Generate bounded follow-up tasks directly justified by unresolved gaps."""
        max_new = budget.config.max_followups_per_iteration
        if max_new <= 0 or not gaps.gaps:
            return [], "No follow-up tasks requested or follow-up limit is 0."

        permitted = self._permitted_tool_names(session)
        existing_tasks = list(session.research_plan.tasks) if session.research_plan else []
        max_depth = budget.config.max_research_depth
        completed_ids = [
            t.task_id for t in existing_tasks if t.status == TaskStatus.COMPLETED
        ]

        candidates: list[ResearchTask] = []
        prioritized_gaps = sorted(
            gaps.gaps,
            key=lambda g: (-g.priority, 0 if g.severity == "high" else 1),
        )

        # Calculate parent depth cleanly
        depth_by_task = {t.task_id: t.depth for t in existing_tasks}
        next_idx = len(existing_tasks) + 1

        for gap in prioritized_gaps:
            if len(candidates) >= max_new:
                break
            if gap.addressed_by_task_id:
                existing_addr = next(
                    (t for t in existing_tasks if t.task_id == gap.addressed_by_task_id), None
                )
                if existing_addr and existing_addr.status != TaskStatus.FAILED:
                    continue

            parent_depth = depth_by_task.get(gap.related_task_id or "", 0)
            target_depth = min(max_depth, parent_depth + 1)
            if target_depth > max_depth or (parent_depth >= max_depth and max_depth == 0):
                continue

            deps: list[str] = []
            if gap.related_task_id and any(t.task_id == gap.related_task_id for t in existing_tasks):
                deps = [gap.related_task_id]
            elif completed_ids:
                deps = []

            tools = [t for t in gap.suggested_tools if t in permitted]
            if not tools:
                tools = [t for t in ("web_search", "url_retrieve", "evidence_search") if t in permitted]

            candidate = ResearchTask(
                task_id=f"task-{next_idx}",
                question=gap.suggested_followup_question.strip()[:580],
                objective=gap.suggested_objective.strip()[:780],
                expected_evidence=gap.expected_evidence,
                entities_keywords=extract_keywords(
                    gap.suggested_followup_question, max_keywords=8
                ),
                suggested_tools=tools,
                priority=gap.priority,
                dependencies=deps,
                completion_criteria=(
                    f"Resolve {gap.category.value} gap ({gap.gap_id}) with verified passages."
                ),
                depth=target_depth,
                iteration_created=session.current_iteration,
                parent_gap_id=gap.gap_id,
                origin="gap_followup",
                fingerprint=compute_task_fingerprint(
                    gap.suggested_followup_question, []
                ),
            )
            if is_duplicate_task(candidate, [*existing_tasks, *candidates]) is not None:
                continue

            gap.addressed_by_task_id = candidate.task_id
            candidates.append(candidate)
            next_idx += 1

        # Optional LLM query phrasing refinement when connected & budget permits
        note = (
            f"Generated {len(candidates)} targeted follow-up task(s) from "
            f"{len(gaps.gaps)} unresolved gap(s)."
        )
        return candidates, note

    async def _plan_with_llm(
        self,
        *,
        session: ResearchSession,
        adapter: UniversalLocalModelAdapter,
        permitted_tools: set[str],
        budget: ExecutionBudgetTracker,
    ) -> tuple[ResearchPlanSpec | None, str | None]:
        """Prompt local LLM for strict JSON plan with 1 validation-recovery retry before deterministic fallback."""
        target_tasks = self._target_task_count(session.research_depth)
        tools_list = ", ".join(sorted(permitted_tools))
        base_prompt = (
            f"Research Objective: {session.research_objective}\n"
            f"Original Query: {session.original_query}\n"
            f"Depth: {session.research_depth.value} (produce {target_tasks} bounded, non-overlapping tasks)\n"
            f"Allowed tools ONLY: {tools_list}\n\n"
            "Decompose this objective into concrete, investigable research tasks with explicit "
            "completion criteria and optional DAG dependencies on prior task_ids (task-1, task-2...). "
            'Respond ONLY with JSON: {"planning_notes": "brief rationale", "tasks": [{"task_id": "task-1", '
            '"question": "...", "objective": "...", "expected_evidence": "...", '
            '"entities_keywords": ["kw1"], "suggested_tools": ["web_search", "url_retrieve"], '
            '"priority": 5, "dependencies": [], "completion_criteria": "..."}]}'
        )

        correction_error: str | None = None
        for attempt in range(2):
            if budget.usage.inference_calls >= budget.config.max_inference_calls:
                return None, "Inference budget reached before LLM planning."
            prompt = base_prompt if correction_error is None else (
                f"{base_prompt}\n\nYour previous JSON plan failed schema validation: "
                f"{correction_error}\nFix the JSON structure and obey all tool/DAG rules."
            )
            req = LLMInferenceRequest(
                messages=[
                    ChatMessage(
                        role="system",
                        content=(
                            "You are a structured research planning engine. Output strictly valid JSON. "
                            "Never reference shell, python, browser, or code execution tools."
                        ),
                    ),
                    ChatMessage(role="user", content=prompt),
                ],
                model=session.active_model_id or None,
                temperature=0.1,
                max_tokens=900,
                response_format="json_object" if adapter.capabilities().supports_json_mode else "text",
            )
            try:
                budget.record_planner_or_synthesis_inference(estimated_tokens=350)
                raw = await adapter.generate_json(req)
                plan = self._validate_llm_plan_payload(
                    raw=raw,
                    session=session,
                    permitted_tools=permitted_tools,
                    model_id=session.active_model_id,
                )
                return plan, None
            except Exception as exc:
                correction_error = str(exc)

        return None, f"LLM plan validation failed twice ({correction_error}); using deterministic plan."

    def _validate_llm_plan_payload(
        self,
        *,
        raw: Any,
        session: ResearchSession,
        permitted_tools: set[str],
        model_id: str | None,
    ) -> ResearchPlanSpec:
        if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), list) or not raw["tasks"]:
            raise PlanValidationError("LLM response must be a JSON object with a non-empty 'tasks' list.")

        raw_tasks = raw["tasks"][:8]
        tasks: list[ResearchTask] = []
        seen_fps: list[ResearchTask] = []

        for idx, item in enumerate(raw_tasks, start=1):
            if not isinstance(item, dict):
                raise PlanValidationError(f"Task entry #{idx} is not a JSON object.")
            tid = str(item.get("task_id") or f"task-{idx}").strip()
            question = str(item.get("question") or "").strip()
            objective = str(item.get("objective") or question).strip()
            if len(question) < 3:
                raise PlanValidationError(f"Task '{tid}' has an empty or too-short question.")

            raw_tools = item.get("suggested_tools") or []
            if not isinstance(raw_tools, list):
                raw_tools = []
            for bad in raw_tools:
                if str(bad).strip().lower() in FORBIDDEN_TOOL_NAMES:
                    raise PlanValidationError(f"LLM plan attempted forbidden tool '{bad}'.")
            clean_tools = [
                str(t).strip().lower()
                for t in raw_tools
                if str(t).strip().lower() in permitted_tools
            ]
            if not clean_tools:
                clean_tools = [t for t in ("web_search", "url_retrieve") if t in permitted_tools] or sorted(permitted_tools)[:1]

            kws = [
                str(k).strip()
                for k in (item.get("entities_keywords") or [])
                if str(k).strip()
            ][:12]
            if not kws:
                kws = extract_keywords(question, max_keywords=6)

            deps = [
                str(d).strip()
                for d in (item.get("dependencies") or [])
                if isinstance(d, str) and d.strip()
            ]
            priority = min(5, max(1, int(item.get("priority") or (6 - min(idx, 5)))))

            task = ResearchTask(
                task_id=tid,
                question=question[:580],
                objective=objective[:780],
                expected_evidence=str(
                    item.get("expected_evidence") or "Primary documentation or empirical metrics."
                )[:400],
                entities_keywords=kws,
                suggested_tools=clean_tools,
                priority=priority,
                dependencies=deps,
                completion_criteria=str(
                    item.get("completion_criteria")
                    or "Retain >= 1 relevant passage with explicit source provenance."
                )[:400],
                depth=0,
                iteration_created=0,
                origin="initial_plan",
                fingerprint=compute_task_fingerprint(question, kws),
            )
            if is_duplicate_task(task, seen_fps) is not None:
                continue
            seen_fps.append(task)
            tasks.append(task)

        if not tasks:
            raise PlanValidationError("No unique valid tasks parsed from LLM output.")

        TaskManager.validate_dag(tasks)
        norm_obj = normalize_query(session.research_objective)
        return ResearchPlanSpec(
            plan_id=f"plan-{uuid.uuid4().hex[:10]}",
            version=1,
            objective=session.research_objective,
            normalized_objective=norm_obj,
            planned_by="local_llm_planner",
            model_id=model_id,
            planning_notes=str(raw.get("planning_notes") or "Structured LLM task plan.")[:500],
            tasks=tasks,
        )

    def _build_deterministic_plan(
        self,
        session: ResearchSession,
        *,
        permitted_tools: set[str],
    ) -> ResearchPlanSpec:
        """Build structured multi-task research plan deterministically when LLM is unavailable."""
        norm, _intent, _topics, base_kw, subtopics, _planned = decompose_query(
            session.original_query
        )
        target_n = self._target_task_count(session.research_depth)
        base_kw = base_kw[:8]
        default_tools = [
            t for t in ("local_doc_retrieve", "web_search", "url_retrieve") if t in permitted_tools
        ] or sorted(permitted_tools)[:1]
        core_subject = " ".join(base_kw[:5]) or session.original_query

        facets: list[tuple[str, str, list[str], int, list[str]]] = [
            (
                f"What primary definitions, core specifications, and architectures apply to: {session.original_query}",
                "Establish foundational facts, named systems, and primary technical capabilities.",
                base_kw[:6],
                5,
                [],
            ),
            (
                f"What empirical benchmarks, resource requirements, and performance tradeoffs exist for {core_subject}?",
                "Gather quantitative measurements, hardware/RAM metrics, latency, and empirical evaluations.",
                list(dict.fromkeys([*base_kw[:4], "benchmark", "memory", "performance"])),
                4,
                [],
            ),
            (
                f"What limitations, reliability risks, privacy boundaries, or conflicting findings affect {core_subject}?",
                "Identify failure modes, adversarial or boundary risks, and contradictory evaluations.",
                list(dict.fromkeys([*base_kw[:4], "privacy", "limitations", "conflict"])),
                4,
                [],
            ),
        ]
        # Add subtopics from query decomposition when helpful for broad/deep objectives
        for subtopic in subtopics:
            if len(facets) >= target_n:
                break
            sub_kw = extract_keywords(subtopic, max_keywords=6)
            facets.append(
                (
                    f"Investigate facet '{subtopic}' in relation to {session.original_query}",
                    f"Resolve subtopic '{subtopic}' with independent sources.",
                    list(dict.fromkeys([*base_kw[:3], *sub_kw])),
                    3,
                    [],
                )
            )
        if session.research_depth != ResearchDepth.QUICK and len(facets) < target_n:
            facets.append(
                (
                    f"Cross-verify conflicting or single-source empirical claims regarding {core_subject}",
                    "Reconcile prior findings across independent primary/reference sources before final synthesis.",
                    list(dict.fromkeys([*base_kw[:4], "verification", "comparison"])),
                    2,
                    ["task-1", "task-2"],
                )
            )

        tasks: list[ResearchTask] = []
        for idx, (question, objective, kws, prio, deps) in enumerate(facets[:target_n], start=1):
            tid = f"task-{idx}"
            tasks.append(
                ResearchTask(
                    task_id=tid,
                    question=question[:580],
                    objective=objective[:780],
                    expected_evidence="Retained passages and extracted verifiable claims with provenance.",
                    entities_keywords=kws[:10],
                    suggested_tools=default_tools,
                    priority=prio,
                    dependencies=[d for d in deps if any(t.task_id == d for t in tasks)],
                    completion_criteria="At least one relevant retrieved passage or corroborated claim.",
                    depth=0,
                    iteration_created=0,
                    origin="initial_plan",
                    fingerprint=compute_task_fingerprint(question, kws),
                )
            )

        TaskManager.validate_dag(tasks)
        return ResearchPlanSpec(
            plan_id=f"plan-{uuid.uuid4().hex[:10]}",
            version=1,
            objective=session.research_objective,
            normalized_objective=normalize_query(session.research_objective) or norm,
            planned_by="deterministic_planner",
            model_id=None,
            planning_notes="Structured deterministic research decomposition.",
            tasks=tasks,
        )

    def _permitted_tool_names(self, session: ResearchSession) -> set[str]:
        return {
            d.name
            for d in self._registry.list_tools()
            if self._registry.is_tool_available(d.name, session)[0]
        }

    @staticmethod
    def _target_task_count(depth: ResearchDepth) -> int:
        if depth == ResearchDepth.QUICK:
            return 2
        if depth == ResearchDepth.DEEP:
            return 5
        return 3
