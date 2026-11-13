"""Hybrid LLM-assisted and deterministic ToolSelector validated against ToolRegistry."""

from __future__ import annotations

import uuid
from typing import Any

from web_intelligence.app.agent.models import (
    ResearchSession,
    ResearchTask,
    ToolDescriptor,
    ToolProposal,
)
from web_intelligence.app.agent.research_memory import ResearchWorkingMemory
from web_intelligence.app.agent.tool_registry import FORBIDDEN_TOOL_NAMES, ToolRegistry
from web_intelligence.app.llm.base import UniversalLocalModelAdapter
from web_intelligence.app.query.decomposition import classify_intent
from web_intelligence.app.schemas.llm import ChatMessage, LLMInferenceRequest


class ToolSelector:
    """Chooses safe, schema-validated tool actions for an eligible ResearchTask."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        memory: ResearchWorkingMemory | None = None,
    ) -> None:
        self._registry = registry
        self._memory = memory or ResearchWorkingMemory()

    def build_search_query(self, task: ResearchTask, session: ResearchSession) -> str:
        """Construct a concise, non-private search query tailored to the task."""
        kw = [k.strip() for k in task.entities_keywords if k.strip() and len(k.strip()) >= 2]
        base = task.question.strip().rstrip("?")
        if kw:
            missing_kw = [k for k in kw[:4] if k.lower() not in base.lower()]
            if missing_kw:
                base = f"{base} {' '.join(missing_kw[:3])}"
        # Ensure we avoid reusing an exact previously executed query on follow-ups
        executed = self._memory.executed_search_queries(session)
        sanitized, _ = self._memory.sanitize_external_query(base)
        candidate = sanitized or task.question.strip()
        if candidate.lower() in executed and task.entities_keywords:
            refined = f"{candidate} {' '.join(task.entities_keywords[:3])} benchmark evaluation"
            candidate, _ = self._memory.sanitize_external_query(refined)
        return (candidate or session.original_query)[:240]

    async def select_tools_for_task(
        self,
        task: ResearchTask,
        session: ResearchSession,
        *,
        adapter: UniversalLocalModelAdapter | None = None,
        max_results_per_query: int = 5,
    ) -> tuple[list[ToolProposal], str]:
        """Select validated tool calls for `task`, preferring safe LLM suggestions + deterministic fallback."""
        available = [
            d for d in self._registry.list_tools() if self._registry.is_tool_available(d.name, session)[0]
        ]
        available_names = {d.name for d in available}
        rejected_notes: list[str] = []
        proposals: list[ToolProposal] = []
        selector_mode = "deterministic_selector"

        # 1. Optionally ask connected local LLM which permitted tools fit the task
        llm_suggested_tools: list[dict[str, Any]] = []
        if (
            adapter is not None
            and session.use_local_llm
            and session.execution_budget.max_inference_calls > 0
        ):
            llm_suggested_tools, err = await self._ask_llm_for_tools(
                adapter, task, session, available
            )
            if err:
                rejected_notes.append(err)

        for raw in llm_suggested_tools[:3]:
            raw_name = str(raw.get("tool_name", "")).strip().lower()
            raw_args = raw.get("arguments")
            if raw_name in FORBIDDEN_TOOL_NAMES:
                rejected_notes.append(f"Rejected prohibited LLM tool recommendation '{raw_name}'.")
                continue
            if raw_name not in available_names:
                rejected_notes.append(f"Rejected unavailable/unregistered LLM tool '{raw_name}'.")
                continue
            if not isinstance(raw_args, dict):
                raw_args = {}
            normalized_args = self._normalize_tool_arguments(
                raw_name, raw_args, task, session, max_results_per_query=max_results_per_query
            )
            try:
                self._registry.validate_arguments(raw_name, normalized_args)
            except Exception as exc:
                rejected_notes.append(f"Rejected invalid LLM arguments for '{raw_name}': {exc}")
                continue
            desc = self._registry.get(raw_name)
            proposals.append(
                ToolProposal(
                    proposal_id=f"prop-{uuid.uuid4().hex[:8]}",
                    task_id=task.task_id,
                    tool_name=raw_name,
                    arguments=normalized_args,
                    rationale=str(raw.get("rationale") or f"LLM recommended '{raw_name}' for task objective."),
                    selected_by="local_llm_validated",
                    requires_approval=bool(
                        desc and desc.requires_external_network and session.require_user_approval_for_external
                    ),
                )
            )
            selector_mode = "local_llm_validated"

        # 2. Deterministic baseline guarantees actionable, budgeted progression
        if not proposals:
            if "local_doc_retrieve" in available_names and session.current_iteration <= 1:
                proposals.append(
                    ToolProposal(
                        proposal_id=f"prop-{uuid.uuid4().hex[:8]}",
                        task_id=task.task_id,
                        tool_name="local_doc_retrieve",
                        arguments={"query": self.build_search_query(task, session), "limit": 4},
                        rationale="Check locally cached research sources before issuing external requests.",
                        selected_by="deterministic_selector",
                    )
                )

            # Check existing session evidence first when follow-up can also use web_search
            if "evidence_search" in available_names and session.evidence_records:
                proposals.append(
                    ToolProposal(
                        proposal_id=f"prop-{uuid.uuid4().hex[:8]}",
                        task_id=task.task_id,
                        tool_name="evidence_search",
                        arguments={"query": task.question, "top_k": 6},
                        rationale="Inspect already-collected session evidence for relevant passages before additional search.",
                        selected_by="deterministic_selector",
                    )
                )

            prefer_search = any(t in task.suggested_tools for t in ("web_search", "url_retrieve"))
            if prefer_search and "web_search" in available_names:
                q = self.build_search_query(task, session)
                desc = self._registry.get("web_search")
                intent_val = classify_intent(q).value
                proposals.append(
                    ToolProposal(
                        proposal_id=f"prop-{uuid.uuid4().hex[:8]}",
                        task_id=task.task_id,
                        tool_name="web_search",
                        arguments={
                            "query": q,
                            "max_results": max_results_per_query,
                            "providers": session.providers,
                            "intent": intent_val,
                        },
                        rationale=(
                            f"Search needed to satisfy task '{task.task_id}' completion criteria: "
                            f"{task.completion_criteria}"
                        ),
                        selected_by="deterministic_selector",
                        requires_approval=bool(
                            desc and desc.requires_external_network and session.require_user_approval_for_external
                        ),
                    )
                )

        # Deduplicate tool proposals by tool_name while preserving sequence
        deduped: list[ToolProposal] = []
        seen_names: set[str] = set()
        for prop in proposals:
            if prop.tool_name in seen_names:
                continue
            seen_names.add(prop.tool_name)
            deduped.append(prop)

        note = "; ".join(rejected_notes) if rejected_notes else f"Selected via {selector_mode}."
        return deduped, note

    def _normalize_tool_arguments(
        self,
        tool_name: str,
        raw_args: dict[str, Any],
        task: ResearchTask,
        session: ResearchSession,
        *,
        max_results_per_query: int,
    ) -> dict[str, Any]:
        if tool_name == "web_search":
            q = str(raw_args.get("query") or self.build_search_query(task, session))
            return {
                "query": q[:300],
                "max_results": min(20, max(1, int(raw_args.get("max_results") or max_results_per_query))),
                "providers": session.providers,
                "intent": str(raw_args.get("intent") or "exploratory")[:40],
            }
        if tool_name == "evidence_search":
            return {
                "query": str(raw_args.get("query") or task.question)[:300],
                "top_k": min(25, max(1, int(raw_args.get("top_k") or 6))),
            }
        if tool_name == "local_doc_retrieve":
            return {
                "query": str(raw_args.get("query") or task.question)[:300],
                "limit": min(20, max(1, int(raw_args.get("limit") or 4))),
            }
        if tool_name == "context_inspect":
            return {
                "question": str(raw_args.get("question") or task.question)[:500],
            }
        return raw_args

    async def _ask_llm_for_tools(
        self,
        adapter: UniversalLocalModelAdapter,
        task: ResearchTask,
        session: ResearchSession,
        available: list[ToolDescriptor],
    ) -> tuple[list[dict[str, Any]], str | None]:
        tools_summary = "\n".join(
            f"- {d.name}: {d.description}" for d in available if d.name != "local_llm_inference"
        )
        prompt = (
            f"Research task: {task.question}\n"
            f"Objective: {task.objective}\n"
            f"Expected evidence: {task.expected_evidence}\n"
            f"Current collected sources: {len(session.collected_sources)}\n\n"
            f"Available registered tools:\n{tools_summary}\n\n"
            'Return strict JSON: {"tools": [{"tool_name": "name", "arguments": {...}, "rationale": "why"}]}'
        )
        req = LLMInferenceRequest(
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "Select only from the explicitly listed registered tools. "
                        "Never propose shell, python, or code execution. Respond with JSON only."
                    ),
                ),
                ChatMessage(role="user", content=prompt),
            ],
            model=session.active_model_id or None,
            temperature=0.1,
            max_tokens=320,
            response_format="json_object" if adapter.capabilities().supports_json_mode else "text",
        )
        try:
            payload = await adapter.generate_json(req)
            if isinstance(payload, dict) and isinstance(payload.get("tools"), list):
                return [item for item in payload["tools"] if isinstance(item, dict)], None
            return [], "LLM tool proposal JSON omitted valid 'tools' array; fell back deterministically."
        except Exception as exc:
            return [], f"LLM tool selection unavailable ({exc}); used deterministic selector."
