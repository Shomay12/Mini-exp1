"""Model-independent structured ContextPackage builder and local model adapter converter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from web_intelligence.app.context.budget_manager import DynamicContextBudgetManager
from web_intelligence.app.context.context_compressor import ContextCompressor
from web_intelligence.app.context.evidence_selector import EvidenceSelector
from web_intelligence.app.context.history_manager import ConversationHistoryManager
from web_intelligence.app.context.models import (
    AllocationProfile,
    CompressionResult,
    CompressionStrategy,
    ContextBudgetConfig,
    ContextPackage,
    ContextSectionTokenUsage,
    ConversationTurn,
    ExcludedEvidenceRecord,
    ExpectedOutputLength,
    HistoryPackage,
    ModelContextCapabilities,
    SelectedEvidenceRecord,
    SourceReferenceEntry,
    WorkspaceNote,
)
from web_intelligence.app.context.token_estimator import (
    BaseContextTokenEstimator,
    ConservativeHeuristicEstimator,
)
from web_intelligence.app.schemas.documents import SourceRecord
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CoverageAnalysis,
    EvidenceItem,
    ExtractedClaim,
)
from web_intelligence.app.schemas.llm import ChatMessage, LLMInferenceRequest
from web_intelligence.app.schemas.research import ResearchPackage

DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are a local research synthesis assistant. Ground every statement in the provided "
    "[E#] evidence citations. Preserve exact numbers, dates, and conflicting source positions "
    "without inventing claims or extrapolating unsupported facts."
)


class StructuredContextBuilder:
    """Builds model-independent ContextPackages from research runs, history, and policies."""

    def __init__(
        self,
        *,
        budget_manager: DynamicContextBudgetManager | None = None,
        selector: EvidenceSelector | None = None,
        compressor: ContextCompressor | None = None,
        history_manager: ConversationHistoryManager | None = None,
    ) -> None:
        self._budget_manager = budget_manager or DynamicContextBudgetManager()
        self._selector = selector
        self._compressor = compressor
        self._history_manager = history_manager or ConversationHistoryManager()

    def build(
        self,
        package: ResearchPackage,
        capabilities: ModelContextCapabilities,
        budget_config: ContextBudgetConfig | None = None,
        *,
        profile: AllocationProfile = AllocationProfile.STANDARD_RESEARCH,
        compression_strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE,
        turns: list[ConversationTurn] | None = None,
        workspace_notes: list[WorkspaceNote] | None = None,
        estimator: BaseContextTokenEstimator | None = None,
        system_instructions: str | None = None,
        research_objective: str | None = None,
        expected_output_length: ExpectedOutputLength | None = None,
    ) -> ContextPackage:
        """Convenience synchronous build wrapper accepting an optional profile or budget config."""
        resolved_budget = budget_config or ContextBudgetConfig(
            profile=profile,
            compression_strategy=compression_strategy,
        )
        return self.build_from_research_package(
            package,
            capabilities,
            resolved_budget,
            turns=turns,
            workspace_notes=workspace_notes,
            estimator=estimator,
            system_instructions=system_instructions,
            research_objective=research_objective,
            expected_output_length=expected_output_length,
        )

    def build_from_research_package(
        self,
        package: ResearchPackage,
        capabilities: ModelContextCapabilities,
        budget_config: ContextBudgetConfig,
        *,
        turns: list[ConversationTurn] | None = None,
        workspace_notes: list[WorkspaceNote] | None = None,
        estimator: BaseContextTokenEstimator | None = None,
        system_instructions: str | None = None,
        research_objective: str | None = None,
        expected_output_length: ExpectedOutputLength | None = None,
    ) -> ContextPackage:
        """Synchronous build helper for deterministic pipelines and unit tests."""
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.build_async(
                    question=package.original_query,
                    research_id=package.research_id,
                    research_objective=research_objective,
                    system_instructions=system_instructions,
                    evidence_items=package.evidence,
                    sources=package.sources,
                    claims=package.claims,
                    conflicts=package.conflicts,
                    coverage=package.coverage,
                    capabilities=capabilities,
                    budget_config=budget_config,
                    turns=turns,
                    workspace_notes=workspace_notes,
                    estimator=estimator,
                    expected_output_length=expected_output_length,
                )
            )
        raise RuntimeError(
            "build_from_research_package() cannot be awaited inside an active event loop; use build_async()."
        )

    async def build_async(
        self,
        *,
        question: str,
        capabilities: ModelContextCapabilities,
        budget_config: ContextBudgetConfig,
        research_id: str | None = None,
        research_objective: str | None = None,
        system_instructions: str | None = None,
        evidence_items: list[EvidenceItem] | None = None,
        sources: list[SourceRecord] | None = None,
        claims: list[ExtractedClaim] | None = None,
        conflicts: list[ClaimConflict] | None = None,
        coverage: CoverageAnalysis | None = None,
        turns: list[ConversationTurn] | None = None,
        workspace_notes: list[WorkspaceNote] | None = None,
        estimator: BaseContextTokenEstimator | None = None,
        adapter: Any = None,
        expected_output_length: ExpectedOutputLength | None = None,
    ) -> ContextPackage:
        """Assemble a complete, inspectable, model-independent ContextPackage."""
        counter = estimator or ConservativeHeuristicEstimator()
        clean_question = question.strip()
        clean_system = (
            system_instructions.strip()
            if system_instructions is not None
            else DEFAULT_SYSTEM_INSTRUCTIONS
        )
        clean_objective = (
            research_objective.strip()
            if research_objective and research_objective.strip()
            else f"Answer the research question accurately with [E#] source citations: {clean_question}"
        )

        ev_list = list(evidence_items or [])
        claim_list = list(claims or [])
        conflict_list = list(conflicts or [])
        cov = coverage or CoverageAnalysis()
        turn_list = list(turns or [])
        note_list = list(workspace_notes or [])
        sources_by_id = {s.source_id: s for s in (sources or [])}

        # Estimate candidate supplies cleanly for adaptive allocation
        ev_supply_tokens = sum(
            counter.estimate(e.passage_text).tokens + 14 for e in ev_list
        )
        hist_supply_tokens = sum(
            counter.estimate(t.content).tokens + 8 for t in turn_list
        ) + sum(
            counter.estimate(f"{n.title}: {n.content}").tokens + 8
            for n in note_list
            if n.enabled
        )
        meta_supply_tokens = (
            counter.estimate(self._render_coverage_block(cov)).tokens
            + len(conflict_list) * 48
            + len(ev_list) * 20
        )

        prompt_query_text = f"Question: {clean_question}\nObjective: {clean_objective}"
        allocation = self._budget_manager.plan_budget(
            capabilities,
            budget_config,
            system_text=clean_system,
            query_text=prompt_query_text,
            available_evidence_tokens=ev_supply_tokens,
            available_history_tokens=hist_supply_tokens,
            available_metadata_tokens=meta_supply_tokens,
            expected_output_length=expected_output_length,
            estimator=counter,
        )

        # Build bounded history + workspace notes
        history_pkg = self._history_manager.build_history_package(
            turns=turn_list,
            workspace_notes=note_list,
            history_budget_tokens=allocation.conversation_history_tokens,
            max_history_turns=budget_config.max_history_turns,
            estimator=counter,
        )

        # Select evidence candidates (expand candidate window modestly only when compression applies)
        selector = self._selector or EvidenceSelector(estimator=counter)
        comp_strategy = budget_config.compression_strategy
        candidate_multiplier = (
            1.0
            if comp_strategy == CompressionStrategy.NONE
            else 2.5
        )
        # If there is a single very large passage and compression is enabled, let compressor distill it
        effective_selection_budget = allocation.research_evidence_tokens
        if comp_strategy != CompressionStrategy.NONE and ev_list:
            max_single = max(
                selector._estimate_item_tokens(item) for item in ev_list
            )
            if (
                max_single > allocation.research_evidence_tokens * candidate_multiplier
                and allocation.research_evidence_tokens >= 128
            ):
                effective_selection_budget = max_single + 16

        selected_records, excluded_records = selector.select(
            ev_list,
            query=clean_question,
            conflicts=conflict_list,
            coverage=cov,
            evidence_budget_tokens=effective_selection_budget,
            budget_multiplier=candidate_multiplier if effective_selection_budget == allocation.research_evidence_tokens else 1.0,
        )

        compressor = self._compressor or ContextCompressor(estimator=counter)
        compression_result = await compressor.compress_async(
            selected_records,
            strategy=comp_strategy,
            query=clean_question,
            target_tokens=allocation.research_evidence_tokens,
            claims=claim_list,
            conflicts=conflict_list,
            adapter=adapter,
            allow_model_assisted=budget_config.allow_model_assisted_compression,
        )

        selected_ids = {rec.evidence_id for rec in selected_records}
        active_claims = [c for c in claim_list if c.evidence_id in selected_ids]

        # Build canonical source reference table matching selected citation labels
        ev_by_id = {e.evidence_id: e for e in ev_list}
        source_refs: list[SourceReferenceEntry] = []
        for rec in selected_records:
            orig_ev = ev_by_id.get(rec.evidence_id)
            src_doc = sources_by_id.get(rec.source_id)
            canonical = (
                orig_ev.provenance.canonical_url
                if orig_ev is not None
                else (src_doc.canonical_url if src_doc else rec.source_url)
            )
            classification = (
                orig_ev.provenance.source_classification.value
                if orig_ev is not None
                else "unknown"
            )
            published_at = (
                orig_ev.provenance.published_at if orig_ev is not None else None
            )
            source_refs.append(
                SourceReferenceEntry(
                    citation_label=rec.citation_label,
                    source_id=rec.source_id,
                    evidence_id=rec.evidence_id,
                    url=rec.source_url,
                    canonical_url=canonical,
                    domain=rec.source_domain,
                    title=rec.source_title,
                    classification=classification,
                    published_at=published_at,
                )
            )

        rendered_evidence = self._render_evidence_block(compression_result)
        rendered_history = self._render_history_block(history_pkg)
        rendered_conflicts = self._render_conflicts_block(conflict_list, selected_records)
        rendered_coverage = self._render_coverage_block(cov)
        rendered_refs = self._render_source_references_block(source_refs)

        prompt_sections: list[str] = [
            f"## Question\n{clean_question}",
            f"## Research Objective\n{clean_objective}",
        ]
        if rendered_history:
            prompt_sections.append(f"## Conversation History & Notes\n{rendered_history}")
        if rendered_evidence:
            prompt_sections.append(f"## Selected Evidence ({len(selected_records)} passages)\n{rendered_evidence}")
        else:
            prompt_sections.append("## Selected Evidence\nNo evidence passages selected for active context.")
        if rendered_conflicts:
            prompt_sections.append(f"## Conflicting Claims\n{rendered_conflicts}")
        if rendered_coverage:
            prompt_sections.append(f"## Research Coverage\n{rendered_coverage}")
        if rendered_refs:
            prompt_sections.append(f"## Source References\n{rendered_refs}")

        rendered_user_prompt = "\n\n".join(prompt_sections)

        # Measure per-section token usage accurately using the active estimator
        section_usage = self._measure_sections(
            counter=counter,
            allocation=allocation,
            system_text=clean_system,
            question_text=clean_question,
            objective_text=clean_objective,
            history_text=rendered_history,
            evidence_text=rendered_evidence,
            conflicts_text=rendered_conflicts,
            refs_text=rendered_refs,
            coverage_text=rendered_coverage,
        )

        adapter_msgs = ContextAdapterConverter.to_messages_from_parts(
            clean_system, rendered_user_prompt, capabilities=capabilities
        )
        total_input_tokens = counter.estimate_messages(adapter_msgs).tokens
        total_with_output = total_input_tokens + allocation.output_reservation_tokens

        return ContextPackage(
            context_id=f"ctx-{uuid4().hex[:12]}",
            research_id=research_id,
            created_at=datetime.now(timezone.utc),
            system_instructions=clean_system,
            original_user_question=clean_question,
            research_objective=clean_objective,
            selected_evidence=selected_records,
            excluded_evidence=excluded_records,
            compression=compression_result,
            supporting_claims=active_claims,
            conflicting_claims=conflict_list,
            source_references=source_refs,
            history=history_pkg,
            research_coverage=cov,
            capabilities=capabilities,
            budget_allocation=allocation,
            section_token_usage=section_usage,
            total_input_tokens=total_input_tokens,
            output_reservation_tokens=allocation.output_reservation_tokens,
            total_with_output_tokens=total_with_output,
            estimation_is_approximate=not counter.tokenizer_available,
            estimation_strategy=counter.strategy_name,
            rendered_evidence_block=rendered_evidence,
            rendered_user_prompt=rendered_user_prompt,
        )

    @staticmethod
    def _render_evidence_block(compression: CompressionResult) -> str:
        if not compression.units:
            return ""
        return "\n".join(unit.compressed_text for unit in compression.units)

    @staticmethod
    def _render_history_block(history: HistoryPackage) -> str:
        parts: list[str] = []
        if history.workspace_notes:
            note_lines = [f"- {n.title}: {n.content}" for n in history.workspace_notes]
            parts.append("Workspace Notes:\n" + "\n".join(note_lines))
        if history.compacted_summary is not None:
            parts.append("Earlier Session Summary:\n" + history.compacted_summary.summary_text)
        if history.session_turns:
            turn_lines = [f"[{t.role}] {t.content}" for t in history.session_turns]
            parts.append("Recent Turns:\n" + "\n".join(turn_lines))
        if history.current_turn_items:
            curr_lines = [f"[current {t.role}] {t.content}" for t in history.current_turn_items]
            parts.append("Current Turn:\n" + "\n".join(curr_lines))
        return "\n\n".join(parts)

    @staticmethod
    def _render_conflicts_block(
        conflicts: list[ClaimConflict],
        selected: list[SelectedEvidenceRecord],
    ) -> str:
        if not conflicts:
            return ""
        cite_by_ev = {rec.evidence_id: rec.citation_label for rec in selected}
        lines: list[str] = []
        for cf in conflicts:
            ca = cite_by_ev.get(cf.claim_a.evidence_id, cf.claim_a.source_domain)
            cb = cite_by_ev.get(cf.claim_b.evidence_id, cf.claim_b.source_domain)
            lines.append(
                f"- [{cf.conflict_type}] {cf.entity}.{cf.attribute}: {ca} ({cf.claim_a.normalized_value}) "
                f"vs {cb} ({cf.claim_b.normalized_value})"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_coverage_block(coverage: CoverageAnalysis) -> str:
        lines: list[str] = [f"Status: {coverage.coverage_status.value}"]
        if coverage.covered_topics:
            lines.append("Covered: " + ", ".join(coverage.covered_topics[:8]))
        if coverage.missing_topics:
            lines.append("Missing: " + ", ".join(coverage.missing_topics[:8]))
        if coverage.conflicted_areas:
            lines.append("Conflicted: " + ", ".join(coverage.conflicted_areas[:6]))
        return "\n".join(lines)

    @staticmethod
    def _render_source_references_block(refs: list[SourceReferenceEntry]) -> str:
        return "\n".join(
            f"{ref.citation_label} {ref.url} ({ref.domain}; {ref.classification})"
            for ref in refs
        )

    @staticmethod
    def _measure_sections(
        *,
        counter: BaseContextTokenEstimator,
        allocation: Any,
        system_text: str,
        question_text: str,
        objective_text: str,
        history_text: str,
        evidence_text: str,
        conflicts_text: str,
        refs_text: str,
        coverage_text: str,
    ) -> list[ContextSectionTokenUsage]:
        approx = not counter.tokenizer_available
        conflicts_tok = counter.estimate(conflicts_text).tokens
        refs_tok = counter.estimate(refs_text).tokens
        coverage_tok = counter.estimate(coverage_text).tokens
        meta_budget = allocation.structured_metadata_tokens
        meta_split = max(1, meta_budget // 3)
        return [
            ContextSectionTokenUsage(
                section="system_instructions",
                estimated_tokens=counter.estimate(system_text).tokens,
                allocated_tokens=allocation.system_instructions_tokens,
                is_approximate=approx,
            ),
            ContextSectionTokenUsage(
                section="user_query",
                estimated_tokens=counter.estimate(question_text).tokens,
                allocated_tokens=allocation.user_query_tokens,
                is_approximate=approx,
            ),
            ContextSectionTokenUsage(
                section="research_objective",
                estimated_tokens=counter.estimate(objective_text).tokens,
                allocated_tokens=max(16, allocation.user_query_tokens // 2),
                is_approximate=approx,
            ),
            ContextSectionTokenUsage(
                section="conversation_history",
                estimated_tokens=counter.estimate(history_text).tokens,
                allocated_tokens=allocation.conversation_history_tokens,
                is_approximate=approx,
            ),
            ContextSectionTokenUsage(
                section="selected_evidence",
                estimated_tokens=counter.estimate(evidence_text).tokens,
                allocated_tokens=allocation.research_evidence_tokens,
                is_approximate=approx,
            ),
            ContextSectionTokenUsage(
                section="conflicting_claims",
                estimated_tokens=conflicts_tok,
                allocated_tokens=meta_split,
                is_approximate=approx,
            ),
            ContextSectionTokenUsage(
                section="source_references",
                estimated_tokens=refs_tok,
                allocated_tokens=meta_split,
                is_approximate=approx,
            ),
            ContextSectionTokenUsage(
                section="research_coverage",
                estimated_tokens=coverage_tok,
                allocated_tokens=max(0, meta_budget - 2 * meta_split),
                is_approximate=approx,
            ),
        ]


class ContextAdapterConverter:
    """Clean conversion layer translating model-independent ContextPackages for local adapters."""

    @staticmethod
    def to_messages_from_parts(
        system_instructions: str,
        rendered_user_prompt: str,
        *,
        capabilities: ModelContextCapabilities | None = None,
        prefer_json: bool = False,
    ) -> list[ChatMessage]:
        """Format chat messages cleanly honoring model metadata (system-role and JSON support)."""
        sys_text = system_instructions.strip()
        if prefer_json and capabilities and capabilities.supports_structured_output:
            sys_text = f"{sys_text}\nReturn a valid JSON object matching the requested schema."

        supports_system = True
        if capabilities and capabilities.raw_metadata:
            supports_system = bool(capabilities.raw_metadata.get("supports_system_role", True))

        if supports_system:
            return [
                ChatMessage(role="system", content=sys_text),
                ChatMessage(role="user", content=rendered_user_prompt),
            ]
        return [
            ChatMessage(role="user", content=f"{sys_text}\n\n{rendered_user_prompt}"),
        ]

    @classmethod
    def to_messages(
        cls,
        package: ContextPackage,
        *,
        prefer_json: bool = False,
    ) -> list[ChatMessage]:
        """Convert ContextPackage to ChatMessage list for chat-capable local adapters."""
        return cls.to_messages_from_parts(
            package.system_instructions,
            package.rendered_user_prompt,
            capabilities=package.capabilities,
            prefer_json=prefer_json,
        )

    @staticmethod
    def to_prompt_string(package: ContextPackage) -> str:
        """Convert ContextPackage to a flat prompt string for generate()-only adapters."""
        return f"### System\n{package.system_instructions}\n\n{package.rendered_user_prompt}"

    @classmethod
    def to_inference_request(
        cls,
        package: ContextPackage,
        *,
        temperature: float = 0.1,
        stream: bool = False,
        prefer_json: bool = False,
    ) -> LLMInferenceRequest:
        """Build a ready-to-execute LLMInferenceRequest honoring reserved output budget."""
        return LLMInferenceRequest(
            model=package.capabilities.model_name or "local-model",
            messages=cls.to_messages(package, prefer_json=prefer_json),
            temperature=temperature,
            max_tokens=package.output_reservation_tokens,
            stream=stream,
        )
