"""Multi-stage LongFormReportGenerator with per-section Stage 5 context budgeting and cross-section checks."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any

from web_intelligence.app.context.budget_manager import DynamicContextBudgetManager
from web_intelligence.app.context.context_builder import StructuredContextBuilder
from web_intelligence.app.context.models import (
    AllocationProfile,
    ContextBudgetConfig,
    ExpectedOutputLength,
    ModelContextCapabilities,
)
from web_intelligence.app.context.token_estimator import resolve_token_estimator
from web_intelligence.app.generation.citations import (
    CitationValidator,
    merge_citations_preserving_order,
)
from web_intelligence.app.generation.executor import (
    InferenceCancelledError,
    InferenceExecutor,
    ResolvedInferenceTarget,
)
from web_intelligence.app.generation.prompt_compiler import PromptCompiler
from web_intelligence.app.schemas.evidence import EvidenceItem
from web_intelligence.app.schemas.generation import (
    CitationEntry,
    GenerationFormat,
    GenerationLength,
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    LongFormReportConfig,
    LongFormReportResult,
    LongFormSectionPlan,
    LongFormSectionResult,
    PromptTemplateType,
)
from web_intelligence.app.schemas.llm import TokenUsage
from web_intelligence.app.schemas.research import ResearchPackage

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2}


def _aggregate_usage(usages: list[TokenUsage | None]) -> TokenUsage | None:
    prompt_vals = [u.prompt_tokens for u in usages if u and u.prompt_tokens is not None]
    comp_vals = [u.completion_tokens for u in usages if u and u.completion_tokens is not None]
    total_vals = [u.total_tokens for u in usages if u and u.total_tokens is not None]
    if not prompt_vals and not comp_vals and not total_vals:
        return None
    p_sum = sum(prompt_vals) if prompt_vals else None
    c_sum = sum(comp_vals) if comp_vals else None
    if total_vals and len(total_vals) == len([u for u in usages if u is not None]):
        t_sum: int | None = sum(total_vals)
    elif p_sum is not None and c_sum is not None:
        t_sum = p_sum + c_sum
    else:
        t_sum = sum(total_vals) if total_vals else None
    return TokenUsage(prompt_tokens=p_sum, completion_tokens=c_sum, total_tokens=t_sum)


def _detect_repetition(sections: list[LongFormSectionResult]) -> list[str]:
    """Identify duplicated factual sentences or high shingle overlap across generated sections."""
    warnings: list[str] = []
    seen_sentences: dict[str, str] = {}
    shingles_by_section: dict[str, set[tuple[str, ...]]] = {}

    for sec in sections:
        content = sec.content.strip()
        if not content:
            continue
        for raw_sent in _SENTENCE_SPLIT_RE.split(content):
            norm = " ".join(_WORD_RE.findall(raw_sent.lower()))
            if len(norm) >= 42:
                prev_title = seen_sentences.get(norm)
                if prev_title and prev_title != sec.title:
                    warnings.append(
                        f"Repeated sentence detected between '{prev_title}' and '{sec.title}'."
                    )
                else:
                    seen_sentences[norm] = sec.title

        words = _WORD_RE.findall(content.lower())
        if len(words) >= 8:
            shingles_by_section[sec.title] = {
                tuple(words[i : i + 4]) for i in range(len(words) - 3)
            }

    titles = list(shingles_by_section.keys())
    for idx in range(len(titles)):
        for jdx in range(idx + 1, len(titles)):
            a, b = titles[idx], titles[jdx]
            sa, sb = shingles_by_section[a], shingles_by_section[b]
            if not sa or not sb:
                continue
            overlap = len(sa & sb) / max(1, min(len(sa), len(sb)))
            if overlap >= 0.45:
                warnings.append(
                    f"High content overlap ({int(overlap * 100)}%) between '{a}' and '{b}'."
                )
    return list(dict.fromkeys(warnings))


def _detect_cross_section_contradictions(
    sections: list[LongFormSectionResult],
    package: ResearchPackage,
) -> list[str]:
    """Check whether known source conflicts remain unacknowledged or numeric claims diverge."""
    warnings: list[str] = []
    combined_lower = "\n".join(s.content.lower() for s in sections)
    for conf in package.conflicts:
        if conf.entity.lower() not in combined_lower:
            warnings.append(
                f"Detected conflict about '{conf.entity} / {conf.attribute}' "
                f"({conf.claim_a.normalized_value} vs {conf.claim_b.normalized_value}) may be under-covered."
            )

    # Check if two sections mention conflicting normalized values for same known conflict entity
    for conf in package.conflicts:
        val_a = conf.claim_a.normalized_value.lower()
        val_b = conf.claim_b.normalized_value.lower()
        hits_a = [s.title for s in sections if val_a and val_a in s.content.lower()]
        hits_b = [s.title for s in sections if val_b and val_b in s.content.lower()]
        if hits_a and hits_b and set(hits_a) != set(hits_b):
            warnings.append(
                f"Cross-section divergence on '{conf.entity}': '{conf.claim_a.normalized_value}' in "
                f"{hits_a[0]} vs '{conf.claim_b.normalized_value}' in {hits_b[0]}."
            )
    return list(dict.fromkeys(warnings))


class LongFormReportGenerator:
    """Executes structured 10-step multi-stage report planning, per-section allocation, and synthesis."""

    def __init__(
        self,
        *,
        executor: InferenceExecutor,
        prompt_compiler: PromptCompiler | None = None,
        citation_validator: CitationValidator | None = None,
        budget_manager: DynamicContextBudgetManager | None = None,
    ) -> None:
        self._executor = executor
        self._compiler = prompt_compiler or PromptCompiler()
        self._validator = citation_validator or CitationValidator()
        self._budget_manager = budget_manager or DynamicContextBudgetManager()

    def build_outline(
        self,
        package: ResearchPackage,
        config: LongFormReportConfig,
        global_citation_map: dict[str, CitationEntry],
    ) -> list[LongFormSectionPlan]:
        """Plan structured report outline and assign evidence + [S#] subsets per section."""
        source_to_cid = {entry.source_id: cid for cid, entry in global_citation_map.items()}
        all_ev = list(package.evidence)
        plans: list[LongFormSectionPlan] = []
        order = 1

        # Reserve slots cleanly inside max_sections
        reserved_tail = (
            1
            if config.include_contradiction_section
            and (package.conflicts or package.coverage.missing_topics or package.coverage.partial_topics)
            else 0
        )
        reserved_head = 1 if config.include_executive_summary else 0
        body_slots = max(1, config.max_sections - reserved_head - reserved_tail)

        if config.include_executive_summary:
            top_ev = all_ev[: min(4, len(all_ev))]
            plans.append(
                LongFormSectionPlan(
                    section_id=f"sec-{order}",
                    order=order,
                    title="Executive Summary",
                    objective=(
                        f"Synthesize high-level findings directly answering '{package.original_query}' "
                        "and highlight primary evidence limits."
                    ),
                    assigned_subtopics=[],
                    assigned_evidence_ids=[e.evidence_id for e in top_ev],
                    assigned_citation_ids=[
                        source_to_cid[e.source_id]
                        for e in top_ev
                        if e.source_id in source_to_cid
                    ],
                )
            )
            order += 1

        # Determine topical buckets from Stage 4 subtopic coverage & evidence headings
        topic_candidates: list[str] = []
        for detail in package.coverage.subtopic_details:
            if detail.subtopic and detail.subtopic not in topic_candidates:
                topic_candidates.append(detail.subtopic)
        for topic in [*package.coverage.covered_topics, *package.coverage.partial_topics]:
            if topic and topic not in topic_candidates:
                topic_candidates.append(topic)
        for ev in all_ev:
            if ev.section_heading and ev.section_heading not in topic_candidates:
                topic_candidates.append(ev.section_heading)

        if not topic_candidates:
            topic_candidates = ["Core Evidence Analysis"]
        selected_topics = topic_candidates[:body_slots]

        subtopic_detail_map = {d.subtopic: d for d in package.coverage.subtopic_details}
        assigned_global: set[str] = set()
        for topic in selected_topics:
            detail = subtopic_detail_map.get(topic)
            ev_ids: list[str] = []
            if detail and detail.supporting_evidence_ids:
                ev_ids = [eid for eid in detail.supporting_evidence_ids if eid]
            if not ev_ids:
                t_kw = _keywords(topic)
                scored = [
                    (
                        len(t_kw & _keywords(f"{e.section_heading or ''} {e.passage_text}")),
                        e.relevance.combined_score,
                        e.evidence_id,
                    )
                    for e in all_ev
                ]
                scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
                ev_ids = [row[2] for row in scored[: min(5, len(scored))] if row[0] > 0]
            if not ev_ids:
                unassigned = [e.evidence_id for e in all_ev if e.evidence_id not in assigned_global]
                ev_ids = (unassigned or [e.evidence_id for e in all_ev])[: min(4, len(all_ev))]

            assigned_global.update(ev_ids)
            ev_objs = [e for e in all_ev if e.evidence_id in set(ev_ids)]
            cids = list(
                dict.fromkeys(
                    source_to_cid[e.source_id] for e in ev_objs if e.source_id in source_to_cid
                )
            )
            plans.append(
                LongFormSectionPlan(
                    section_id=f"sec-{order}",
                    order=order,
                    title=topic,
                    objective=(
                        f"Examine retrieved evidence regarding '{topic}' with exact metrics, "
                        "dates, and explicit attribution."
                    ),
                    assigned_subtopics=[topic],
                    assigned_evidence_ids=ev_ids,
                    assigned_citation_ids=cids,
                )
            )
            order += 1

        if reserved_tail:
            conflict_ev_ids = list(
                dict.fromkeys(
                    cid
                    for conf in package.conflicts
                    for cid in (conf.claim_a.evidence_id, conf.claim_b.evidence_id)
                )
            )
            if not conflict_ev_ids:
                conflict_ev_ids = [e.evidence_id for e in all_ev[: min(4, len(all_ev))]]
            cids = list(
                dict.fromkeys(
                    source_to_cid[e.source_id]
                    for e in all_ev
                    if e.evidence_id in set(conflict_ev_ids) and e.source_id in source_to_cid
                )
            )
            plans.append(
                LongFormSectionPlan(
                    section_id=f"sec-{order}",
                    order=order,
                    title="Evidence Conflicts, Limitations & Unanswered Questions",
                    objective=(
                        "Document source disagreements transparently without picking an unverified winner, "
                        "and enumerate missing or partially covered research gaps."
                    ),
                    assigned_subtopics=list(package.coverage.missing_topics),
                    assigned_evidence_ids=conflict_ev_ids,
                    assigned_citation_ids=cids,
                )
            )

        return plans

    async def _build_section_context(
        self,
        *,
        package: ResearchPackage,
        plan: LongFormSectionPlan,
        capabilities: ModelContextCapabilities,
        section_output_budget: int,
        adapter: Any,
    ):
        ev_by_id = {e.evidence_id: e for e in package.evidence}
        section_evidence: list[EvidenceItem] = [
            ev_by_id[eid] for eid in plan.assigned_evidence_ids if eid in ev_by_id
        ]
        if not section_evidence and package.evidence:
            section_evidence = package.evidence[: min(4, len(package.evidence))]

        seen_ev_ids = {e.evidence_id for e in section_evidence}
        src_ids = {e.source_id for e in section_evidence}
        candidate_conflicts = [
            c
            for c in package.conflicts
            if c.claim_a.evidence_id in seen_ev_ids
            or c.claim_b.evidence_id in seen_ev_ids
            or c.claim_a.source_id in src_ids
            or c.claim_b.source_id in src_ids
        ]
        # Pair opposing evidence items together so Stage 5 never flags PARTIAL_CONFLICT_PAIR.
        for c in candidate_conflicts:
            for eid in (c.claim_a.evidence_id, c.claim_b.evidence_id):
                if eid not in seen_ev_ids and eid in ev_by_id:
                    section_evidence.append(ev_by_id[eid])
                    seen_ev_ids.add(eid)
        section_conflicts = [
            c
            for c in candidate_conflicts
            if c.claim_a.evidence_id in seen_ev_ids and c.claim_b.evidence_id in seen_ev_ids
        ]
        src_ids = {e.source_id for e in section_evidence}
        for c in section_conflicts:
            src_ids.add(c.claim_a.source_id)
            src_ids.add(c.claim_b.source_id)

        section_sources = [s for s in package.sources if s.source_id in src_ids]
        section_claims = [c for c in package.claims if c.source_id in src_ids]

        estimator = resolve_token_estimator(adapter)
        builder = StructuredContextBuilder(budget_manager=self._budget_manager)
        budget_cfg = ContextBudgetConfig(
            profile=AllocationProfile.DEEP_RESEARCH,
            output_reservation_tokens=section_output_budget,
            expected_output_length=ExpectedOutputLength.MEDIUM,
        )
        return await builder.build_async(
            question=package.original_query,
            research_id=package.research_id,
            research_objective=f"{plan.title} — {plan.objective}",
            evidence_items=section_evidence,
            sources=section_sources,
            claims=section_claims,
            conflicts=section_conflicts,
            coverage=package.coverage,
            capabilities=capabilities,
            budget_config=budget_cfg,
            estimator=estimator,
            adapter=adapter,
            expected_output_length=ExpectedOutputLength.MEDIUM,
        )

    async def stream_generate(
        self,
        *,
        generation_id: str,
        request: GenerationRequest,
        package: ResearchPackage,
        target: ResolvedInferenceTarget,
        capabilities: ModelContextCapabilities,
        global_citation_map: dict[str, CitationEntry],
        cancel_event: asyncio.Event | None = None,
        is_disconnected: Callable[[], Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute multi-stage long-form report yielding progress/chunk events and final GenerationResult."""
        started = time.monotonic()
        config = request.long_form_config or LongFormReportConfig()
        outline = self.build_outline(package, config, global_citation_map)
        section_out_budget = (
            config.section_max_output_tokens
            or request.max_output_tokens
            or max(192, min(capabilities.max_output_tokens, capabilities.effective_context_limit // 4))
        )
        # Ensure output reservation leaves safe input headroom per section pass
        section_out_budget = min(section_out_budget, max(128, capabilities.effective_context_limit // 2))

        yield {
            "event": "outline",
            "generation_id": generation_id,
            "outline": [p.model_dump(mode="json") for p in outline],
        }

        section_results: list[LongFormSectionResult] = []
        usages: list[TokenUsage | None] = []
        warnings: list[str] = []
        total_estimated_prompt_tokens = 0
        was_cancelled = False

        for plan in outline:
            if cancel_event is not None and cancel_event.is_set():
                was_cancelled = True
                break

            sec_started = time.monotonic()
            sec_ctx = await self._build_section_context(
                package=package,
                plan=plan,
                capabilities=capabilities,
                section_output_budget=section_out_budget,
                adapter=target.adapter,
            )
            # Re-bind compiled citation map to the unified global [S#] source IDs when present!
            compiled = self._compiler.compile(
                sec_ctx,
                template_type=PromptTemplateType.RESEARCH_REPORT,
                output_format=(
                    GenerationFormat.MARKDOWN
                    if request.output_format == GenerationFormat.JSON
                    else request.output_format
                ),
                answer_length=GenerationLength.MEDIUM,
                require_citations=request.require_citations,
                reserved_output_tokens=section_out_budget,
                custom_instruction=request.custom_instruction,
                section_focus=(
                    f"Section {plan.order}/{len(outline)}: '{plan.title}'. {plan.objective} "
                    "Write ONLY this section body (without repeating other sections)."
                ),
                token_estimator=resolve_token_estimator(target.adapter),
            )
            # Remap local [S#] in prompt back to consistent global [S#] IDs cleanly!
            source_to_global = {e.source_id: cid for cid, e in global_citation_map.items()}
            local_to_global: dict[str, str] = {}
            remapped_user_prompt = compiled.user_prompt
            for local_cid, entry in compiled.citation_map.items():
                g_cid = source_to_global.get(entry.source_id, local_cid)
                local_to_global[local_cid] = g_cid
            if any(k != v for k, v in local_to_global.items()):
                remapped_user_prompt = re.sub(
                    r"\[(S\d+)\]",
                    lambda m: f"[{local_to_global.get(m.group(1), m.group(1))}]",
                    compiled.user_prompt,
                )
                remapped_user_prompt = re.sub(
                    r'id="(S\d+)"',
                    lambda m: f'id="{local_to_global.get(m.group(1), m.group(1))}"',
                    remapped_user_prompt,
                )
                compiled = compiled.model_copy(
                    update={
                        "user_prompt": remapped_user_prompt,
                        "messages": [
                            compiled.messages[0],
                            compiled.messages[1].model_copy(update={"content": remapped_user_prompt}),
                        ],
                        "citation_map": global_citation_map,
                    }
                )
            else:
                compiled = compiled.model_copy(update={"citation_map": global_citation_map})

            total_estimated_prompt_tokens += compiled.estimated_prompt_tokens
            yield {
                "event": "section_start",
                "generation_id": generation_id,
                "section_id": plan.section_id,
                "order": plan.order,
                "title": plan.title,
            }
            yield {
                "event": "delta",
                "generation_id": generation_id,
                "delta": f"## {plan.title}\n\n",
                "section_id": plan.section_id,
            }

            sec_content = ""
            sec_finish: str | None = None
            sec_usage: TokenUsage | None = None
            sec_warnings = list(compiled.warnings)
            sec_status = GenerationStatus.COMPLETED

            try:
                async for ev in self._executor.execute_stream(
                    target=target,
                    context_package=sec_ctx,
                    compiled_prompt=compiled,
                    temperature=request.temperature,
                    max_output_tokens=section_out_budget,
                    stop_sequences=request.stop_sequences,
                    timeout_seconds=request.timeout_seconds,
                    cancel_event=cancel_event,
                    is_disconnected=is_disconnected,
                ):
                    if ev.kind == "warning" and ev.warning:
                        sec_warnings.append(ev.warning)
                    elif ev.kind == "delta":
                        sec_content = ev.accumulated
                        yield {
                            "event": "delta",
                            "generation_id": generation_id,
                            "delta": ev.delta,
                            "section_id": plan.section_id,
                        }
                    elif ev.kind == "done":
                        sec_content = ev.accumulated
                        sec_finish = ev.finish_reason
                        sec_usage = ev.usage
            except InferenceCancelledError as exc:
                sec_content = exc.partial_content or sec_content
                sec_usage = exc.usage
                sec_status = GenerationStatus.CANCELLED
                was_cancelled = True

            yield {
                "event": "delta",
                "generation_id": generation_id,
                "delta": "\n\n",
                "section_id": plan.section_id,
            }

            truncated = sec_finish == "length"
            if truncated:
                sec_warnings.append(
                    f"Section '{plan.title}' reached max_output_tokens ({section_out_budget}) and was truncated."
                )

            sec_citations, sec_val = self._validator.validate(
                sec_content,
                global_citation_map,
                stage5_to_citation_id=compiled.stage5_to_citation_id,
                require_citations=request.require_citations,
            )
            usages.append(sec_usage)
            warnings.extend(sec_warnings)
            sec_res = LongFormSectionResult(
                section_id=plan.section_id,
                order=plan.order,
                title=plan.title,
                content=sec_content.strip(),
                status=sec_status,
                finish_reason=sec_finish,
                truncated=truncated,
                citations=sec_citations,
                citation_validation=sec_val,
                usage=sec_usage,
                allocated_input_tokens=sec_ctx.total_input_tokens,
                duration_ms=max(0, int((time.monotonic() - sec_started) * 1000)),
                warnings=sec_warnings,
            )
            section_results.append(sec_res)
            yield {
                "event": "section_complete",
                "generation_id": generation_id,
                "section": sec_res.model_dump(mode="json"),
            }
            if was_cancelled:
                break

        full_markdown = "\n\n".join(
            f"## {sec.title}\n\n{sec.content}".strip() for sec in section_results
        )
        repetition_warnings = (
            _detect_repetition(section_results) if config.detect_repetition else []
        )
        contradiction_warnings = (
            _detect_cross_section_contradictions(section_results, package)
            if config.detect_contradictions
            else []
        )
        all_warnings = list(dict.fromkeys([*warnings, *repetition_warnings, *contradiction_warnings]))

        all_citations, full_val = self._validator.validate(
            full_markdown,
            global_citation_map,
            require_citations=request.require_citations,
        )
        merged_citations, source_map = merge_citations_preserving_order(
            [all_citations]
        )
        any_truncated = any(s.truncated for s in section_results)
        final_status = (
            GenerationStatus.CANCELLED
            if was_cancelled
            else GenerationStatus.COMPLETED
        )

        result = GenerationResult(
            generation_id=generation_id,
            research_id=package.research_id,
            status=final_status,
            partial=was_cancelled,
            mode=GenerationMode.LONG_FORM_REPORT,
            template_type=request.template_type,
            output_format=GenerationFormat.MARKDOWN,
            connection_id=target.connection_id,
            provider_id=target.provider_id,
            server_endpoint=target.server_endpoint,
            model_id=target.model_id,
            content=full_markdown,
            finish_reason="cancelled" if was_cancelled else ("length" if any_truncated else "stop"),
            truncated=any_truncated,
            citations=merged_citations,
            source_map=source_map,
            citation_validation=full_val,
            usage=_aggregate_usage(usages),
            estimated_prompt_tokens=total_estimated_prompt_tokens,
            long_form=LongFormReportResult(
                objective=package.original_query,
                outline=outline,
                sections=section_results,
                repetition_warnings=repetition_warnings,
                contradiction_warnings=contradiction_warnings,
                synthesis_notes=(
                    f"Completed {len(section_results)}/{len(outline)} sections "
                    f"({sum(len(s.citations) for s in section_results)} section source refs)."
                ),
            ),
            warnings=all_warnings,
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            completed_at=datetime.now(timezone.utc),
        )
        yield {"event": "result", "result": result}
