"""Adaptive dynamic context budget allocator tailored to connected local LLM capabilities."""

from __future__ import annotations

from web_intelligence.app.research.future_interfaces import (
    HeuristicTokenEstimator,
    TokenEstimator,
)
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CoverageAnalysis,
    EvidenceItem,
)
from web_intelligence.app.schemas.llm import ModelCapabilities
from web_intelligence.app.schemas.research import ContextAllocationPlan

_CARD_WRAPPER_OVERHEAD_TOKENS = 28
_PROMPT_FRAMING_OVERHEAD_TOKENS = 96


class AdaptiveContextBudgetManager:
    """Packs ranked evidence within model context limits while protecting diversity and conflicts."""

    def __init__(self, token_estimator: TokenEstimator | None = None) -> None:
        self._estimator = token_estimator or HeuristicTokenEstimator()

    def _estimate_item_tokens(self, item: EvidenceItem) -> int:
        base = (
            item.token_estimate
            if item.token_estimate > 0
            else self._estimator.estimate_tokens(item.passage_text)
        )
        return base + _CARD_WRAPPER_OVERHEAD_TOKENS

    @staticmethod
    def _conflict_evidence_ids(
        evidence_items: list[EvidenceItem], conflicts: list[ClaimConflict]
    ) -> set[str]:
        if not conflicts:
            return set()
        conflict_source_ids = {
            c.claim_a.source_id for c in conflicts
        } | {c.claim_b.source_id for c in conflicts}
        conflict_snippets = {
            c.claim_a.statement_text[:80] for c in conflicts if c.claim_a.statement_text
        } | {c.claim_b.statement_text[:80] for c in conflicts if c.claim_b.statement_text}

        matched: set[str] = set()
        for item in evidence_items:
            if item.corroboration.conflicting_claim_ids:
                matched.add(item.evidence_id)
                continue
            if item.source_id in conflict_source_ids and any(
                snippet and snippet in item.passage_text for snippet in conflict_snippets
            ):
                matched.add(item.evidence_id)
        return matched

    def allocate(
        self,
        evidence_items: list[EvidenceItem],
        capabilities: ModelCapabilities,
        *,
        system_prompt: str = "",
        query_prompt: str = "",
        conflicts: list[ClaimConflict] | None = None,
    ) -> tuple[list[EvidenceItem], ContextAllocationPlan]:
        """Select evidence items balancing score, domain diversity, and conflict coverage."""
        conflicts = conflicts or []
        max_ctx = max(512, capabilities.max_context_tokens)
        reserved_out = min(
            max(64, capabilities.recommended_output_tokens),
            max(128, max_ctx // 3),
        )
        reserved_sys = (
            self._estimator.estimate_tokens(system_prompt)
            + self._estimator.estimate_tokens(query_prompt)
            + _PROMPT_FRAMING_OVERHEAD_TOKENS
        )
        available_evidence = max(0, max_ctx - reserved_out - reserved_sys)

        ranked = sorted(
            evidence_items,
            key=lambda item: (item.relevance.combined_score, item.relevance.bm25_score),
            reverse=True,
        )
        conflict_ids = self._conflict_evidence_ids(ranked, conflicts)

        selected_ids: list[str] = []
        selected_set: set[str] = set()
        covered_domains: list[str] = []
        domain_set: set[str] = set()
        used_tokens = 0

        def try_pick(item: EvidenceItem) -> bool:
            nonlocal used_tokens
            if item.evidence_id in selected_set:
                return False
            cost = self._estimate_item_tokens(item)
            if used_tokens + cost > available_evidence:
                return False
            selected_set.add(item.evidence_id)
            selected_ids.append(item.evidence_id)
            used_tokens += cost
            if item.source_domain and item.source_domain not in domain_set:
                domain_set.add(item.source_domain)
                covered_domains.append(item.source_domain)
            return True

        # Pass 1: Prioritize conflicting evidence so downstream synthesis sees both sides.
        conflict_included = 0
        for item in ranked:
            if item.evidence_id in conflict_ids and try_pick(item):
                conflict_included += 1

        # Pass 2: Coverage balance — pick highest-ranked passage from each distinct domain first.
        for item in ranked:
            if item.source_domain not in domain_set:
                try_pick(item)

        # Pass 3: Greedy fill remaining context budget by relevance score.
        for item in ranked:
            try_pick(item)

        # Preserve display order in original ranked priority for selected items
        selected_items = [item for item in ranked if item.evidence_id in selected_set]
        # Edge guard for very small context windows: fit truncated first item if space >= 96 tokens
        if not selected_items and ranked and available_evidence >= 96:
            first = ranked[0]
            max_chars = max(120, (available_evidence - _CARD_WRAPPER_OVERHEAD_TOKENS) * 4)
            clipped_text = first.passage_text[:max_chars].rsplit(" ", 1)[0].strip() or first.passage_text[:max_chars]
            clipped_tokens = min(
                available_evidence,
                self._estimator.estimate_tokens(clipped_text) + _CARD_WRAPPER_OVERHEAD_TOKENS,
            )
            clipped_item = first.model_copy(
                update={
                    "passage_text": f"{clipped_text}...",
                    "token_estimate": max(1, clipped_tokens - _CARD_WRAPPER_OVERHEAD_TOKENS),
                }
            )
            selected_items = [clipped_item]
            selected_set = {first.evidence_id}
            used_tokens = clipped_tokens
            if first.source_domain:
                covered_domains = [first.source_domain]
            if first.evidence_id in conflict_ids:
                conflict_included = 1

        dropped_ids = [
            item.evidence_id for item in ranked if item.evidence_id not in selected_set
        ]

        plan = ContextAllocationPlan(
            model_name=capabilities.model_name or "local-model",
            max_context_tokens=max_ctx,
            reserved_system_tokens=reserved_sys,
            reserved_output_tokens=reserved_out,
            available_evidence_tokens=available_evidence,
            allocated_tokens=used_tokens,
            selected_evidence_ids=[item.evidence_id for item in selected_items],
            dropped_evidence_ids=dropped_ids,
            included_conflict_count=conflict_included,
            covered_domains=covered_domains,
        )
        return selected_items, plan

    @staticmethod
    def format_evidence_context(
        selected_items: list[EvidenceItem],
        conflicts: list[ClaimConflict],
        coverage: CoverageAnalysis,
    ) -> tuple[str, dict[str, EvidenceItem]]:
        """Render numbered citation cards [E1]..[En] mapped to EvidenceItems."""
        ref_map: dict[str, EvidenceItem] = {}
        blocks: list[str] = []

        for idx, item in enumerate(selected_items, start=1):
            tag = f"E{idx}"
            ref_map[tag] = item
            heading = f" | section={item.section_heading}" if item.section_heading else ""
            title = item.source_title or item.source_domain or "Untitled"
            blocks.append(
                f"[{tag}] (source_id={item.source_id} | domain={item.source_domain} | "
                f"title={title}{heading} | score={item.relevance.combined_score:.2f})\n"
                f"{item.passage_text}"
            )

        if conflicts:
            conflict_lines = ["DETECTED SOURCE CONFLICTS:"]
            for idx, conf in enumerate(conflicts[:5], start=1):
                conflict_lines.append(
                    f"[C{idx}] ({conf.conflict_type}) {conf.summary} | "
                    f"Claim A ({conf.claim_a.source_domain}): {conf.claim_a.statement_text} || "
                    f"Claim B ({conf.claim_b.source_domain}): {conf.claim_b.statement_text}"
                )
            blocks.append("\n".join(conflict_lines))

        if coverage.unanswered_questions or coverage.missing_topics:
            gap_lines = list(coverage.unanswered_questions[:4])
            for topic in coverage.missing_topics[:3]:
                gap_lines.append(f"Missing coverage for topic: {topic}")
            blocks.append("COVERAGE GAPS:\n- " + "\n- ".join(gap_lines))

        return "\n\n".join(blocks), ref_map
