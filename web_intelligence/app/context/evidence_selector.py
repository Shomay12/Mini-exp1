"""Deterministic multi-signal evidence selection with conflict preservation and extensible scoring."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from web_intelligence.app.context.models import (
    EvidenceScoreBreakdown,
    ExcludedEvidenceRecord,
    SelectedEvidenceRecord,
)
from web_intelligence.app.context.token_estimator import (
    BaseContextTokenEstimator,
    ConservativeHeuristicEstimator,
)
from web_intelligence.app.schemas.documents import SourceClassification
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CorroborationStatus,
    CoverageAnalysis,
    EvidenceItem,
)

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

_PROVENANCE_SCORES: dict[SourceClassification, float] = {
    SourceClassification.PRIMARY: 1.0,
    SourceClassification.REFERENCE: 0.85,
    SourceClassification.SECONDARY: 0.60,
    SourceClassification.UNKNOWN: 0.40,
}


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 0.0


@dataclass(frozen=True)
class SelectionContext:
    """Context provided to EvidenceScoringStrategy for dynamic per-candidate scoring."""

    query_tokens: set[str]
    uncovered_subtopics: dict[str, set[str]]
    conflict_evidence_ids: set[str]
    selected_domains: Counter[str]
    selected_clusters: set[str]
    evidence_budget_tokens: int
    reference_time: datetime


class EvidenceScoringStrategy(ABC):
    """Replaceable scoring interface (allows swapping baseline heuristic for embeddings/reranker)."""

    @abstractmethod
    def score(
        self,
        item: EvidenceItem,
        token_cost: int,
        context: SelectionContext,
    ) -> tuple[EvidenceScoreBreakdown, list[str]]:
        """Score an evidence item across multi-factor signals and list matched subtopics."""


class DeterministicEvidenceScoringStrategy(EvidenceScoringStrategy):
    """Baseline deterministic multi-factor scoring strategy (zero external ML runtime needed)."""

    def score(
        self,
        item: EvidenceItem,
        token_cost: int,
        context: SelectionContext,
    ) -> tuple[EvidenceScoreBreakdown, list[str]]:
        relevance = float(item.relevance.combined_score)

        item_tokens = _tokenize(
            f"{item.section_heading or ''} {item.source_title or ''} {item.passage_text}"
        )

        matched_subtopics: list[str] = []
        best_subtopic_overlap = 0.0
        for label, sub_tokens in context.uncovered_subtopics.items():
            if not sub_tokens:
                continue
            overlap = len(item_tokens & sub_tokens) / len(sub_tokens)
            if overlap >= 0.35:
                matched_subtopics.append(label)
            if overlap > best_subtopic_overlap:
                best_subtopic_overlap = overlap

        query_overlap = (
            len(item_tokens & context.query_tokens) / len(context.query_tokens)
            if context.query_tokens
            else 0.0
        )
        topic_coverage = round(min(1.0, max(best_subtopic_overlap, query_overlap)), 4)

        # Source independence: reward domains and underlying clusters not yet selected
        domain_count = context.selected_domains.get(item.source_domain, 0)
        cluster_ids = set(item.corroboration.same_underlying_source_cluster or [])
        shares_cluster = bool(cluster_ids & context.selected_clusters)
        if shares_cluster:
            source_independence = 0.25
        elif domain_count == 0:
            source_independence = 1.0
        elif domain_count == 1:
            source_independence = 0.55
        else:
            source_independence = 0.25

        provenance = _PROVENANCE_SCORES.get(item.provenance.source_classification, 0.40)

        # Claim support: reward corroborated independent multi-source items
        corr = item.corroboration
        if corr.corroboration_status == CorroborationStatus.CORROBORATED:
            claim_support = min(1.0, 0.70 + 0.10 * min(3, corr.independent_supporting_sources))
        elif corr.corroboration_status == CorroborationStatus.CONTESTED:
            claim_support = 0.65
        elif corr.corroboration_status == CorroborationStatus.SINGLE_SOURCE:
            claim_support = 0.40
        else:
            claim_support = 0.25

        # Explicit conflict priority so contested items are never swept away blindly
        is_conflict = (
            item.evidence_id in context.conflict_evidence_ids
            or bool(corr.conflicting_claim_ids)
            or corr.corroboration_status == CorroborationStatus.CONTESTED
        )
        conflict_priority = 1.0 if is_conflict else 0.0

        recency = self._compute_recency(item.provenance.published_at, context.reference_time)

        budget_denom = max(256, context.evidence_budget_tokens)
        token_cost_penalty = round(min(1.0, token_cost / budget_denom), 4)

        final_score = round(
            (0.32 * relevance)
            + (0.18 * topic_coverage)
            + (0.15 * source_independence)
            + (0.10 * provenance)
            + (0.10 * claim_support)
            + (0.15 * conflict_priority)
            + (0.06 * recency)
            - (0.08 * token_cost_penalty),
            4,
        )
        final_score = max(0.0, final_score)

        breakdown = EvidenceScoreBreakdown(
            relevance=round(relevance, 4),
            topic_coverage=topic_coverage,
            source_independence=round(source_independence, 4),
            provenance=round(provenance, 4),
            claim_support=round(claim_support, 4),
            conflict_priority=conflict_priority,
            recency=round(recency, 4),
            token_cost_penalty=token_cost_penalty,
            final_score=final_score,
        )
        return breakdown, matched_subtopics

    @staticmethod
    def _compute_recency(published_at: datetime | None, reference_time: datetime) -> float:
        if published_at is None:
            return 0.5
        pub = (
            published_at.replace(tzinfo=timezone.utc)
            if published_at.tzinfo is None
            else published_at
        )
        ref = (
            reference_time.replace(tzinfo=timezone.utc)
            if reference_time.tzinfo is None
            else reference_time
        )
        age_days = max(0.0, (ref - pub).total_seconds() / 86400.0)
        if age_days <= 30:
            return 1.0
        if age_days <= 180:
            return 0.85
        if age_days <= 365:
            return 0.70
        if age_days <= 1095:
            return 0.50
        return 0.30


class EvidenceSelector:
    """Selects diverse, non-redundant, conflict-preserving evidence within token budgets."""

    def __init__(
        self,
        scoring_strategy: EvidenceScoringStrategy | None = None,
        estimator: BaseContextTokenEstimator | None = None,
        *,
        redundancy_similarity_threshold: float = 0.78,
    ) -> None:
        self._scoring = scoring_strategy or DeterministicEvidenceScoringStrategy()
        self._estimator = estimator or ConservativeHeuristicEstimator()
        self._redundancy_threshold = redundancy_similarity_threshold

    def select(
        self,
        evidence_items: list[EvidenceItem],
        *,
        query: str = "",
        conflicts: list[ClaimConflict] | None = None,
        coverage: CoverageAnalysis | None = None,
        evidence_budget_tokens: int = 2048,
        budget_multiplier: float = 1.0,
        reference_time: datetime | None = None,
    ) -> tuple[list[SelectedEvidenceRecord], list[ExcludedEvidenceRecord]]:
        """Rank and select evidence items while preserving conflicts and source attribution."""
        if not evidence_items:
            return [], []

        effective_budget = max(0, int(evidence_budget_tokens * max(1.0, budget_multiplier)))
        ref_time = reference_time or datetime.now(timezone.utc)
        conflicts = conflicts or []
        by_id: dict[str, EvidenceItem] = {item.evidence_id: item for item in evidence_items}

        # Build conflict map & paired groups so both sides of a dispute stay together
        conflict_ids_by_ev: dict[str, list[str]] = {}
        conflict_pairs: list[tuple[ClaimConflict, EvidenceItem, EvidenceItem | None]] = []
        conflict_evidence_ids: set[str] = set()
        for conflict in conflicts:
            ev_a = conflict.claim_a.evidence_id
            ev_b = conflict.claim_b.evidence_id
            conflict_ids_by_ev.setdefault(ev_a, []).append(conflict.conflict_id)
            conflict_ids_by_ev.setdefault(ev_b, []).append(conflict.conflict_id)
            conflict_evidence_ids.add(ev_a)
            conflict_evidence_ids.add(ev_b)
            item_a = by_id.get(ev_a)
            item_b = by_id.get(ev_b)
            if item_a is not None:
                conflict_pairs.append((conflict, item_a, item_b))

        # Initial uncovered subtopic token sets from CoverageAnalysis
        uncovered_subtopics: dict[str, set[str]] = {}
        if coverage is not None:
            for sub in coverage.subtopic_details:
                sub_tokens = _tokenize(sub.subtopic)
                if sub_tokens:
                    uncovered_subtopics[sub.subtopic] = sub_tokens

        query_tokens = _tokenize(query)
        token_costs: dict[str, int] = {
            item.evidence_id: self._estimate_item_tokens(item) for item in evidence_items
        }
        passage_tokens: dict[str, set[str]] = {
            item.evidence_id: _tokenize(item.passage_text) for item in evidence_items
        }

        selected_items: list[tuple[EvidenceItem, EvidenceScoreBreakdown, list[str], str, bool]] = []
        selected_ids: set[str] = set()
        selected_domains: Counter[str] = Counter()
        selected_clusters: set[str] = set()
        excluded: list[ExcludedEvidenceRecord] = []
        excluded_ids: set[str] = set()
        used_tokens = 0

        def make_selection_ctx() -> SelectionContext:
            return SelectionContext(
                query_tokens=query_tokens,
                uncovered_subtopics=uncovered_subtopics,
                conflict_evidence_ids=conflict_evidence_ids,
                selected_domains=selected_domains,
                selected_clusters=selected_clusters,
                evidence_budget_tokens=max(256, evidence_budget_tokens),
                reference_time=ref_time,
            )

        def is_redundant(candidate: EvidenceItem, matched_subs: list[str]) -> bool:
            cand_toks = passage_tokens[candidate.evidence_id]
            if not cand_toks:
                return False
            for existing, *_ in selected_items:
                sim = _jaccard_similarity(cand_toks, passage_tokens[existing.evidence_id])
                if sim >= self._redundancy_threshold:
                    # Keep independent source corroboration only if text isn't practically verbatim
                    is_distinct_domain = candidate.source_domain != existing.source_domain
                    adds_new_subtopic = bool(matched_subs)
                    if not (is_distinct_domain and sim < 0.90) and not adds_new_subtopic:
                        return True
            return False

        def record_selection(
            item: EvidenceItem,
            breakdown: EvidenceScoreBreakdown,
            matched_subs: list[str],
            reason: str,
            is_conflict: bool,
        ) -> None:
            nonlocal used_tokens
            selected_items.append((item, breakdown, matched_subs, reason, is_conflict))
            selected_ids.add(item.evidence_id)
            selected_domains[item.source_domain] += 1
            for cl in item.corroboration.same_underlying_source_cluster:
                selected_clusters.add(cl)
            for sub in matched_subs:
                uncovered_subtopics.pop(sub, None)
            used_tokens += token_costs[item.evidence_id]

        # Phase 1: Preserve contradictory claim pairs together first.
        for conflict, item_a, item_b in conflict_pairs:
            pair = [item for item in (item_a, item_b) if item is not None and item.evidence_id not in selected_ids]
            if not pair:
                continue
            pair_cost = sum(token_costs[item.evidence_id] for item in pair)
            if used_tokens + pair_cost <= effective_budget:
                ctx = make_selection_ctx()
                for item in pair:
                    breakdown, matched_subs = self._scoring.score(item, token_costs[item.evidence_id], ctx)
                    record_selection(
                        item,
                        breakdown,
                        matched_subs,
                        f"Preserved conflict pair ({conflict.conflict_id}: {conflict.conflict_type})",
                        True,
                    )

        # Phase 2: Iterative greedy marginal selection (re-scores remaining items as topics/domains fill)
        remaining = [
            item for item in evidence_items if item.evidence_id not in selected_ids
        ]
        while remaining:
            ctx = make_selection_ctx()
            scored_candidates: list[tuple[EvidenceItem, EvidenceScoreBreakdown, list[str]]] = []
            for item in remaining:
                breakdown, matched_subs = self._scoring.score(
                    item, token_costs[item.evidence_id], ctx
                )
                scored_candidates.append((item, breakdown, matched_subs))

            # Deterministic tie-break: higher final_score, higher relevance, lexical evidence_id
            scored_candidates.sort(
                key=lambda row: (-row[1].final_score, -row[1].relevance, row[0].evidence_id)
            )

            picked_any = False
            next_remaining: list[EvidenceItem] = []
            for item, breakdown, matched_subs in scored_candidates:
                cost = token_costs[item.evidence_id]
                is_conf = item.evidence_id in conflict_evidence_ids
                if not picked_any:
                    if not is_conf and is_redundant(item, matched_subs):
                        excluded.append(
                            self._build_excluded(
                                item, cost, "redundant_passage", breakdown
                            )
                        )
                        excluded_ids.add(item.evidence_id)
                        continue
                    if used_tokens + cost <= effective_budget:
                        reason = self._describe_selection_reason(breakdown, matched_subs, is_conf, selected_domains[item.source_domain])
                        record_selection(item, breakdown, matched_subs, reason, is_conf)
                        picked_any = True
                        continue
                next_remaining.append(item)

            if not picked_any:
                # None of the remaining items fit inside effective_budget
                for item, breakdown, _ in scored_candidates:
                    if item.evidence_id in excluded_ids or item.evidence_id in selected_ids:
                        continue
                    cost = token_costs[item.evidence_id]
                    reason = (
                        "single_passage_exceeds_evidence_budget"
                        if cost > effective_budget
                        else "evidence_budget_exhausted"
                    )
                    excluded.append(self._build_excluded(item, cost, reason, breakdown))
                    excluded_ids.add(item.evidence_id)
                break

            remaining = next_remaining

        selected_records: list[SelectedEvidenceRecord] = []
        for idx, (item, breakdown, matched_subs, reason, is_conf) in enumerate(
            selected_items, start=1
        ):
            tok = token_costs[item.evidence_id]
            selected_records.append(
                SelectedEvidenceRecord(
                    evidence_id=item.evidence_id,
                    citation_label=f"[E{idx}]",
                    source_id=item.source_id,
                    source_url=item.source_url,
                    source_domain=item.source_domain,
                    source_title=item.source_title,
                    passage_text=item.passage_text,
                    original_token_count=tok,
                    allocated_token_count=tok,
                    selection_reason=reason,
                    is_conflict_evidence=is_conf,
                    conflict_ids=conflict_ids_by_ev.get(
                        item.evidence_id, list(item.corroboration.conflicting_claim_ids)
                    ),
                    covered_subtopics=matched_subs,
                    scores=breakdown,
                )
            )

        return selected_records, excluded

    def _estimate_item_tokens(self, item: EvidenceItem) -> int:
        # Include wrapper overhead (citation tag + domain + heading)
        header = f"{item.source_domain} {item.section_heading or ''} {item.source_url}"
        est = self._estimator.estimate(f"{header}\n{item.passage_text}").tokens + 12
        return max(est, 1)

    @staticmethod
    def _describe_selection_reason(
        breakdown: EvidenceScoreBreakdown,
        matched_subs: list[str],
        is_conflict: bool,
        prior_domain_count: int,
    ) -> str:
        reasons: list[str] = []
        if is_conflict:
            reasons.append("conflict_evidence")
        if matched_subs:
            reasons.append(f"covers_subtopic({','.join(matched_subs[:2])})")
        if prior_domain_count == 0:
            reasons.append("independent_domain")
        if breakdown.claim_support >= 0.70:
            reasons.append("corroborated")
        reasons.append(f"score={breakdown.final_score:.2f}")
        return "; ".join(reasons)

    @staticmethod
    def _build_excluded(
        item: EvidenceItem,
        token_cost: int,
        reason: str,
        breakdown: EvidenceScoreBreakdown | None,
    ) -> ExcludedEvidenceRecord:
        preview = item.passage_text[:180].strip()
        if len(item.passage_text) > 180:
            preview += "..."
        return ExcludedEvidenceRecord(
            evidence_id=item.evidence_id,
            source_id=item.source_id,
            source_url=item.source_url,
            source_domain=item.source_domain,
            source_title=item.source_title,
            passage_preview=preview,
            original_token_count=token_cost,
            exclusion_reason=reason,
            scores=breakdown,
        )
