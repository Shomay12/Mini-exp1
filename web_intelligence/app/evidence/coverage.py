"""Research coverage evaluator tracking supported, conflicted, single-source, and missing facets."""

from __future__ import annotations

import re

from web_intelligence.app.query.processor import STOPWORDS
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CoverageAnalysis,
    CoverageStatus,
    EvidenceItem,
    ExtractedClaim,
    SubtopicCoverage,
    SubtopicStatus,
)
from web_intelligence.app.schemas.research import QueryPlan

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _meaningful_tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower())
        if len(t) >= 2 and t not in STOPWORDS
    }


class ResearchCoverageAnalyzer:
    """Evaluates how thoroughly retrieved evidence covers decomposed query subtopics."""

    def __init__(self, *, min_relevance_threshold: float = 0.12) -> None:
        self._min_rel = min_relevance_threshold

    def analyze(
        self,
        query_plan: QueryPlan,
        evidence: list[EvidenceItem],
        claims: list[ExtractedClaim],
        conflicts: list[ClaimConflict],
        corroborated_groups: dict[str, list[ExtractedClaim]],
    ) -> CoverageAnalysis:
        """Produce explicit coverage sufficiency, gaps, single-source claims, and conflicts."""
        subtopics = query_plan.subtopics or [query_plan.normalized_query]

        covered_topics: list[str] = []
        partial_topics: list[str] = []
        missing_topics: list[str] = []
        subtopic_details: list[SubtopicCoverage] = []
        unanswered_questions: list[str] = []

        for subtopic in subtopics:
            sub_tokens = _meaningful_tokens(subtopic)
            if not sub_tokens:
                continue

            matching_ev: list[EvidenceItem] = []
            for item in evidence:
                if item.relevance.combined_score < self._min_rel and item.relevance.bm25_score <= 0.0:
                    continue
                ev_tokens = _meaningful_tokens(
                    f"{item.section_heading or ''} {item.source_title or ''} {item.passage_text}"
                )
                overlap = len(sub_tokens & ev_tokens)
                required_hits = 1 if len(sub_tokens) <= 2 else max(2, len(sub_tokens) // 2)
                if overlap >= required_hits:
                    matching_ev.append(item)

            ev_ids = [ev.evidence_id for ev in matching_ev]
            src_ids = sorted({ev.source_id for ev in matching_ev})
            domains = sorted({ev.source_domain for ev in matching_ev})
            ev_id_set = set(ev_ids)

            sub_has_conflict = any(
                c.claim_a.evidence_id in ev_id_set or c.claim_b.evidence_id in ev_id_set
                for c in conflicts
            )

            if not matching_ev:
                status = SubtopicStatus.MISSING
                missing_topics.append(subtopic)
                unanswered_questions.append(f"No supporting evidence retrieved for subtopic: '{subtopic}'.")
                note = "No retrieved passage met term-overlap and relevance thresholds."
            elif len(domains) >= 2 and not sub_has_conflict:
                status = SubtopicStatus.COVERED
                covered_topics.append(subtopic)
                note = f"Supported by {len(domains)} independent domains without detected conflicts."
            else:
                status = SubtopicStatus.PARTIAL
                partial_topics.append(subtopic)
                reasons: list[str] = []
                if len(domains) < 2:
                    reasons.append("only a single independent domain retrieved")
                    unanswered_questions.append(
                        f"Independent corroboration needed for '{subtopic}' (currently single-source)."
                    )
                if sub_has_conflict:
                    reasons.append("conflicting claims detected across sources")
                    unanswered_questions.append(
                        f"Resolve conflicting claims observed for '{subtopic}'."
                    )
                note = "Partial coverage: " + "; ".join(reasons) + "."

            subtopic_details.append(
                SubtopicCoverage(
                    subtopic=subtopic,
                    status=status,
                    supporting_evidence_ids=ev_ids,
                    supporting_source_ids=src_ids,
                    independent_domain_count=len(domains),
                    has_conflict=sub_has_conflict,
                    notes=note,
                )
            )

        # Track claim corroboration inventory
        corroborated_claim_ids: set[str] = set()
        well_corroborated: list[str] = []
        for group in corroborated_groups.values():
            domains = sorted({c.source_domain for c in group})
            if len(domains) >= 2:
                for c in group:
                    corroborated_claim_ids.add(c.claim_id)
                rep = group[0]
                unit_str = f" {rep.unit}" if rep.unit else ""
                well_corroborated.append(
                    f"{rep.entity} {rep.attribute}={rep.normalized_value}{unit_str} "
                    f"(supported by {len(domains)} domains: {', '.join(domains)})"
                )

        conflicted_claim_ids: set[str] = set()
        conflicted_areas: list[str] = []
        for conflict in conflicts:
            conflicted_claim_ids.add(conflict.claim_a.claim_id)
            conflicted_claim_ids.add(conflict.claim_b.claim_id)
            if conflict.summary not in conflicted_areas:
                conflicted_areas.append(conflict.summary)

        single_source_claims: list[str] = []
        for claim in claims:
            if claim.claim_id not in corroborated_claim_ids and claim.claim_id not in conflicted_claim_ids:
                unit_str = f" {claim.unit}" if claim.unit else ""
                desc = f"{claim.entity} {claim.attribute}={claim.normalized_value}{unit_str} ({claim.source_domain})"
                if desc not in single_source_claims:
                    single_source_claims.append(desc)

        # Overall coverage status
        if not evidence or (not covered_topics and not partial_topics):
            overall = CoverageStatus.INSUFFICIENT
        elif covered_topics and not missing_topics and not conflicts and not partial_topics:
            overall = CoverageStatus.SUFFICIENT
        else:
            overall = CoverageStatus.PARTIAL

        summary = (
            f"Coverage is {overall.value}: {len(covered_topics)} covered, "
            f"{len(partial_topics)} partial, {len(missing_topics)} missing subtopic(s); "
            f"{len(conflicts)} conflict(s) detected."
        )

        return CoverageAnalysis(
            covered_topics=covered_topics,
            partial_topics=partial_topics,
            missing_topics=missing_topics,
            subtopic_details=subtopic_details,
            well_corroborated_claims=well_corroborated,
            single_source_claims=single_source_claims,
            conflicted_areas=conflicted_areas,
            unanswered_questions=unanswered_questions,
            coverage_status=overall,
            summary=summary,
        )
