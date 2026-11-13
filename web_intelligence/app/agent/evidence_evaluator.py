"""Phase 4 evidence quality evaluator separating relevance from support strength and conflicts."""

from __future__ import annotations

import re
from datetime import datetime

from web_intelligence.app.agent.models import (
    EvidenceEvaluationReport,
    EvidenceSupportStatus,
    ResearchSession,
    ResearchTask,
    TaskEvidenceAssessment,
    TaskStatus,
)
from web_intelligence.app.evidence.coverage import ResearchCoverageAnalyzer
from web_intelligence.app.query.decomposition import decompose_query
from web_intelligence.app.query.processor import STOPWORDS
from web_intelligence.app.schemas.documents import SourceClassification
from web_intelligence.app.schemas.evidence import (
    CorroborationStatus,
    EvidenceItem,
)
from web_intelligence.app.schemas.research import QueryPlan

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_METHODOLOGICAL_TERMS = (
    "benchmark",
    "evaluation",
    "evaluated",
    "measurement",
    "measured",
    "latency",
    "tokens per second",
    "tok/s",
    "hardware",
    "experiment",
    "methodology",
    "sample",
    "empirical",
    "dataset",
)


def _content_tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) >= 2
    }


class EvidenceQualityEvaluator:
    """Assesses accumulated evidence across provenance, independence, recency, specificity, and conflicts."""

    def __init__(
        self,
        *,
        coverage_analyzer: ResearchCoverageAnalyzer | None = None,
    ) -> None:
        self._coverage = coverage_analyzer or ResearchCoverageAnalyzer()

    def evaluate_session(
        self,
        session: ResearchSession,
        *,
        snippet_only_by_task: dict[str, int] | None = None,
    ) -> EvidenceEvaluationReport:
        """Compute per-task and session-wide quality assessments without equating relevance with truth."""
        snippet_counts = snippet_only_by_task or {}
        source_by_id = {s.source_id: s for s in session.collected_sources}
        ev_by_id = {e.evidence_id: e for e in session.evidence_records}
        tasks = list(session.research_plan.tasks) if session.research_plan else []

        conflict_by_ev_id: dict[str, list[str]] = {}
        for conflict in session.conflicts:
            conflict_by_ev_id.setdefault(conflict.claim_a.evidence_id, []).append(
                conflict.conflict_id
            )
            conflict_by_ev_id.setdefault(conflict.claim_b.evidence_id, []).append(
                conflict.conflict_id
            )

        task_assessments: list[TaskEvidenceAssessment] = []
        confirmed_n = corroborated_n = weak_n = conflicting_n = unverified_n = 0
        coverage_points = 0.0
        scored_tasks = 0

        for task in tasks:
            if task.status == TaskStatus.SKIPPED or task.status == TaskStatus.CANCELLED:
                continue
            scored_tasks += 1
            assessment = self._assess_task(
                task,
                session=session,
                ev_by_id=ev_by_id,
                source_by_id=source_by_id,
                conflict_by_ev_id=conflict_by_ev_id,
                snippet_only_count=snippet_counts.get(task.task_id, 0),
            )
            task_assessments.append(assessment)
            if assessment.support_status == EvidenceSupportStatus.CONFIRMED:
                confirmed_n += 1
                coverage_points += 1.0
            elif assessment.support_status == EvidenceSupportStatus.CORROBORATED:
                corroborated_n += 1
                coverage_points += 1.0
            elif assessment.support_status == EvidenceSupportStatus.CONFLICTING:
                conflicting_n += 1
                coverage_points += 0.5
            elif assessment.support_status == EvidenceSupportStatus.WEAK:
                weak_n += 1
                coverage_points += 0.45
            else:
                unverified_n += 1

        # Build CoverageAnalysis mapped onto research task subtopics
        norm, intent, topics, keywords, base_subtopics, planned_queries = decompose_query(
            session.original_query
        )
        task_subtopics = [t.question for t in tasks if t.status != TaskStatus.CANCELLED]
        merged_subtopics = list(dict.fromkeys([*task_subtopics, *base_subtopics]))
        coverage_plan = QueryPlan(
            original_query=session.original_query,
            normalized_query=norm,
            intent=intent,
            topics=topics,
            keywords=keywords,
            subtopics=merged_subtopics or [session.research_objective],
            planned_queries=planned_queries,
        )

        # Derive corroborated groups dict for coverage analyzer compatibility
        claim_groups: dict[str, list[Any]] = {}
        for claim in session.claims:
            k = f"{claim.entity}|{claim.attribute}|{claim.normalized_value}"
            claim_groups.setdefault(k, []).append(claim)
        corroborated_groups = {
            k: g
            for k, g in claim_groups.items()
            if len({c.source_domain for c in g}) >= 2
        }

        coverage_analysis = self._coverage.analyze(
            coverage_plan,
            session.evidence_records,
            session.claims,
            session.conflicts,
            corroborated_groups,
        )

        # Compute objective coverage ratio cleanly across planned tasks
        task_ratio = round(coverage_points / max(1, scored_tasks), 4)
        unique_primary_domains = len(
            {
                s.domain
                for s in session.collected_sources
                if not s.is_duplicate
                and s.source_id in {d.source_id for d in session.source_decisions if d.retained}
                or (not session.source_decisions and not s.is_duplicate and bool(s.main_content))
            }
        )
        retained_n = sum(1 for d in session.source_decisions if d.retained)
        rejected_n = sum(1 for d in session.source_decisions if not d.retained)
        if not session.source_decisions and session.collected_sources:
            retained_n = sum(1 for s in session.collected_sources if not s.is_duplicate)
            rejected_n = sum(1 for s in session.collected_sources if s.is_duplicate)

        if not session.evidence_records:
            overall_status = EvidenceSupportStatus.UNVERIFIED
        elif session.conflicts or conflicting_n > 0:
            overall_status = EvidenceSupportStatus.CONFLICTING
        elif confirmed_n > 0 and unverified_n == 0 and weak_n == 0:
            overall_status = EvidenceSupportStatus.CONFIRMED
        elif (confirmed_n + corroborated_n) > 0 and task_ratio >= 0.70 and unverified_n == 0:
            overall_status = EvidenceSupportStatus.CORROBORATED
        else:
            overall_status = EvidenceSupportStatus.WEAK

        summary = (
            f"Overall evidence status: {overall_status.value} "
            f"(coverage={task_ratio:.0%}, {unique_primary_domains} independent domain(s), "
            f"{confirmed_n} confirmed / {corroborated_n} corroborated / {weak_n} weak / "
            f"{conflicting_n} conflicting / {unverified_n} unverified task(s))."
        )
        return EvidenceEvaluationReport(
            iteration=session.current_iteration,
            overall_support_status=overall_status,
            objective_coverage_ratio=task_ratio,
            total_sources_evaluated=len(session.collected_sources),
            unique_primary_domains=unique_primary_domains,
            total_accepted_evidence=len(session.evidence_records),
            total_rejected_sources=rejected_n,
            confirmed_tasks=confirmed_n,
            corroborated_tasks=corroborated_n,
            weak_tasks=weak_n,
            conflicting_tasks=conflicting_n,
            unverified_tasks=unverified_n,
            conflict_count=len(session.conflicts),
            task_assessments=task_assessments,
            coverage_analysis=coverage_analysis,
            summary=summary,
        )

    def _assess_task(
        self,
        task: ResearchTask,
        *,
        session: ResearchSession,
        ev_by_id: dict[str, EvidenceItem],
        source_by_id: dict[str, Any],
        conflict_by_ev_id: dict[str, list[str]],
        snippet_only_count: int,
    ) -> TaskEvidenceAssessment:
        # Match explicitly attached evidence + semantically overlapping session evidence
        task_ev: list[EvidenceItem] = [
            ev_by_id[eid] for eid in task.evidence_ids if eid in ev_by_id
        ]
        seen_eids = {e.evidence_id for e in task_ev}
        q_tokens = _content_tokens(f"{task.question} {' '.join(task.entities_keywords)}")

        if not task_ev and q_tokens:
            for item in session.evidence_records:
                if item.evidence_id in seen_eids:
                    continue
                ev_tokens = _content_tokens(
                    f"{item.section_heading or ''} {item.source_title or ''} {item.passage_text}"
                )
                overlap = len(q_tokens & ev_tokens) / max(1, len(q_tokens))
                if overlap >= 0.45:
                    task_ev.append(item)
                    seen_eids.add(item.evidence_id)

        if not task_ev:
            note = (
                f"Only {snippet_only_count} search snippet(s) returned without full page content."
                if snippet_only_count > 0
                else "No retained primary or secondary passages address this task yet."
            )
            return TaskEvidenceAssessment(
                task_id=task.task_id,
                question=task.question,
                support_status=EvidenceSupportStatus.UNVERIFIED,
                answered=False,
                snippet_only_count=snippet_only_count,
                notes=note,
            )

        # Compute source independence excluding mirror/duplicate clusters
        primary_domains: set[str] = set()
        secondary_domains: set[str] = set()
        reference_domains: set[str] = set()
        unique_domains: set[str] = set()
        mirror_count = 0
        dates: list[datetime] = []
        cues_found: set[str] = set()
        directness_vals: list[float] = []
        rel_vals: list[float] = []
        conflict_ids: set[str] = set()

        for item in task_ev:
            rel_vals.append(float(item.relevance.combined_score))
            ev_tokens = _content_tokens(
                f"{item.section_heading or ''} {item.source_title or ''} {item.passage_text}"
            )
            directness = len(q_tokens & ev_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
            directness_vals.append(directness)

            unique_domains.add(item.source_domain)
            cls = item.provenance.source_classification
            if cls == SourceClassification.PRIMARY:
                primary_domains.add(item.source_domain)
            elif cls == SourceClassification.SECONDARY:
                secondary_domains.add(item.source_domain)
            elif cls == SourceClassification.REFERENCE:
                reference_domains.add(item.source_domain)

            src = source_by_id.get(item.source_id)
            if src and src.alias_urls:
                mirror_count += len(src.alias_urls)

            if item.provenance.published_at is not None:
                dates.append(item.provenance.published_at)

            low_text = item.passage_text.lower()
            for cue in _METHODOLOGICAL_TERMS:
                if cue in low_text:
                    cues_found.add(cue)

            for cid in conflict_by_ev_id.get(item.evidence_id, []):
                conflict_ids.add(cid)
            for cid in item.corroboration.conflicting_claim_ids:
                conflict_ids.add(cid)

        # Count single-source claims tied to task evidence
        task_ev_ids = {e.evidence_id for e in task_ev}
        task_claims = [c for c in session.claims if c.evidence_id in task_ev_ids]
        single_source_claim_n = sum(
            1
            for claim in task_claims
            if claim.evidence_id in task_ev_ids
            and ev_by_id[claim.evidence_id].corroboration.corroboration_status
            in {CorroborationStatus.SINGLE_SOURCE, CorroborationStatus.UNVERIFIED}
        )

        max_rel = round(max(rel_vals) if rel_vals else 0.0, 4)
        max_direct = round(max(directness_vals) if directness_vals else 0.0, 4)
        indep_n = len(unique_domains)
        corroborated_items = any(
            item.corroboration.corroboration_status == CorroborationStatus.CORROBORATED
            and item.corroboration.independent_supporting_sources >= 2
            for item in task_ev
        )

        # Classification strictly avoids equating high relevance with strong support!
        if conflict_ids:
            status = EvidenceSupportStatus.CONFLICTING
            notes = (
                f"Evidence spans {indep_n} domain(s) with {len(conflict_ids)} unresolved conflict(s); "
                "requires conflict reconciliation before treating claims as settled."
            )
        elif (len(primary_domains) >= 1 and indep_n >= 2 and max_direct >= 0.22) or (
            len(primary_domains) >= 1 and corroborated_items
        ):
            status = EvidenceSupportStatus.CONFIRMED
            notes = (
                f"Confirmed by {len(primary_domains)} primary domain(s) across {indep_n} "
                "independent domains."
            )
        elif (indep_n >= 2 and max_direct >= 0.20) or corroborated_items:
            status = EvidenceSupportStatus.CORROBORATED
            notes = (
                f"Corroborated across {indep_n} independent domain(s), though lacking multiple "
                "authoritative primary sources."
                if not primary_domains
                else f"Corroborated across {indep_n} independent domain(s)."
            )
        else:
            status = EvidenceSupportStatus.WEAK
            reasons: list[str] = []
            if indep_n < 2:
                reasons.append("only a single independent domain supports passages so far")
            if not primary_domains:
                reasons.append("no primary/official source retrieved")
            if max_direct < 0.20:
                reasons.append("low direct term overlap with task question")
            if snippet_only_count > 0:
                reasons.append(f"{snippet_only_count} result(s) remain snippet-only")
            notes = "Weak support: " + ("; ".join(reasons) or "limited corroboration.")

        newest_dt = max(dates) if dates else None
        return TaskEvidenceAssessment(
            task_id=task.task_id,
            question=task.question,
            support_status=status,
            answered=max_direct >= 0.20 or max_rel >= 0.15,
            relevance_score=max_rel,
            directness_score=max_direct,
            primary_source_count=len(primary_domains),
            secondary_source_count=len(secondary_domains),
            reference_source_count=len(reference_domains),
            independent_domain_count=indep_n,
            mirror_or_duplicate_count=mirror_count,
            snippet_only_count=snippet_only_count,
            dated_source_count=len(dates),
            newest_source_date=newest_dt,
            methodological_cues_found=sorted(cues_found),
            conflict_ids=sorted(conflict_ids),
            unsupported_claim_count=single_source_claim_n,
            evidence_ids=[e.evidence_id for e in task_ev],
            notes=notes,
        )
