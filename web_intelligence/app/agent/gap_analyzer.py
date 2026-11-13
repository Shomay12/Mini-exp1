"""Seven-category GapAnalyzer turning evidence weaknesses and conflicts into justified follow-ups."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from web_intelligence.app.agent.models import (
    EvidenceEvaluationReport,
    EvidenceSupportStatus,
    GapAnalysisReport,
    GapCategory,
    ResearchGap,
    ResearchSession,
    TaskStatus,
)
from web_intelligence.app.query.decomposition import classify_intent
from web_intelligence.app.schemas.research import SearchIntent


class GapAnalyzer:
    """Identifies concrete unanswered questions, missing primary evidence, conflicts, and staleness."""

    def __init__(self, *, stale_days: int = 400) -> None:
        self._stale_days = stale_days

    def analyze(
        self,
        session: ResearchSession,
        evaluation: EvidenceEvaluationReport,
        *,
        temporal_focus: bool = False,
    ) -> GapAnalysisReport:
        """Detect actionable gaps while avoiding duplicate re-emission of already tracked gaps."""
        existing_keys: dict[str, ResearchGap] = {
            self._gap_key(g): g for g in session.identified_gaps
        }
        detected: list[ResearchGap] = []

        assessment_by_task = {a.task_id: a for a in evaluation.task_assessments}
        tasks = list(session.research_plan.tasks) if session.research_plan else []
        task_by_id = {t.task_id: t for t in tasks}

        # 1 & 6 & 7: Per-task gaps (unanswered, weak support, missing primary source)
        for task in tasks:
            if task.status in {TaskStatus.CANCELLED, TaskStatus.SKIPPED}:
                continue
            assess = assessment_by_task.get(task.task_id)
            if assess is None:
                continue

            if assess.support_status == EvidenceSupportStatus.UNVERIFIED and task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
            }:
                kw_hint = " ".join(task.entities_keywords[:3]) or task.question
                self._register_gap(
                    existing_keys,
                    detected,
                    ResearchGap(
                        gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                        category=GapCategory.UNANSWERED_QUESTION,
                        severity="high",
                        description=(
                            f"Task '{task.task_id}' ('{task.question}') has no verified supporting evidence."
                        ),
                        related_task_id=task.task_id,
                        suggested_followup_question=f"{kw_hint} overview evidence documentation benchmark",
                        suggested_objective=(
                            f"Find concrete primary or technical evidence answering: {task.question}"
                        ),
                        expected_evidence="At least two passages from independent sources.",
                        suggested_tools=["web_search", "url_retrieve"],
                        priority=5,
                    ),
                )
            elif assess.support_status == EvidenceSupportStatus.WEAK:
                if assess.primary_source_count == 0 and (
                    assess.secondary_source_count > 0 or assess.reference_source_count > 0
                ):
                    self._register_gap(
                        existing_keys,
                        detected,
                        ResearchGap(
                            gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                            category=GapCategory.MISSING_PRIMARY_SOURCE,
                            severity="high",
                            description=(
                                f"Task '{task.task_id}' relies solely on secondary/reference pages "
                                "without an authoritative primary or specification source."
                            ),
                            related_task_id=task.task_id,
                            suggested_followup_question=(
                                f"{task.question} official specification documentation paper arxiv"
                            ),
                            suggested_objective=(
                                f"Locate primary documentation or empirical paper for: {task.question}"
                            ),
                            expected_evidence="Official docs, RFC, spec, repository, or peer-reviewed paper.",
                            suggested_tools=["web_search", "url_retrieve"],
                            priority=4,
                        ),
                    )
                else:
                    self._register_gap(
                        existing_keys,
                        detected,
                        ResearchGap(
                            gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                            category=GapCategory.WEAK_SUPPORT,
                            severity="medium",
                            description=(
                                f"Task '{task.task_id}' currently has weak support: {assess.notes}"
                            ),
                            related_task_id=task.task_id,
                            suggested_followup_question=(
                                f"{task.question} independent evaluation verification"
                            ),
                            suggested_objective=(
                                f"Strengthen evidence with independent verification for: {task.question}"
                            ),
                            suggested_tools=["web_search", "url_retrieve"],
                            priority=3,
                        ),
                    )

        # 2. Missing Coverage Categories / Missing Subtopics
        if evaluation.coverage_analysis:
            for missing in evaluation.coverage_analysis.missing_topics[:3]:
                self._register_gap(
                    existing_keys,
                    detected,
                    ResearchGap(
                        gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                        category=GapCategory.MISSING_CATEGORY,
                        severity="high",
                        description=f"Subtopic lacks evidence coverage: '{missing}'.",
                        suggested_followup_question=f"{session.original_query} {missing}",
                        suggested_objective=f"Cover missing objective facet: {missing}",
                        expected_evidence="Retained full-text passages addressing this subtopic.",
                        suggested_tools=["web_search", "url_retrieve"],
                        priority=4,
                    ),
                )

        # 4. Conflicting Results -> targeted verification/reconciliation follow-up
        for conflict in session.conflicts:
            self._register_gap(
                existing_keys,
                detected,
                ResearchGap(
                    gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                    category=GapCategory.CONFLICTING_RESULTS,
                    severity="high",
                    description=conflict.summary,
                    related_claim_ids=[conflict.claim_a.claim_id, conflict.claim_b.claim_id],
                    related_conflict_id=conflict.conflict_id,
                    suggested_followup_question=(
                        f"{conflict.entity} {conflict.attribute} methodology benchmark comparison primary source"
                    ),
                    suggested_objective=(
                        f"Investigate root cause of discrepancy on '{conflict.entity} {conflict.attribute}' "
                        f"({conflict.claim_a.source_domain} vs {conflict.claim_b.source_domain})."
                    ),
                    expected_evidence=(
                        "Independent measurement explaining methodology, revision date, or conditions."
                    ),
                    suggested_tools=["web_search", "url_retrieve"],
                    priority=5,
                ),
            )

        # 3. Single-Source Claims (limit to top 2 impactful metrics/dates not already in a conflict)
        conflicted_claim_ids = {
            cid
            for conf in session.conflicts
            for cid in (conf.claim_a.claim_id, conf.claim_b.claim_id)
        }
        claim_domains_by_sig: dict[str, set[str]] = {}
        for c in session.claims:
            sig = f"{c.entity}|{c.attribute}"
            claim_domains_by_sig.setdefault(sig, set()).add(c.source_domain)

        single_emitted = 0
        for c in session.claims:
            if single_emitted >= 2:
                break
            if c.claim_id in conflicted_claim_ids:
                continue
            sig = f"{c.entity}|{c.attribute}"
            if len(claim_domains_by_sig.get(sig, set())) >= 2:
                continue
            unit = f" {c.unit}" if c.unit else ""
            self._register_gap(
                existing_keys,
                detected,
                ResearchGap(
                    gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                    category=GapCategory.SINGLE_SOURCE_CLAIM,
                    severity="medium",
                    description=(
                        f"Single-source {c.attribute} claim ({c.normalized_value}{unit}) reported "
                        f"only by {c.source_domain}."
                    ),
                    related_claim_ids=[c.claim_id],
                    suggested_followup_question=(
                        f"{c.entity} {c.attribute} independent verification benchmark"
                    ),
                    suggested_objective=(
                        f"Verify single-source claim '{c.entity} {c.attribute}={c.normalized_value}{unit}' "
                        "against an independent source."
                    ),
                    suggested_tools=["web_search", "url_retrieve"],
                    priority=3,
                ),
            )
            single_emitted += 1

        # 5. Outdated / Undated Information when temporal precision matters
        plan_temporal = temporal_focus or (
            classify_intent(session.original_query) == SearchIntent.TEMPORAL
        )
        now = datetime.now(timezone.utc)
        dated_sources = [s.published_at for s in session.collected_sources if s.published_at and not s.is_duplicate]
        all_stale = bool(dated_sources) and all(
            (now - (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc))) > timedelta(days=self._stale_days)
            for dt in dated_sources
        )
        all_undated = bool(session.collected_sources) and not dated_sources
        if (plan_temporal and (all_stale or all_undated)) or all_stale:
            desc = (
                "All retrieved sources appear older than expected for a temporal query."
                if all_stale
                else "Retrieved sources lack publication dates despite temporal query focus."
            )
            self._register_gap(
                existing_keys,
                detected,
                ResearchGap(
                    gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                    category=GapCategory.OUTDATED_INFORMATION,
                    severity="medium" if plan_temporal else "low",
                    description=desc,
                    suggested_followup_question=f"{session.original_query} 2026 release notes update",
                    suggested_objective="Find explicitly dated, up-to-date primary information.",
                    suggested_tools=["web_search", "url_retrieve"],
                    priority=4 if plan_temporal else 2,
                ),
            )

        # Merge newly detected unique gaps into session.identified_gaps cleanly
        existing_ids = {g.gap_id for g in session.identified_gaps}
        for gap in detected:
            if gap.gap_id not in existing_ids:
                existing_ids.add(gap.gap_id)
                session.identified_gaps.append(gap)

        # Mark resolved when conflict gets >=3 domains supporting or assigned task completed with evidence
        for gap in session.identified_gaps:
            if gap.resolved:
                continue
            if gap.related_task_id and gap.related_task_id in task_by_id:
                assess = assessment_by_task.get(gap.related_task_id)
                if assess and assess.support_status in {
                    EvidenceSupportStatus.CONFIRMED,
                    EvidenceSupportStatus.CORROBORATED,
                }:
                    gap.resolved = True
            if gap.addressed_by_task_id and gap.addressed_by_task_id in task_by_id:
                t = task_by_id[gap.addressed_by_task_id]
                if t.status == TaskStatus.COMPLETED and t.evidence_ids:
                    gap.resolved = True

        unresolved = [g for g in session.identified_gaps if not g.resolved]
        high_n = sum(1 for g in unresolved if g.severity == "high")
        med_n = sum(1 for g in unresolved if g.severity == "medium")
        low_n = sum(1 for g in unresolved if g.severity == "low")
        summary = (
            f"Identified {len(unresolved)} unresolved gap(s) ({high_n} high, {med_n} medium, {low_n} low) "
            f"across {len(session.identified_gaps)} total tracked gap(s)."
        )
        return GapAnalysisReport(
            iteration=session.current_iteration,
            gaps=unresolved,
            high_priority_count=high_n,
            medium_priority_count=med_n,
            low_priority_count=low_n,
            summary=summary,
        )

    @staticmethod
    def _gap_key(gap: ResearchGap) -> str:
        target = (
            gap.related_conflict_id
            or gap.related_task_id
            or (",".join(sorted(gap.related_claim_ids)) if gap.related_claim_ids else gap.description[:80])
        )
        return f"{gap.category.value}::{target}"

    def _register_gap(
        self,
        existing_keys: dict[str, ResearchGap],
        detected: list[ResearchGap],
        candidate: ResearchGap,
    ) -> None:
        key = self._gap_key(candidate)
        if key not in existing_keys:
            existing_keys[key] = candidate
            detected.append(candidate)
