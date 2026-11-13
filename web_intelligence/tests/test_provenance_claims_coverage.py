"""Tests for provenance evaluation, claim conflicts, corroboration, and coverage gaps."""

from __future__ import annotations

from datetime import datetime, timezone

from web_intelligence.app.evidence.claims import (
    DeterministicClaimComparator,
    RuleBasedClaimExtractor,
)
from web_intelligence.app.evidence.coverage import ResearchCoverageAnalyzer
from web_intelligence.app.evidence.evaluator import SourceProvenanceEvaluator
from web_intelligence.app.query.planner import DeterministicQueryPlanner
from web_intelligence.app.schemas.documents import (
    ExtractionStatus,
    SourceClassification,
    SourceMetadata,
    SourceRecord,
)
from web_intelligence.app.schemas.evidence import (
    CorroborationStatus,
    CoverageStatus,
    EvidenceItem,
    RelevanceBreakdown,
    SubtopicStatus,
)


def _make_evidence(
    ev_id: str,
    source_id: str,
    domain: str,
    passage: str,
    *,
    title: str = "Report",
    author: str | None = None,
    publisher: str | None = None,
    url_path: str = "/article",
    relevance: float = 0.6,
) -> tuple[SourceRecord, EvidenceItem]:
    url = f"https://{domain}{url_path}"
    src = SourceRecord(
        source_id=source_id,
        original_url=url,
        canonical_url=url,
        final_url=url,
        domain=domain,
        title=title,
        author=author,
        retrieved_at=datetime.now(timezone.utc),
        main_content=passage,
        content_hash=ev_id,
        extraction_status=ExtractionStatus.SUCCESS,
        source_metadata=SourceMetadata(publisher=publisher),
    )
    prov = SourceProvenanceEvaluator().build_provenance(src)
    src = src.model_copy(update={"provenance": prov})
    ev = EvidenceItem(
        evidence_id=ev_id,
        source_id=source_id,
        source_url=url,
        source_domain=domain,
        source_title=title,
        chunk_index=0,
        section_heading=title,
        passage_text=passage,
        char_start=0,
        char_end=len(passage),
        token_estimate=len(passage.split()),
        relevance=RelevanceBreakdown(
            bm25_score=relevance,
            keyword_match_score=relevance,
            heading_relevance_score=relevance,
            query_term_coverage=relevance,
            document_relevance_score=relevance,
            combined_score=relevance,
        ),
        provenance=prov,
    )
    return src, ev


async def test_conflicting_dates_numerical_claims_and_corroboration() -> None:
    s1, e1 = _make_evidence(
        "ev1",
        "src1",
        "docs.openmodels.org",
        (
            "Aurora-SLM-4B was released in January 2026 with Apache 2.0 licensing. "
            "On Apple M2 laptops, Aurora-SLM-4B achieves 48 tokens per second. "
            "A 4-bit quantized 7B model requires 4.5 GB of RAM."
        ),
        publisher="OpenModels Official Docs",
        url_path="/slm/overview",
    )
    s2, e2 = _make_evidence(
        "ev2",
        "src2",
        "research-digest.example.com",
        (
            "Vendor sheets state Aurora-SLM-4B was released in March 2026. "
            "Hardware measurements show a 4-bit quantized 7B model requires 4.5 GB of RAM."
        ),
        author="Dr. Maya Lin",
        url_path="/review/slm",
    )
    s3, e3 = _make_evidence(
        "ev3",
        "src3",
        "lab-audit.example.edu",
        "Sustained thermal benchmarks show Aurora-SLM-4B reached 34 tokens per second on Apple M2.",
        url_path="/papers/audit",
    )

    assert s1.provenance is not None and s1.provenance.source_classification == SourceClassification.PRIMARY
    assert s2.provenance is not None and s2.provenance.source_classification == SourceClassification.SECONDARY
    assert s3.provenance is not None and s3.provenance.source_classification == SourceClassification.PRIMARY

    extractor = RuleBasedClaimExtractor()
    comparator = DeterministicClaimComparator()
    claims = extractor.extract_claims([e1, e2, e3])
    conflicts, corroborated = await comparator.compare(claims)

    conflict_types = {c.conflict_type for c in conflicts}
    assert "date_mismatch" in conflict_types
    assert "numeric_mismatch" in conflict_types
    assert any("2026-01" in c.claim_a.normalized_value or "2026-01" in c.claim_b.normalized_value for c in conflicts)
    assert any("2026-03" in c.claim_a.normalized_value or "2026-03" in c.claim_b.normalized_value for c in conflicts)

    # 7B RAM 4.5 GB claim appears across docs.openmodels.org and research-digest.example.com
    assert any("4.5" in key for key in corroborated)

    evaluator = SourceProvenanceEvaluator()
    updated_ev = evaluator.evaluate_corroboration([e1, e2, e3], [s1, s2, s3], claims, conflicts, corroborated)
    assert any(ev.corroboration.corroboration_status == CorroborationStatus.CONTESTED for ev in updated_ev)


async def test_research_coverage_identifies_covered_partial_and_missing_gaps() -> None:
    planner = DeterministicQueryPlanner()
    plan = await planner.plan("Aurora-SLM-4B memory footprint and pricing licensing and GPU clustering")

    s1, e1 = _make_evidence(
        "ev1",
        "src1",
        "docs.openmodels.org",
        "Aurora-SLM-4B memory footprint requires 4.5 GB of RAM in 4-bit quantization.",
    )
    s2, e2 = _make_evidence(
        "ev2",
        "src2",
        "benchmarks.example.edu",
        "Memory footprint measurements confirm Aurora-SLM-4B fits within 4.5 GB RAM.",
    )
    s3, e3 = _make_evidence(
        "ev3",
        "src3",
        "blog.example.com",
        "Pricing licensing uses permissive Apache 2.0 terms with zero API fees.",
        author="Sam",
    )

    claims = RuleBasedClaimExtractor().extract_claims([e1, e2, e3])
    conflicts, corroborated = await DeterministicClaimComparator().compare(claims)
    coverage = ResearchCoverageAnalyzer().analyze(plan, [e1, e2, e3], claims, conflicts, corroborated)

    by_topic = {d.subtopic: d for d in coverage.subtopic_details}
    assert coverage.coverage_status == CoverageStatus.PARTIAL
    assert len(coverage.missing_topics) >= 1
    assert any("gpu clustering" in t for t in coverage.missing_topics)
    assert len(coverage.unanswered_questions) >= 1
    assert any(d.status == SubtopicStatus.COVERED for d in by_topic.values())
    assert any(d.status == SubtopicStatus.PARTIAL for d in by_topic.values())
