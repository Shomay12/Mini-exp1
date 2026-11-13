"""Model-independent ResearchPackage context builder."""

from __future__ import annotations

from datetime import datetime, timezone

from web_intelligence.app.privacy.policy import PrivacyPolicy
from web_intelligence.app.research.research_state import ResearchState
from web_intelligence.app.schemas.documents import ExtractionStatus, SourceRecord
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CoverageAnalysis,
    EvidenceItem,
    ExtractedClaim,
)
from web_intelligence.app.schemas.research import (
    QueryPlan,
    ResearchMetadata,
    ResearchPackage,
    ResearchStatus,
)
from web_intelligence.app.schemas.search import DisclosedSearchQuery, ProviderDiagnostic


class ContextBuilder:
    """Assembles a validated, model-agnostic ResearchPackage from Stage 3 & Stage 4 outputs."""

    def build(
        self,
        *,
        state: ResearchState,
        query_plan: QueryPlan | None,
        search_queries: list[DisclosedSearchQuery],
        sources: list[SourceRecord],
        evidence: list[EvidenceItem],
        claims: list[ExtractedClaim],
        conflicts: list[ClaimConflict],
        coverage: CoverageAnalysis,
        provider_diagnostics: list[ProviderDiagnostic],
        privacy_policy: PrivacyPolicy,
        error_message: str | None = None,
    ) -> ResearchPackage:
        """Compile all verified artifacts into a structured ResearchPackage."""
        processed_docs = sum(
            1
            for s in sources
            if not s.is_duplicate
            and s.extraction_status in {ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL}
        )
        duplicate_docs = sum(1 for s in sources if s.is_duplicate)
        failed_docs = sum(
            1
            for s in sources
            if s.extraction_status not in {ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL}
        )
        total_tokens = sum(ev.token_estimate for ev in evidence)

        ranked_evidence = sorted(
            evidence,
            key=lambda item: (item.relevance.combined_score, item.relevance.bm25_score),
            reverse=True,
        )

        metadata = ResearchMetadata(
            retrieved_sources=len(sources),
            processed_documents=processed_docs,
            duplicate_documents=duplicate_docs,
            failed_documents=failed_docs,
            evidence_items=len(ranked_evidence),
            claims_extracted=len(claims),
            conflicts_detected=len(conflicts),
            total_token_estimate=total_tokens,
            retrieved_at=datetime.now(timezone.utc),
            provider_diagnostics=provider_diagnostics,
            privacy_notice=privacy_policy.disclosure_notice(),
        )

        return ResearchPackage(
            research_id=state.research_id,
            original_query=state.original_query,
            research_status=state.status if error_message is None else ResearchStatus.FAILED,
            query_plan=query_plan,
            search_queries=search_queries,
            sources=sources,
            evidence=ranked_evidence,
            claims=claims,
            conflicts=conflicts,
            coverage=coverage,
            metadata=metadata,
            progress=state.to_progress(),
            error_message=error_message,
        )
