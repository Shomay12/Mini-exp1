"""End-to-end research pipeline orchestrator connecting Stages 3 and 4."""

from __future__ import annotations

import uuid

from web_intelligence.app.core.config import Settings
from web_intelligence.app.core.logging import get_logger
from web_intelligence.app.core.security import AsyncResolver
from web_intelligence.app.evidence.chunker import PassageChunker, TextChunk
from web_intelligence.app.evidence.claims import (
    ClaimComparator,
    DeterministicClaimComparator,
    RuleBasedClaimExtractor,
)
from web_intelligence.app.evidence.coverage import ResearchCoverageAnalyzer
from web_intelligence.app.evidence.deduplicator import MultiLevelDeduplicator
from web_intelligence.app.evidence.evaluator import SourceProvenanceEvaluator
from web_intelligence.app.evidence.relevance import HybridRelevanceEngine
from web_intelligence.app.evidence.reranker import CrossEncoderReranker, SemanticRetriever
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.providers.router import SearchProviderRouter
from web_intelligence.app.query.planner import DeterministicQueryPlanner, QueryPlanner
from web_intelligence.app.research.context_builder import ContextBuilder
from web_intelligence.app.research.research_state import ResearchState
from web_intelligence.app.retrieval.search_engine import WebRetrievalEngine
from web_intelligence.app.schemas.documents import ExtractionStatus, SourceRecord
from web_intelligence.app.schemas.evidence import CoverageAnalysis, EvidenceItem
from web_intelligence.app.schemas.research import (
    QueryPlan,
    ResearchPackage,
    ResearchRequest,
    ResearchStatus,
)
from web_intelligence.app.schemas.search import (
    DisclosedSearchQuery,
    NormalizedSearchResult,
    ProviderDiagnostic,
)
from web_intelligence.app.storage.repositories import ResearchRepository

logger = get_logger("research.orchestrator")


class ResearchOrchestrator:
    """Executes the complete local-first search, retrieval, evidence, and coverage workflow."""

    def __init__(
        self,
        settings: Settings,
        gateway: PrivacyGateway,
        repository: ResearchRepository,
        *,
        planner: QueryPlanner | None = None,
        router: SearchProviderRouter | None = None,
        retrieval_engine: WebRetrievalEngine | None = None,
        deduplicator: MultiLevelDeduplicator | None = None,
        chunker: PassageChunker | None = None,
        relevance_engine: HybridRelevanceEngine | None = None,
        provenance_evaluator: SourceProvenanceEvaluator | None = None,
        claim_extractor: RuleBasedClaimExtractor | None = None,
        claim_comparator: ClaimComparator | None = None,
        coverage_analyzer: ResearchCoverageAnalyzer | None = None,
        context_builder: ContextBuilder | None = None,
        semantic_retriever: SemanticRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        dns_resolver: AsyncResolver | None = None,
    ) -> None:
        self._settings = settings
        self._gateway = gateway
        self._repository = repository
        self._planner = planner or DeterministicQueryPlanner()
        self._router = router or SearchProviderRouter(settings, gateway)
        self._retrieval = retrieval_engine or WebRetrievalEngine(
            settings, gateway, router=self._router, dns_resolver=dns_resolver
        )
        self._deduplicator = deduplicator or MultiLevelDeduplicator(
            near_duplicate_threshold=settings.near_duplicate_threshold
        )
        self._chunker = chunker or PassageChunker(
            target_tokens=settings.default_chunk_target_tokens,
            overlap_sentences=settings.default_chunk_overlap_sentences,
        )
        self._relevance = relevance_engine or HybridRelevanceEngine(
            semantic_retriever=semantic_retriever,
            reranker=reranker,
        )
        self._provenance = provenance_evaluator or SourceProvenanceEvaluator()
        self._claim_extractor = claim_extractor or RuleBasedClaimExtractor()
        self._claim_comparator = claim_comparator or DeterministicClaimComparator()
        self._coverage = coverage_analyzer or ResearchCoverageAnalyzer()
        self._context_builder = context_builder or ContextBuilder()

    async def run_research(
        self,
        request: ResearchRequest,
        *,
        research_id: str | None = None,
    ) -> ResearchPackage:
        """Execute all Stage 3 & Stage 4 pipeline steps and persist the ResearchPackage."""
        rid = research_id or str(uuid.uuid4())
        state = ResearchState(research_id=rid, original_query=request.query)

        query_plan: QueryPlan | None = None
        disclosed_queries: list[DisclosedSearchQuery] = []
        provider_diagnostics: list[ProviderDiagnostic] = []
        all_sources: list[SourceRecord] = []
        evidence_items: list[EvidenceItem] = []

        try:
            # 1. Query Planning & Decomposition
            state.advance("planning", ResearchStatus.PLANNING)
            query_plan = await self._planner.plan(request.query)
            state.queries_planned = len(query_plan.planned_queries)
            state.mark_stage_done("planning")

            # 2. Sanitized Search across Providers
            state.advance("searching", ResearchStatus.SEARCHING)
            merged_hits: list[NormalizedSearchResult] = []
            seen_hit_urls: set[str] = set()
            for planned_q in query_plan.planned_queries:
                hits, disclosed, diagnostics, _, _ = await self._router.search(
                    planned_q.query,
                    max_results=request.max_results_per_query,
                    providers=request.providers,
                    intent=planned_q.intent.value,
                    failover_only=True,
                )
                disclosed_queries.extend(disclosed)
                provider_diagnostics.extend(diagnostics)
                for hit in hits:
                    key = hit.canonical_url or hit.url
                    if key not in seen_hit_urls:
                        seen_hit_urls.add(key)
                        merged_hits.append(hit)
                    elif hit.url not in {h.url for h in merged_hits}:
                        # Keep distinct raw URLs too so deduplication stage records exact/canonical dups cleanly
                        merged_hits.append(hit)

            state.urls_found = len(merged_hits)
            state.mark_stage_done("searching")

            # 3. Webpage Retrieval, Extraction & Normalization
            state.advance("retrieving", ResearchStatus.RETRIEVING)
            retrieved_pairs = await self._retrieval.retrieve_results(
                merged_hits,
                max_fetch_urls=request.max_fetch_urls,
            )
            state.documents_fetched = len(retrieved_pairs)
            state.mark_stage_done("retrieving")

            # 4. Normalization -> Multi-Level Deduplication -> Provenance -> Chunking -> Relevance -> Claims -> Coverage
            state.advance("processing", ResearchStatus.PROCESSING)
            dedup_input = [
                (doc, hit.snippet, hit.provider) for hit, doc in retrieved_pairs
            ]
            dedup_result = self._deduplicator.deduplicate(dedup_input)
            state.duplicates_removed = dedup_result.duplicate_count

            # Attach provenance to all retrieved sources (unique + duplicates)
            enriched_sources: list[SourceRecord] = []
            primary_enriched: list[SourceRecord] = []
            for src in dedup_result.all_sources:
                prov = self._provenance.build_provenance(src)
                updated_src = src.model_copy(update={"provenance": prov})
                enriched_sources.append(updated_src)
                if not updated_src.is_duplicate and updated_src.extraction_status in {
                    ExtractionStatus.SUCCESS,
                    ExtractionStatus.PARTIAL,
                }:
                    primary_enriched.append(updated_src)

            state.documents_processed = len(primary_enriched)

            # Passage-aware chunking over unique usable documents
            all_chunks: list[TextChunk] = []
            for src in primary_enriched:
                all_chunks.extend(
                    self._chunker.chunk_document(
                        src,
                        target_tokens=request.chunk_target_tokens,
                    )
                )

            # Hybrid relevance scoring (BM25 + keyword + heading + coverage + optional semantic/rerank)
            doc_scores, chunk_breakdowns = await self._relevance.score_chunks(
                query_plan.normalized_query,
                query_plan.keywords,
                primary_enriched,
                all_chunks,
            )

            source_by_id = {s.source_id: s for s in enriched_sources}
            # Update document relevance score on sources
            all_sources = [
                s.model_copy(update={"document_relevance_score": doc_scores.get(s.source_id, 0.0)})
                for s in enriched_sources
            ]
            source_by_id = {s.source_id: s for s in all_sources}

            raw_evidence: list[EvidenceItem] = []
            for chunk in all_chunks:
                src = source_by_id[chunk.source_id]
                assert src.provenance is not None
                raw_evidence.append(
                    EvidenceItem(
                        evidence_id=chunk.chunk_id,
                        source_id=chunk.source_id,
                        source_url=src.final_url or src.original_url,
                        source_domain=src.domain,
                        source_title=src.title,
                        chunk_index=chunk.chunk_index,
                        section_heading=chunk.section_heading,
                        passage_text=chunk.passage_text,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        token_estimate=chunk.token_estimate,
                        relevance=chunk_breakdowns[chunk.chunk_id],
                        provenance=src.provenance,
                    )
                )

            # Claim extraction & conflict detection
            claims = self._claim_extractor.extract_claims(raw_evidence)
            conflicts, corroborated_groups = await self._claim_comparator.compare(claims)

            # Cross-source corroboration evaluation
            evidence_items = self._provenance.evaluate_corroboration(
                raw_evidence,
                all_sources,
                claims,
                conflicts,
                corroborated_groups,
            )
            state.evidence_items_count = len(evidence_items)

            # Research coverage analysis
            coverage = self._coverage.analyze(
                query_plan,
                evidence_items,
                claims,
                conflicts,
                corroborated_groups,
            )

            state.mark_stage_done("processing")
            state.status = ResearchStatus.COMPLETED
            state.mark_stage_done("completed")

            package = self._context_builder.build(
                state=state,
                query_plan=query_plan,
                search_queries=disclosed_queries,
                sources=all_sources,
                evidence=evidence_items,
                claims=claims,
                conflicts=conflicts,
                coverage=coverage,
                provider_diagnostics=provider_diagnostics,
                privacy_policy=self._gateway.policy,
            )
            await self._repository.save_research_package(package)
            return package

        except Exception as exc:
            logger.exception("Research workflow failed", extra={"extra_data": {"research_id": rid}})
            state.status = ResearchStatus.FAILED
            package = self._context_builder.build(
                state=state,
                query_plan=query_plan,
                search_queries=disclosed_queries,
                sources=all_sources,
                evidence=evidence_items,
                claims=[],
                conflicts=[],
                coverage=CoverageAnalysis(summary=f"Research failed: {exc}"),
                provider_diagnostics=provider_diagnostics,
                privacy_policy=self._gateway.policy,
                error_message=str(exc),
            )
            await self._repository.save_research_package(package)
            return package
