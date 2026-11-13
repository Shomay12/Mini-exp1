"""Incremental observation processor wrapping Stage 3-4 retrieval, deduplication, and claims."""

from __future__ import annotations

import uuid
from typing import Any

from web_intelligence.app.agent.models import (
    NormalizedObservation,
    ResearchSession,
    ResearchTask,
    SourceDecisionRecord,
    ToolExecutionResult,
)
from web_intelligence.app.evidence.chunker import PassageChunker, TextChunk
from web_intelligence.app.evidence.claims import (
    DeterministicClaimComparator,
    RuleBasedClaimExtractor,
)
from web_intelligence.app.evidence.deduplicator import (
    MultiLevelDeduplicator,
    compute_shingle_similarity,
)
from web_intelligence.app.evidence.evaluator import SourceProvenanceEvaluator
from web_intelligence.app.evidence.relevance import HybridRelevanceEngine
from web_intelligence.app.query.processor import extract_keywords
from web_intelligence.app.schemas.documents import (
    ExtractionStatus,
    RetrievedDocument,
    SourceRecord,
)
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CorroborationSignals,
    EvidenceItem,
    ExtractedClaim,
)
from web_intelligence.app.schemas.search import (
    DisclosedSearchQuery,
    NormalizedSearchResult,
    ProviderDiagnostic,
)


class ObservationProcessor:
    """Normalizes raw tool outputs through Stage 3-4 provenance, deduplication, and conflict pipelines."""

    def __init__(
        self,
        *,
        deduplicator: MultiLevelDeduplicator | None = None,
        provenance_evaluator: SourceProvenanceEvaluator | None = None,
        chunker: PassageChunker | None = None,
        relevance_engine: HybridRelevanceEngine | None = None,
        claim_extractor: RuleBasedClaimExtractor | None = None,
        claim_comparator: DeterministicClaimComparator | None = None,
        near_duplicate_threshold: float = 0.72,
    ) -> None:
        self._deduplicator = deduplicator or MultiLevelDeduplicator(
            near_duplicate_threshold=near_duplicate_threshold
        )
        self._provenance = provenance_evaluator or SourceProvenanceEvaluator()
        self._chunker = chunker or PassageChunker()
        self._relevance = relevance_engine or HybridRelevanceEngine()
        self._claim_extractor = claim_extractor or RuleBasedClaimExtractor()
        self._claim_comparator = claim_comparator or DeterministicClaimComparator()
        self._dup_threshold = near_duplicate_threshold

    async def process_task_results(
        self,
        *,
        task: ResearchTask,
        session: ResearchSession,
        results: list[ToolExecutionResult],
        chunk_target_tokens: int = 180,
    ) -> NormalizedObservation:
        """Transform tool outputs into deduplicated sources, evidence, claims, and conflicts."""
        tool_names = list(dict.fromkeys(r.tool_name for r in results))
        disclosed_new: list[DisclosedSearchQuery] = []
        diagnostics_new: list[ProviderDiagnostic] = []
        warnings: list[str] = []

        search_hits: list[NormalizedSearchResult] = []
        raw_pairs: list[tuple[RetrievedDocument, str, str]] = []
        referenced_evidence_ids: list[str] = []

        for res in results:
            if not res.success:
                if res.error:
                    warnings.append(f"{res.tool_name}: {res.error}")
                continue
            if res.tool_name == "web_search":
                for raw_hit in res.data.get("hits", []):
                    search_hits.append(NormalizedSearchResult.model_validate(raw_hit))
                for raw_dq in res.data.get("disclosed_queries", []):
                    disclosed_new.append(DisclosedSearchQuery.model_validate(raw_dq))
                for raw_diag in res.data.get("provider_diagnostics", []):
                    diagnostics_new.append(ProviderDiagnostic.model_validate(raw_diag))
                san_q = res.data.get("sanitized_query")
                if san_q and san_q not in task.search_queries_used:
                    task.search_queries_used.append(san_q)
            elif res.tool_name == "url_retrieve":
                for pair in res.data.get("retrieved_pairs", []):
                    hit = NormalizedSearchResult.model_validate(pair["hit"])
                    doc = RetrievedDocument.model_validate(pair["document"])
                    raw_pairs.append((doc, hit.snippet, hit.provider))
            elif res.tool_name == "local_doc_retrieve":
                for raw_src in res.data.get("sources", []):
                    src_obj = SourceRecord.model_validate(raw_src)
                    doc = RetrievedDocument(
                        **src_obj.model_dump(
                            exclude={
                                "search_snippet",
                                "search_provider",
                                "is_duplicate",
                                "duplicate_of_source_id",
                                "duplicate_reason",
                                "duplicate_similarity",
                                "alias_urls",
                            }
                        )
                    )
                    raw_pairs.append(
                        (
                            doc,
                            src_obj.search_snippet or "",
                            src_obj.search_provider or "local_cache",
                        )
                    )
            elif res.tool_name == "evidence_search":
                for match in res.data.get("evidence", []):
                    ev_id = (match.get("evidence") or {}).get("evidence_id")
                    if ev_id:
                        referenced_evidence_ids.append(ev_id)

        # Track disclosed queries & diagnostics without duplicates
        known_dq_keys = {(d.provider, d.sanitized_query) for d in session.disclosed_queries}
        persisted_dq: list[DisclosedSearchQuery] = []
        for dq in disclosed_new:
            key = (dq.provider, dq.sanitized_query)
            if key not in known_dq_keys:
                known_dq_keys.add(key)
                session.disclosed_queries.append(dq)
                persisted_dq.append(dq)
        session.provider_diagnostics.extend(diagnostics_new)

        # Count search hits that had no successful/partial document fetch (snippet-only)
        fetched_urls = {
            doc.original_url
            for doc, _, _ in raw_pairs
            if doc.extraction_status in {ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL}
            and doc.main_content
        }
        known_usable_urls = {
            s.original_url
            for s in session.collected_sources
            if s.extraction_status in {ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL}
            and s.main_content
        } | {
            s.canonical_url
            for s in session.collected_sources
            if s.canonical_url
            and s.extraction_status in {ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL}
            and s.main_content
        }
        snippet_only_hits = sum(
            1
            for h in search_hits
            if h.url not in fetched_urls and (h.canonical_url or h.url) not in known_usable_urls
        )

        # Deduplicate batch first via Stage 4 MultiLevelDeduplicator, then against prior session sources
        dedup_res = self._deduplicator.deduplicate(raw_pairs)
        existing_primaries = [s for s in session.collected_sources if not s.is_duplicate]
        source_decisions: list[SourceDecisionRecord] = []
        new_sources: list[SourceRecord] = []
        candidate_primaries: list[SourceRecord] = []

        for src in dedup_res.all_sources:
            if src.is_duplicate:
                source_decisions.append(
                    SourceDecisionRecord(
                        source_id=src.source_id,
                        url=src.original_url,
                        domain=src.domain,
                        retained=False,
                        reason=f"duplicate ({src.duplicate_reason}) within batch of {src.duplicate_of_source_id}",
                        is_duplicate=True,
                    )
                )
                new_sources.append(src)
                continue

            prior_match, reason, sim = self._match_existing_source(src, existing_primaries)
            if prior_match is not None:
                src.is_duplicate = True
                src.duplicate_of_source_id = prior_match.source_id
                src.duplicate_reason = reason
                src.duplicate_similarity = sim
                if src.original_url not in prior_match.alias_urls:
                    prior_match.alias_urls.append(src.original_url)
                if prior_match.source_id not in task.source_ids:
                    task.source_ids.append(prior_match.source_id)
                source_decisions.append(
                    SourceDecisionRecord(
                        source_id=src.source_id,
                        url=src.original_url,
                        domain=src.domain,
                        retained=False,
                        reason=f"duplicate ({reason}) of existing source {prior_match.source_id}",
                        is_duplicate=True,
                    )
                )
                new_sources.append(src)
                continue

            if (
                src.extraction_status == ExtractionStatus.FAILED
                or not src.main_content
                or len(src.main_content.strip()) < 40
            ):
                source_decisions.append(
                    SourceDecisionRecord(
                        source_id=src.source_id,
                        url=src.original_url,
                        domain=src.domain,
                        retained=False,
                        reason=f"empty_or_failed_extraction ({src.extraction_status.value})",
                    )
                )
                new_sources.append(src)
                continue

            existing_primaries.append(src)
            new_sources.append(src)
            candidate_primaries.append(src)

        # Chunk & score retained primary documents against task question + objective keywords
        keywords = list(dict.fromkeys([*extract_keywords(task.question), *task.entities_keywords]))
        provenance_by_id = {
            src.source_id: self._provenance.build_provenance(src) for src in candidate_primaries
        }
        src_by_id = {src.source_id: src for src in candidate_primaries}

        all_chunks: list[TextChunk] = []
        for src in candidate_primaries:
            all_chunks.extend(self._chunker.chunk_document(src, target_tokens=chunk_target_tokens))

        doc_scores, breakdowns = await self._relevance.score_chunks(
            task.question,
            keywords,
            candidate_primaries,
            all_chunks,
        )

        # Filter completely off-topic noise when corpus has meaningfully scored pages
        has_relevant_page = any(score >= 0.08 for score in doc_scores.values())
        retained_ids: set[str] = set()
        for src in candidate_primaries:
            rel_score = doc_scores.get(src.source_id, 0.0)
            if has_relevant_page and rel_score < 0.03:
                source_decisions.append(
                    SourceDecisionRecord(
                        source_id=src.source_id,
                        url=src.original_url,
                        domain=src.domain,
                        retained=False,
                        reason="off_topic_low_relevance",
                        relevance_score=rel_score,
                    )
                )
                continue
            retained_ids.add(src.source_id)
            source_decisions.append(
                SourceDecisionRecord(
                    source_id=src.source_id,
                    url=src.original_url,
                    domain=src.domain,
                    retained=True,
                    reason="retained_relevant_document",
                    relevance_score=rel_score,
                )
            )
            if src.source_id not in task.source_ids:
                task.source_ids.append(src.source_id)

        raw_new_evidence: list[EvidenceItem] = []
        for chunk in all_chunks:
            if chunk.source_id not in retained_ids:
                continue
            src = src_by_id[chunk.source_id]
            bd = breakdowns.get(chunk.chunk_id)
            if bd is None:
                continue
            # Skip near-zero chunks inside a retained page if another chunk in the same page scored better
            ev = EvidenceItem(
                evidence_id=str(uuid.uuid4()),
                source_id=src.source_id,
                source_url=src.final_url or src.original_url,
                source_domain=src.domain,
                source_title=src.title,
                chunk_index=chunk.chunk_index,
                section_heading=chunk.section_heading,
                passage_text=chunk.passage_text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                token_estimate=chunk.token_estimate,
                relevance=bd,
                provenance=provenance_by_id[src.source_id],
                corroboration=CorroborationSignals(),
            )
            raw_new_evidence.append(ev)
            task.evidence_ids.append(ev.evidence_id)

        for ev_id in referenced_evidence_ids:
            if ev_id not in task.evidence_ids:
                task.evidence_ids.append(ev_id)

        # Merge sources and evidence cleanly into session
        known_sids = {s.source_id for s in session.collected_sources}
        for src in new_sources:
            if src.source_id not in known_sids:
                known_sids.add(src.source_id)
                session.collected_sources.append(src)

        new_claims: list[ExtractedClaim] = []
        if raw_new_evidence:
            new_claims = self._claim_extractor.extract_claims(raw_new_evidence)
            session.evidence_records.extend(raw_new_evidence)
            session.claims.extend(new_claims)

        # Re-compare ALL accumulated claims so cross-iteration conflicts & corroborations update cleanly
        all_conflicts, corroborated_groups = await self._claim_comparator.compare(session.claims)
        existing_conflict_pairs = {
            tuple(sorted((c.claim_a.claim_id, c.claim_b.claim_id))): c for c in session.conflicts
        }
        newly_detected_conflicts: list[ClaimConflict] = []
        merged_conflicts: list[ClaimConflict] = list(session.conflicts)
        for c in all_conflicts:
            key = tuple(sorted((c.claim_a.claim_id, c.claim_b.claim_id)))
            if key not in existing_conflict_pairs:
                existing_conflict_pairs[key] = c
                merged_conflicts.append(c)
                newly_detected_conflicts.append(c)
        session.conflicts = merged_conflicts

        if session.evidence_records:
            session.evidence_records = self._provenance.evaluate_corroboration(
                session.evidence_records,
                session.collected_sources,
                session.claims,
                session.conflicts,
                corroborated_groups,
            )

        session.source_decisions.extend(source_decisions)

        retained_count = sum(1 for d in source_decisions if d.retained)
        rejected_count = sum(1 for d in source_decisions if not d.retained)
        summary = (
            f"Processed {len(new_sources)} source(s): {retained_count} retained, {rejected_count} rejected; "
            f"{len(raw_new_evidence)} evidence passage(s), {len(new_claims)} claim(s), "
            f"{len(newly_detected_conflicts)} new conflict(s), {snippet_only_hits} snippet-only hit(s)."
        )
        return NormalizedObservation(
            task_id=task.task_id,
            iteration=session.current_iteration,
            tool_names=tool_names,
            disclosed_queries=persisted_dq,
            provider_diagnostics=diagnostics_new,
            new_sources=new_sources,
            new_evidence=raw_new_evidence,
            new_claims=new_claims,
            new_conflicts=newly_detected_conflicts,
            source_decisions=source_decisions,
            snippet_only_hits=snippet_only_hits,
            summary=summary,
            warnings=warnings,
        )

    def _match_existing_source(
        self,
        candidate: SourceRecord,
        existing_primaries: list[SourceRecord],
    ) -> tuple[SourceRecord | None, str | None, float | None]:
        """Compare candidate document against prior session sources for cross-iteration deduplication."""
        cand_exact = candidate.original_url.strip()
        cand_canon = (candidate.canonical_url or "").strip()
        cand_hash = candidate.content_hash
        usable = candidate.extraction_status in {
            ExtractionStatus.SUCCESS,
            ExtractionStatus.PARTIAL,
        }

        for ex in existing_primaries:
            if ex.original_url.strip() == cand_exact or cand_exact in ex.alias_urls:
                return ex, "exact_url", 1.0
            if cand_canon and ex.canonical_url and ex.canonical_url.strip() == cand_canon:
                return ex, "canonical_url", 1.0
            if usable and cand_hash and ex.content_hash and ex.content_hash == cand_hash:
                return ex, "exact_content_hash", 1.0

        if usable and candidate.main_content:
            best_sim = 0.0
            best_match: SourceRecord | None = None
            for ex in existing_primaries:
                if not ex.main_content:
                    continue
                sim = compute_shingle_similarity(candidate.main_content, ex.main_content)
                if sim >= self._dup_threshold and sim > best_sim:
                    best_sim = sim
                    best_match = ex
            if best_match is not None:
                return best_match, "near_duplicate", round(best_sim, 4)

        return None, None, None
