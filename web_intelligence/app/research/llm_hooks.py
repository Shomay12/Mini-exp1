"""Local LLM integration hooks for query planning, answer synthesis, and verification."""

from __future__ import annotations

import re
import time
from typing import Any

from web_intelligence.app.core.logging import get_logger
from web_intelligence.app.llm.base import UniversalLocalModelAdapter
from web_intelligence.app.llm.security import redact_llm_error
from web_intelligence.app.privacy.sanitizer import QuerySanitizer
from web_intelligence.app.query.planner import QueryPlanner
from web_intelligence.app.research.context_budget import AdaptiveContextBudgetManager
from web_intelligence.app.schemas.documents import SourceRecord
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CoverageAnalysis,
    EvidenceItem,
)
from web_intelligence.app.schemas.llm import ChatMessage, LLMInferenceRequest
from web_intelligence.app.schemas.research import (
    ContextAllocationPlan,
    LocalSynthesisResult,
    PlannedQuery,
    QueryPlan,
    VerificationResult,
)

logger = get_logger("research.llm_hooks")

_EVIDENCE_TAG_RE = re.compile(r"\[?(E\d+)\]?")
_SOURCE_ID_RE = re.compile(r"\bsource_id=([A-Za-z0-9\-_]+)\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a precision local research synthesizer. Answer strictly using the provided "
    "evidence passages labeled [E1], [E2], etc. Cite every factual sentence inline using its "
    "bracketed evidence tag like [E1] or [E2]. If sources conflict in DETECTED SOURCE CONFLICTS, "
    "state both sides explicitly and cite their evidence tags. Do not invent facts, dates, "
    "numbers, URLs, or unseen [E#] tags. If evidence is insufficient, say so clearly."
)


class LocalQueryPlanner(QueryPlanner):
    """Uses an active local LLM for query decomposition with deterministic fallback."""

    def __init__(
        self,
        fallback_planner: QueryPlanner,
        llm_adapter: UniversalLocalModelAdapter | None = None,
    ) -> None:
        self._fallback = fallback_planner
        self._adapter = llm_adapter
        self.name = (
            f"local_llm:{llm_adapter.capabilities().model_name}"
            if llm_adapter is not None
            else fallback_planner.name
        )

    async def plan(self, raw_query: str) -> QueryPlan:
        baseline = await self._fallback.plan(raw_query)
        if self._adapter is None or not self._adapter.config.model.strip():
            return baseline

        caps = self._adapter.capabilities()
        system_msg = ChatMessage(
            role="system",
            content=(
                "Decompose the sanitized research query into 2 to 4 focused non-overlapping web "
                'search queries. Return JSON only: {"subqueries": [{"query": "...", '
                '"subtopic": "...", "rationale": "..."}]}. Do not include PII or URLs.'
            ),
        )
        user_msg = ChatMessage(
            role="user",
            content=(
                f"Sanitized query: {baseline.normalized_query}\n"
                f"Detected intent: {baseline.intent.value}\n"
                f"Keywords: {', '.join(baseline.keywords[:8])}"
            ),
        )
        try:
            payload = await self._adapter.generate_json(
                [system_msg, user_msg],
                max_tokens=min(320, caps.recommended_output_tokens),
                temperature=0.1,
            )
        except Exception as exc:
            logger.warning(
                "Local LLM query planning failed; using deterministic fallback",
                extra={"extra_data": {"error": redact_llm_error(exc)}},
            )
            return baseline

        raw_items: Any = None
        if isinstance(payload, dict):
            raw_items = payload.get("subqueries") or payload.get("queries")
        elif isinstance(payload, list):
            raw_items = payload

        if not isinstance(raw_items, list):
            return baseline

        planned: list[PlannedQuery] = []
        seen: set[str] = set()
        sanitizer = QuerySanitizer(max_query_length=160)
        for entry in raw_items[:4]:
            if isinstance(entry, str):
                q_raw, subtopic, rationale = entry, "llm-subquery", "Local LLM query decomposition."
            elif isinstance(entry, dict):
                q_raw = str(entry.get("query") or "").strip()
                subtopic = str(entry.get("subtopic") or "llm-subquery").strip()[:60]
                rationale = str(entry.get("rationale") or "Local LLM query decomposition.").strip()[:180]
            else:
                continue

            clean_q = sanitizer.sanitize(q_raw).sanitized_query
            if len(clean_q) < 3 or clean_q.lower() in seen:
                continue
            seen.add(clean_q.lower())
            planned.append(
                PlannedQuery(
                    query=clean_q[:160],
                    intent=baseline.intent,
                    subtopic=subtopic or "llm-subquery",
                    rationale=rationale or "Local LLM query decomposition.",
                )
            )

        if not planned:
            return baseline

        return baseline.model_copy(
            update={
                "planned_queries": planned,
                "planner_name": f"local_llm:{caps.model_name or self._adapter.config.model}",
            }
        )


class LocalEvidenceVerifier:
    """Verifies answer citations, numeric grounding, and acknowledgement of source conflicts."""

    @staticmethod
    def _extract_citation_tags(answer: str) -> list[str]:
        tags: list[str] = []
        for match in re.finditer(r"\[(E\d+(?:\s*,\s*E\d+)*)\]", answer):
            group = match.group(1)
            for tag in _EVIDENCE_TAG_RE.findall(group):
                if tag not in tags:
                    tags.append(tag)
        return tags

    def verify(
        self,
        answer: str,
        ref_map: dict[str, EvidenceItem],
        all_sources: list[SourceRecord],
        conflicts: list[ClaimConflict],
    ) -> VerificationResult:
        """Audit generated synthesis against provided evidence cards and detected conflicts."""
        trimmed = answer.strip()
        if not trimmed:
            return VerificationResult(
                verified=False,
                notes=["Generated answer is empty."],
            )

        valid_source_ids = {src.source_id for src in all_sources}
        cited_tags = self._extract_citation_tags(trimmed)
        cited_source_ids: list[str] = []
        invalid_citations: list[str] = []

        for tag in cited_tags:
            item = ref_map.get(tag)
            if item is None:
                if tag not in invalid_citations:
                    invalid_citations.append(tag)
            elif item.source_id not in cited_source_ids:
                cited_source_ids.append(item.source_id)

        for sid in _SOURCE_ID_RE.findall(trimmed):
            if sid not in valid_source_ids and sid not in invalid_citations:
                invalid_citations.append(sid)

        # Check factual sentences carrying specific numbers/years have citation & evidence support
        corpus_numbers: set[str] = set()
        for ev in ref_map.values():
            for num in _NUMERIC_TOKEN_RE.findall(ev.passage_text):
                corpus_numbers.add(num)
                corpus_numbers.add(num.replace(",", ""))

        unsupported_claims: list[str] = []
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(trimmed) if len(s.strip()) > 12]
        for sentence in sentences:
            tags_in_sentence = self._extract_citation_tags(sentence)
            # Ignore bracketed tag numbers like E1/E2 or small list markers
            content_no_tags = re.sub(r"\[E\d+(?:\s*,\s*E\d+)*\]", "", sentence)
            factual_nums = _NUMERIC_TOKEN_RE.findall(content_no_tags)
            ungrounded_nums = [
                n
                for n in factual_nums
                if n.replace(",", "") not in corpus_numbers and n not in corpus_numbers and len(n) >= 2
            ]
            if ungrounded_nums:
                unsupported_claims.append(
                    f"Numeric value(s) {', '.join(ungrounded_nums)} not observed in allocated evidence: {sentence[:140]}"
                )
            elif factual_nums and not tags_in_sentence and ref_map:
                unsupported_claims.append(
                    f"Quantitative statement lacks inline [E#] citation: {sentence[:140]}"
                )

        if not cited_tags and ref_map:
            unsupported_claims.append("Response contains no [E#] evidence citations.")

        acknowledged_conflicts: list[str] = []
        unacknowledged_conflicts: list[str] = []
        answer_lower = trimmed.lower()
        conflict_markers = ("conflict", "differ", "disagree", "contradict", "however", "whereas", "versus", "vs.")
        mentions_divergence = any(marker in answer_lower for marker in conflict_markers)

        for conflict in conflicts:
            val_a = (conflict.claim_a.normalized_value or "").lower()
            val_b = (conflict.claim_b.normalized_value or "").lower()
            both_values_shown = bool(val_a and val_b and val_a in answer_lower and val_b in answer_lower)
            if both_values_shown or mentions_divergence:
                acknowledged_conflicts.append(conflict.conflict_id)
            else:
                unacknowledged_conflicts.append(conflict.conflict_id)

        notes: list[str] = []
        if not cited_tags and ref_map:
            notes.append("No inline [E#] evidence citations detected.")
        if invalid_citations:
            notes.append(f"Detected {len(invalid_citations)} unknown citation(s).")
        if unsupported_claims:
            notes.append(f"Flagged {len(unsupported_claims)} potentially unsupported claim(s).")
        if unacknowledged_conflicts:
            notes.append(
                f"{len(unacknowledged_conflicts)} source conflict(s) were not explicitly acknowledged."
            )
        if not notes:
            notes.append("All cited references match allocated evidence and conflicts were checked.")

        verified = (
            not invalid_citations
            and not unsupported_claims
            and not unacknowledged_conflicts
        )
        return VerificationResult(
            verified=verified,
            cited_source_ids=cited_source_ids,
            invalid_citations=invalid_citations,
            unsupported_claims=unsupported_claims,
            acknowledged_conflicts=acknowledged_conflicts,
            unacknowledged_conflicts=unacknowledged_conflicts,
            notes=notes,
        )


class LocalAnswerGenerator:
    """Generates evidence-grounded synthesis using the AdaptiveContextBudgetManager."""

    def __init__(
        self,
        llm_adapter: UniversalLocalModelAdapter,
        *,
        budget_manager: AdaptiveContextBudgetManager | None = None,
        verifier: LocalEvidenceVerifier | None = None,
    ) -> None:
        self._adapter = llm_adapter
        self._budget = budget_manager or AdaptiveContextBudgetManager()
        self._verifier = verifier or LocalEvidenceVerifier()

    async def generate_and_verify(
        self,
        *,
        query: str,
        evidence_items: list[EvidenceItem],
        all_sources: list[SourceRecord],
        conflicts: list[ClaimConflict],
        coverage: CoverageAnalysis,
    ) -> tuple[ContextAllocationPlan, LocalSynthesisResult]:
        """Allocate context dynamically, run local synthesis, and verify attribution."""
        started = time.monotonic()
        caps = self._adapter.capabilities()
        raw_secret = (
            self._adapter.config.api_key.get_secret_value()
            if self._adapter.config.api_key is not None
            else None
        )

        selected_items, allocation_plan = self._budget.allocate(
            evidence_items,
            caps,
            system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
            query_prompt=query,
            conflicts=conflicts,
        )
        context_block, ref_map = self._budget.format_evidence_context(
            selected_items, conflicts, coverage
        )

        if not selected_items:
            synthesis = LocalSynthesisResult(
                connection_id=self._adapter.config.id,
                model_name=caps.model_name or self._adapter.config.model or "unselected",
                provider=self._adapter.config.provider,
                answer="Insufficient evidence could be fitted within the configured model context budget.",
                context_budget=allocation_plan,
                verification=VerificationResult(
                    verified=False,
                    notes=["No evidence passages fit within available context budget."],
                ),
                latency_ms=0,
                warning="No evidence allocated.",
            )
            return allocation_plan, synthesis

        user_prompt = (
            f"Research Question: {query}\n\n"
            f"Allocated Evidence ({len(selected_items)} passages, "
            f"{allocation_plan.allocated_tokens}/{allocation_plan.available_evidence_tokens} tokens):\n\n"
            f"{context_block}\n\n"
            "Write a concise, well-structured, evidence-grounded synthesis citing [E1], [E2], etc."
        )

        try:
            response = await self._adapter.chat(
                LLMInferenceRequest(
                    messages=[
                        ChatMessage(role="system", content=_SYNTHESIS_SYSTEM_PROMPT),
                        ChatMessage(role="user", content=user_prompt),
                    ],
                    model=self._adapter.config.model or None,
                    temperature=0.15,
                    max_tokens=allocation_plan.reserved_output_tokens,
                    stream=False,
                )
            )
            answer_text = response.content.strip()
            verification = self._verifier.verify(
                answer_text,
                ref_map,
                all_sources,
                conflicts,
            )
            cited_evidence_ids = [
                ref_map[tag].evidence_id
                for tag in self._verifier._extract_citation_tags(answer_text)
                if tag in ref_map
            ]
            # Preserve insertion order without duplicates
            seen_ev: set[str] = set()
            unique_cited_evidence = [
                eid for eid in cited_evidence_ids if not (eid in seen_ev or seen_ev.add(eid))
            ]
            synthesis = LocalSynthesisResult(
                connection_id=self._adapter.config.id,
                model_name=response.model or caps.model_name or self._adapter.config.model,
                provider=self._adapter.config.provider,
                answer=answer_text,
                cited_evidence_ids=unique_cited_evidence,
                cited_source_ids=verification.cited_source_ids,
                context_budget=allocation_plan,
                verification=verification,
                latency_ms=max(0, int((time.monotonic() - started) * 1000)),
            )
            return allocation_plan, synthesis
        except Exception as exc:
            safe_err = redact_llm_error(exc, secrets=[raw_secret])
            logger.warning(
                "Local answer generation failed; returning retrieved Stage 4 evidence package",
                extra={"extra_data": {"error": safe_err, "connection_id": self._adapter.config.id}},
            )
            synthesis = LocalSynthesisResult(
                connection_id=self._adapter.config.id,
                model_name=caps.model_name or self._adapter.config.model or "unselected",
                provider=self._adapter.config.provider,
                answer="",
                context_budget=allocation_plan,
                verification=VerificationResult(
                    verified=False,
                    notes=[f"Local model generation failed: {safe_err}"],
                ),
                latency_ms=max(0, int((time.monotonic() - started) * 1000)),
                warning=f"Local model generation failed: {safe_err}",
            )
            return allocation_plan, synthesis
