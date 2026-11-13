"""Context compression strategies with strict provenance, entity, date, and number preservation."""

from __future__ import annotations

import inspect
import re
from typing import Any

from web_intelligence.app.context.models import (
    CompressedEvidenceUnit,
    CompressionResult,
    CompressionStrategy,
    SelectedEvidenceRecord,
    StructuredFactEntry,
)
from web_intelligence.app.context.token_estimator import (
    BaseContextTokenEstimator,
    ConservativeHeuristicEstimator,
)
from web_intelligence.app.llm.security import redact_llm_error
from web_intelligence.app.schemas.evidence import ClaimConflict, EvidenceItem, ExtractedClaim
from web_intelligence.app.schemas.llm import ChatMessage, LLMInferenceRequest

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[a-z0-9]{2,}")
_YEAR_OR_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9\[])\d+(?:[.,]\d+)?(?:\s?%|x|ms|s|km|GB|MB|B|M|K)?\b")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9\-]{1,}(?:\s+[A-Z][a-zA-Z0-9\-]{1,}){0,3}|[A-Z]{2,}[0-9]*)\b")
_CITATION_TAG_RE = re.compile(r"\[E\d+\]")


def _extract_dates(text: str) -> list[str]:
    return sorted({m.group(0).strip() for m in _YEAR_OR_DATE_RE.finditer(text)})


def _extract_numbers(text: str) -> list[str]:
    # Exclude citation tags like [E1] already avoided by regex, deduplicate preserving normalized value
    nums: list[str] = []
    seen: set[str] = set()
    for m in _NUMBER_RE.finditer(text):
        val = m.group(0).strip()
        if val and val not in seen:
            seen.add(val)
            nums.append(val)
    return nums


_ENTITY_IGNORE = frozenset({
    "The", "This", "That", "These", "Those", "However", "Although", "When", "Where",
    "In", "On", "At", "For", "By", "According", "Between", "Since", "While",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
})

_STOP_WORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "between",
    "according", "over", "under", "was", "were", "are", "is", "in", "on", "to",
    "of", "by", "at", "as", "an", "or", "per",
})


def _extract_entities(text: str, *, limit: int = 12) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in _ENTITY_RE.finditer(text):
        ent = m.group(0).strip()
        if ent in _ENTITY_IGNORE or (ent.startswith("E") and ent[1:].isdigit()):
            continue
        if ent not in seen:
            seen.add(ent)
            found.append(ent)
        if len(found) >= limit:
            break
    return found


def _core_tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if t not in _STOP_WORDS}


def _sentence_similarity(a: str, b: str) -> float:
    ta = _core_tokens(a)
    tb = _core_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    jaccard = inter / len(ta | tb)
    overlap_min = inter / min(len(ta), len(tb))
    if jaccard >= 0.55 and overlap_min >= 0.70:
        return overlap_min
    return jaccard


class ContextCompressor:
    """Compresses selected evidence via Extractive, Deduplication, Structured, or Local-LLM modes."""

    def __init__(self, estimator: BaseContextTokenEstimator | None = None) -> None:
        self._estimator = estimator or ConservativeHeuristicEstimator()

    def compress(
        self,
        selected_evidence: list[SelectedEvidenceRecord],
        *,
        strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE,
        query: str = "",
        target_tokens: int | None = None,
        claims: list[ExtractedClaim] | None = None,
        conflicts: list[ClaimConflict] | None = None,
    ) -> CompressionResult:
        """Execute deterministic compression (NONE, EXTRACTIVE, DEDUPLICATION, STRUCTURED)."""
        if not selected_evidence:
            return CompressionResult(
                strategy_requested=strategy,
                strategy_applied=strategy,
                units=[],
                original_total_tokens=0,
                compressed_total_tokens=0,
                compression_ratio=1.0,
                deduplicated_count=0,
                independent_sources_preserved=0,
                provenance_preserved=True,
            )

        if strategy == CompressionStrategy.MODEL_ASSISTED:
            # Synchronous caller asked for MODEL_ASSISTED without async adapter; fall back cleanly.
            res = self._compress_structured(
                selected_evidence,
                query=query,
                target_tokens=target_tokens,
                claims=claims,
                conflicts=conflicts,
            )
            res.strategy_requested = CompressionStrategy.MODEL_ASSISTED
            res.model_assisted_fallback_reason = (
                "No asynchronous local model adapter provided; used structured deterministic compression."
            )
            res.warnings.append(res.model_assisted_fallback_reason)
            return res

        if strategy == CompressionStrategy.NONE:
            return self._passthrough(selected_evidence)
        if strategy == CompressionStrategy.EXTRACTIVE:
            return self._compress_extractive(
                selected_evidence, query=query, target_tokens=target_tokens
            )
        if strategy == CompressionStrategy.DEDUPLICATION:
            return self._compress_deduplication(
                selected_evidence, target_tokens=target_tokens
            )
        return self._compress_structured(
            selected_evidence,
            query=query,
            target_tokens=target_tokens,
            claims=claims,
            conflicts=conflicts,
        )

    async def compress_async(
        self,
        selected_evidence: list[SelectedEvidenceRecord],
        *,
        strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE,
        query: str = "",
        target_tokens: int | None = None,
        claims: list[ExtractedClaim] | None = None,
        conflicts: list[ClaimConflict] | None = None,
        adapter: Any = None,
        allow_model_assisted: bool = False,
    ) -> CompressionResult:
        """Execute compression, supporting optional local model-assisted summarization with validation."""
        if strategy != CompressionStrategy.MODEL_ASSISTED:
            return self.compress(
                selected_evidence,
                strategy=strategy,
                query=query,
                target_tokens=target_tokens,
                claims=claims,
                conflicts=conflicts,
            )

        if not selected_evidence:
            return self.compress([], strategy=strategy)

        if not allow_model_assisted or adapter is None:
            fallback = self._compress_structured(
                selected_evidence,
                query=query,
                target_tokens=target_tokens,
                claims=claims,
                conflicts=conflicts,
            )
            reason = (
                "Model-assisted compression disabled by configuration; fell back to structured compression."
                if not allow_model_assisted
                else "No active local model adapter available for model-assisted compression; fell back to structured compression."
            )
            fallback.strategy_requested = CompressionStrategy.MODEL_ASSISTED
            fallback.model_assisted_fallback_reason = reason
            fallback.warnings.append(reason)
            return fallback

        return await self._compress_model_assisted(
            selected_evidence,
            query=query,
            target_tokens=target_tokens,
            claims=claims or [],
            conflicts=conflicts or [],
            adapter=adapter,
        )

    def _passthrough(self, selected: list[SelectedEvidenceRecord]) -> CompressionResult:
        units: list[CompressedEvidenceUnit] = []
        orig_total = 0
        comp_total = 0
        domains: set[str] = set()
        for item in selected:
            formatted = f"{item.citation_label} ({item.source_domain} | {item.source_url}): {item.passage_text}"
            c_tok = self._estimator.estimate(formatted).tokens
            o_tok = item.original_token_count or self._estimator.estimate(item.passage_text).tokens
            orig_total += o_tok
            comp_total += c_tok
            domains.add(item.source_domain)
            units.append(
                CompressedEvidenceUnit(
                    unit_id=f"unit-{item.evidence_id}",
                    strategy=CompressionStrategy.NONE,
                    compressed_text=formatted,
                    original_evidence_ids=[item.evidence_id],
                    supporting_source_ids=[item.source_id],
                    supporting_source_urls=[item.source_url],
                    supporting_domains=[item.source_domain],
                    citation_labels=[item.citation_label],
                    original_tokens=o_tok,
                    compressed_tokens=c_tok,
                    provenance_verified=True,
                )
            )
        ratio = round(comp_total / orig_total, 4) if orig_total > 0 else 1.0
        return CompressionResult(
            strategy_requested=CompressionStrategy.NONE,
            strategy_applied=CompressionStrategy.NONE,
            units=units,
            original_total_tokens=orig_total,
            compressed_total_tokens=comp_total,
            compression_ratio=ratio,
            deduplicated_count=0,
            independent_sources_preserved=len(domains),
            provenance_preserved=True,
        )

    def _compress_extractive(
        self,
        selected: list[SelectedEvidenceRecord],
        *,
        query: str,
        target_tokens: int | None,
    ) -> CompressionResult:
        query_terms = set(_WORD_RE.findall(query.lower()))
        orig_total = sum(
            item.original_token_count or self._estimator.estimate(item.passage_text).tokens
            for item in selected
        )

        # Allocate per-unit token caps proportional to importance, giving conflict evidence a floor
        effective_target = target_tokens or orig_total
        per_item_budget = max(48, effective_target // max(1, len(selected)))

        units: list[CompressedEvidenceUnit] = []
        comp_total = 0
        domains: set[str] = set()

        for item in selected:
            raw_sentences = [
                s.strip() for s in _SENTENCE_SPLIT_RE.split(item.passage_text) if s.strip()
            ]
            if not raw_sentences:
                raw_sentences = [item.passage_text.strip()]

            # Break excessively long unpunctuated run-on blocks into extractable verbatim spans
            sentences: list[str] = []
            for sent in raw_sentences:
                words = sent.split()
                if len(words) > 45:
                    for i in range(0, len(words), 35):
                        sentences.append(" ".join(words[i : i + 35]))
                else:
                    sentences.append(sent)

            scored: list[tuple[int, float, str]] = []
            for idx, sent in enumerate(sentences):
                words = set(_WORD_RE.findall(sent.lower()))
                overlap = (len(words & query_terms) / max(1, len(query_terms))) if query_terms else 0.0
                has_num_or_date = 0.25 if (_NUMBER_RE.search(sent) or _YEAR_OR_DATE_RE.search(sent)) else 0.0
                lead_bonus = 0.15 if idx == 0 else 0.0
                conflict_bonus = 0.20 if item.is_conflict_evidence else 0.0
                score = overlap + has_num_or_date + lead_bonus + conflict_bonus
                scored.append((idx, score, sent))

            # Pick top-scored sentences while fitting item cap; keep at least top sentence verbatim!
            ranked = sorted(scored, key=lambda row: (-row[1], row[0]))
            chosen: list[tuple[int, str]] = []
            prefix = f"{item.citation_label} ({item.source_domain}): "
            running_tokens = self._estimator.estimate(prefix).tokens

            item_cap = int(per_item_budget * (1.25 if item.is_conflict_evidence else 1.0))
            for idx, _sc, sent in ranked:
                sent_tok = self._estimator.estimate(sent).tokens + 1
                if not chosen or (running_tokens + sent_tok <= item_cap):
                    chosen.append((idx, sent))
                    running_tokens += sent_tok

            chosen.sort(key=lambda row: row[0])
            compressed_body = " ".join(sent for _, sent in chosen)
            formatted = f"{prefix}{compressed_body}"
            c_tok = self._estimator.estimate(formatted).tokens
            o_tok = item.original_token_count or self._estimator.estimate(item.passage_text).tokens
            comp_total += c_tok
            domains.add(item.source_domain)

            units.append(
                CompressedEvidenceUnit(
                    unit_id=f"ext-{item.evidence_id}",
                    strategy=CompressionStrategy.EXTRACTIVE,
                    compressed_text=formatted,
                    original_evidence_ids=[item.evidence_id],
                    supporting_source_ids=[item.source_id],
                    supporting_source_urls=[item.source_url],
                    supporting_domains=[item.source_domain],
                    citation_labels=[item.citation_label],
                    original_tokens=o_tok,
                    compressed_tokens=c_tok,
                    provenance_verified=True,
                )
            )

        ratio = round(comp_total / orig_total, 4) if orig_total > 0 else 1.0
        return CompressionResult(
            strategy_requested=CompressionStrategy.EXTRACTIVE,
            strategy_applied=CompressionStrategy.EXTRACTIVE,
            units=units,
            original_total_tokens=orig_total,
            compressed_total_tokens=comp_total,
            compression_ratio=ratio,
            deduplicated_count=0,
            independent_sources_preserved=len(domains),
            provenance_preserved=True,
        )

    def _compress_deduplication(
        self,
        selected: list[SelectedEvidenceRecord],
        *,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Merge repeated claims/sentences while preserving all independent supporting sources."""
        del target_tokens  # preserves independent sources rather than silently truncating
        groups: list[dict[str, Any]] = []
        dedup_count = 0

        for item in selected:
            matched_group: dict[str, Any] | None = None
            # Contradictory items are kept distinct so opposing claims remain explicit.
            if not item.is_conflict_evidence:
                for grp in groups:
                    if grp["is_conflict"]:
                        continue
                    if _sentence_similarity(grp["canonical_text"], item.passage_text) >= 0.72:
                        matched_group = grp
                        break

            if matched_group is not None:
                dedup_count += 1
                matched_group["evidence_ids"].append(item.evidence_id)
                if item.source_id not in matched_group["source_ids"]:
                    matched_group["source_ids"].append(item.source_id)
                if item.source_url not in matched_group["source_urls"]:
                    matched_group["source_urls"].append(item.source_url)
                if item.source_domain not in matched_group["domains"]:
                    matched_group["domains"].append(item.source_domain)
                matched_group["citations"].append(item.citation_label)
                matched_group["orig_tokens"] += (
                    item.original_token_count or self._estimator.estimate(item.passage_text).tokens
                )
            else:
                groups.append(
                    {
                        "canonical_text": item.passage_text.strip(),
                        "evidence_ids": [item.evidence_id],
                        "source_ids": [item.source_id],
                        "source_urls": [item.source_url],
                        "domains": [item.source_domain],
                        "citations": [item.citation_label],
                        "orig_tokens": item.original_token_count
                        or self._estimator.estimate(item.passage_text).tokens,
                        "is_conflict": item.is_conflict_evidence,
                    }
                )

        units: list[CompressedEvidenceUnit] = []
        orig_total = 0
        comp_total = 0
        all_domains: set[str] = set()

        for idx, grp in enumerate(groups, start=1):
            cite_prefix = "".join(grp["citations"])
            domain_list = ", ".join(grp["domains"])
            text = f"{cite_prefix} ({domain_list}): {grp['canonical_text']}"
            c_tok = self._estimator.estimate(text).tokens
            o_tok = grp["orig_tokens"]
            orig_total += o_tok
            comp_total += c_tok
            all_domains.update(grp["domains"])
            units.append(
                CompressedEvidenceUnit(
                    unit_id=f"dedup-{idx}",
                    strategy=CompressionStrategy.DEDUPLICATION,
                    compressed_text=text,
                    original_evidence_ids=grp["evidence_ids"],
                    supporting_source_ids=grp["source_ids"],
                    supporting_source_urls=grp["source_urls"],
                    supporting_domains=grp["domains"],
                    citation_labels=grp["citations"],
                    original_tokens=o_tok,
                    compressed_tokens=c_tok,
                    provenance_verified=True,
                )
            )

        ratio = round(comp_total / orig_total, 4) if orig_total > 0 else 1.0
        return CompressionResult(
            strategy_requested=CompressionStrategy.DEDUPLICATION,
            strategy_applied=CompressionStrategy.DEDUPLICATION,
            units=units,
            original_total_tokens=orig_total,
            compressed_total_tokens=comp_total,
            compression_ratio=ratio,
            deduplicated_count=dedup_count,
            independent_sources_preserved=len(all_domains),
            provenance_preserved=True,
        )

    def _compress_structured(
        self,
        selected: list[SelectedEvidenceRecord],
        *,
        query: str,
        target_tokens: int | None,
        claims: list[ExtractedClaim] | None,
        conflicts: list[ClaimConflict] | None,
    ) -> CompressionResult:
        del query, target_tokens
        claims = claims or []
        conflicts = conflicts or []

        claims_by_ev: dict[str, list[ExtractedClaim]] = {}
        for c in claims:
            claims_by_ev.setdefault(c.evidence_id, []).append(c)

        conflict_ev_ids: set[str] = set()
        for cf in conflicts:
            conflict_ev_ids.add(cf.claim_a.evidence_id)
            conflict_ev_ids.add(cf.claim_b.evidence_id)

        units: list[CompressedEvidenceUnit] = []
        orig_total = 0
        comp_total = 0
        domains: set[str] = set()

        for idx, item in enumerate(selected, start=1):
            matched_claims = claims_by_ev.get(item.evidence_id, [])
            dates = _extract_dates(item.passage_text)
            numbers = _extract_numbers(item.passage_text)
            entities = _extract_entities(item.passage_text)
            facts: list[StructuredFactEntry] = []

            if matched_claims:
                for c_idx, c in enumerate(matched_claims, start=1):
                    facts.append(
                        StructuredFactEntry(
                            fact_id=c.claim_id or f"fact-{idx}-{c_idx}",
                            claim_text=c.statement_text,
                            entity=c.entity,
                            attribute=c.attribute,
                            value=f"{c.normalized_value}{' ' + c.unit if c.unit else ''}".strip(),
                            dates=_extract_dates(c.statement_text) or dates,
                            numbers=_extract_numbers(c.statement_text) or numbers,
                            entities=[c.entity] if c.entity else entities,
                            source_refs=[item.citation_label, item.source_url],
                            evidence_ids=[item.evidence_id],
                            source_urls=[item.source_url],
                            is_conflicted=item.is_conflict_evidence or item.evidence_id in conflict_ev_ids,
                        )
                    )
            else:
                # Distill key factual sentence preserving dates, numbers, entities
                sentences = [
                    s.strip() for s in _SENTENCE_SPLIT_RE.split(item.passage_text) if s.strip()
                ]
                lead = sentences[0] if sentences else item.passage_text.strip()
                if len(sentences) > 1 and not (_NUMBER_RE.search(lead) or _YEAR_OR_DATE_RE.search(lead)):
                    for cand in sentences[1:]:
                        if _NUMBER_RE.search(cand) or _YEAR_OR_DATE_RE.search(cand):
                            lead = f"{lead} {cand}"
                            break
                facts.append(
                    StructuredFactEntry(
                        fact_id=f"fact-{item.evidence_id}",
                        claim_text=lead,
                        entity=entities[0] if entities else None,
                        dates=dates,
                        numbers=numbers,
                        entities=entities,
                        source_refs=[item.citation_label, item.source_url],
                        evidence_ids=[item.evidence_id],
                        source_urls=[item.source_url],
                        is_conflicted=item.is_conflict_evidence or item.evidence_id in conflict_ev_ids,
                    )
                )

            fact_lines: list[str] = []
            for f in facts:
                meta_parts: list[str] = []
                if f.dates:
                    meta_parts.append(f"dates={','.join(f.dates[:3])}")
                if f.numbers:
                    meta_parts.append(f"vals={','.join(f.numbers[:4])}")
                if f.entities:
                    meta_parts.append(f"entities={','.join(f.entities[:4])}")
                meta_suffix = f" ({'; '.join(meta_parts)})" if meta_parts else ""
                tag = "[CONFLICT] " if f.is_conflicted else ""
                fact_lines.append(
                    f"- {item.citation_label} {tag}{f.claim_text}{meta_suffix} [{item.source_domain}]"
                )

            formatted = "\n".join(fact_lines)
            o_tok = item.original_token_count or self._estimator.estimate(item.passage_text).tokens
            c_tok = self._estimator.estimate(formatted).tokens
            orig_total += o_tok
            comp_total += c_tok
            domains.add(item.source_domain)

            units.append(
                CompressedEvidenceUnit(
                    unit_id=f"struct-{item.evidence_id}",
                    strategy=CompressionStrategy.STRUCTURED,
                    compressed_text=formatted,
                    original_evidence_ids=[item.evidence_id],
                    supporting_source_ids=[item.source_id],
                    supporting_source_urls=[item.source_url],
                    supporting_domains=[item.source_domain],
                    citation_labels=[item.citation_label],
                    original_tokens=o_tok,
                    compressed_tokens=c_tok,
                    provenance_verified=True,
                    structured_facts=facts,
                )
            )

        ratio = round(comp_total / orig_total, 4) if orig_total > 0 else 1.0
        return CompressionResult(
            strategy_requested=CompressionStrategy.STRUCTURED,
            strategy_applied=CompressionStrategy.STRUCTURED,
            units=units,
            original_total_tokens=orig_total,
            compressed_total_tokens=comp_total,
            compression_ratio=ratio,
            deduplicated_count=0,
            independent_sources_preserved=len(domains),
            provenance_preserved=True,
        )

    async def _compress_model_assisted(
        self,
        selected: list[SelectedEvidenceRecord],
        *,
        query: str,
        target_tokens: int | None,
        claims: list[ExtractedClaim],
        conflicts: list[ClaimConflict],
        adapter: Any,
    ) -> CompressionResult:
        orig_total = sum(
            item.original_token_count or self._estimator.estimate(item.passage_text).tokens
            for item in selected
        )
        source_block = "\n".join(
            f"{item.citation_label} ({item.source_domain} | {item.source_url}): {item.passage_text}"
            for item in selected
        )
        prompt = (
            "Summarize the evidence passages strictly preserving every relevant [E#] citation tag, "
            "exact numerical values, dates, named entities, and conflicting viewpoints without inventing facts.\n"
            f"Query: {query or 'Research synthesis'}\n\n"
            f"Evidence:\n{source_block}"
        )

        max_out = min(1024, max(192, target_tokens or 512))
        raw_summary = ""
        try:
            chat_fn = getattr(adapter, "chat", None)
            generate_fn = getattr(adapter, "generate", None)
            if callable(chat_fn):
                messages = [
                    ChatMessage(
                        role="system",
                        content=(
                            "You are a precision evidence compressor. Preserve all [E#] citations, "
                            "numbers, dates, and entity names verbatim."
                        ),
                    ),
                    ChatMessage(role="user", content=prompt),
                ]
                sig = inspect.signature(chat_fn)
                params = [p for p in sig.parameters.keys() if p != "self"]
                if params and params[0] == "request":
                    model_id = getattr(getattr(adapter, "config", None), "model", None) or "local-model"
                    req = LLMInferenceRequest(
                        model=model_id,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=max_out,
                    )
                    maybe = chat_fn(req)
                else:
                    maybe = chat_fn(messages, temperature=0.0, max_tokens=max_out)
                resp = await maybe if inspect.isawaitable(maybe) else maybe
                raw_summary = getattr(resp, "content", str(resp))
            elif callable(generate_fn):
                maybe = generate_fn(prompt, max_tokens=max_out)
                raw_summary = str(await maybe if inspect.isawaitable(maybe) else maybe)
            else:
                raise RuntimeError("Local model adapter does not expose chat() or generate().")
        except Exception as exc:
            safe_err = redact_llm_error(exc)
            return self._fallback_after_model_failure(
                selected,
                query=query,
                target_tokens=target_tokens,
                claims=claims,
                conflicts=conflicts,
                reason=f"Model-assisted compression failed ({safe_err}); reverted to structured compression.",
            )

        validation_issues = self._validate_model_summary_integrity(
            raw_summary, selected=selected, claims=claims
        )
        if validation_issues:
            reason = (
                "Model-assisted summary failed integrity validation ("
                + "; ".join(validation_issues[:3])
                + "); reverted to structured compression."
            )
            return self._fallback_after_model_failure(
                selected,
                query=query,
                target_tokens=target_tokens,
                claims=claims,
                conflicts=conflicts,
                reason=reason,
            )

        comp_tokens = self._estimator.estimate(raw_summary.strip()).tokens
        unit = CompressedEvidenceUnit(
            unit_id="model-assisted-summary",
            strategy=CompressionStrategy.MODEL_ASSISTED,
            compressed_text=raw_summary.strip(),
            original_evidence_ids=[item.evidence_id for item in selected],
            supporting_source_ids=list(dict.fromkeys(item.source_id for item in selected)),
            supporting_source_urls=list(dict.fromkeys(item.source_url for item in selected)),
            supporting_domains=list(dict.fromkeys(item.source_domain for item in selected)),
            citation_labels=[item.citation_label for item in selected],
            original_tokens=orig_total,
            compressed_tokens=comp_tokens,
            provenance_verified=True,
        )
        ratio = round(comp_tokens / orig_total, 4) if orig_total > 0 else 1.0
        return CompressionResult(
            strategy_requested=CompressionStrategy.MODEL_ASSISTED,
            strategy_applied=CompressionStrategy.MODEL_ASSISTED,
            units=[unit],
            original_total_tokens=orig_total,
            compressed_total_tokens=comp_tokens,
            compression_ratio=ratio,
            deduplicated_count=0,
            independent_sources_preserved=len({item.source_domain for item in selected}),
            provenance_preserved=True,
            model_assisted_used=True,
            warnings=[],
        )

    def _fallback_after_model_failure(
        self,
        selected: list[SelectedEvidenceRecord],
        *,
        query: str,
        target_tokens: int | None,
        claims: list[ExtractedClaim],
        conflicts: list[ClaimConflict],
        reason: str,
    ) -> CompressionResult:
        fallback = self._compress_structured(
            selected,
            query=query,
            target_tokens=target_tokens,
            claims=claims,
            conflicts=conflicts,
        )
        fallback.strategy_requested = CompressionStrategy.MODEL_ASSISTED
        fallback.model_assisted_used = False
        fallback.model_assisted_fallback_reason = reason
        fallback.warnings.append(reason)
        return fallback

    @staticmethod
    def _validate_model_summary_integrity(
        summary: str,
        *,
        selected: list[SelectedEvidenceRecord],
        claims: list[ExtractedClaim],
    ) -> list[str]:
        issues: list[str] = []
        clean = summary.strip()
        if not clean:
            return ["Empty summary returned by local model."]

        valid_tags = {item.citation_label for item in selected}
        used_tags = set(_CITATION_TAG_RE.findall(clean))
        if not used_tags:
            issues.append("Summary omitted all [E#] source citation tags.")
        unknown_tags = used_tags - valid_tags
        if unknown_tags:
            issues.append(f"Summary cited unknown tags: {sorted(unknown_tags)}")

        conflict_tags = {item.citation_label for item in selected if item.is_conflict_evidence}
        missing_conflict_tags = conflict_tags - used_tags
        if missing_conflict_tags:
            issues.append(
                f"Summary dropped conflicting source citations: {sorted(missing_conflict_tags)}"
            )

        selected_ids = {item.evidence_id for item in selected}
        required_dates: set[str] = set()
        required_numbers: set[str] = set()
        required_entities: set[str] = set()

        for claim in claims:
            if claim.evidence_id in selected_ids:
                required_dates.update(_extract_dates(claim.statement_text))
                required_numbers.update(_extract_numbers(claim.normalized_value))
                if claim.entity:
                    required_entities.add(claim.entity)

        for item in selected:
            required_dates.update(_extract_dates(item.passage_text))
            required_numbers.update(_extract_numbers(item.passage_text)[:3])
            required_entities.update(_extract_entities(item.passage_text, limit=3))

        missing_dates = [d for d in sorted(required_dates) if d not in clean]
        if missing_dates:
            issues.append(f"Lost or altered dates: {missing_dates[:4]}")

        lower_clean = clean.lower()
        missing_numbers = []
        for num in sorted(required_numbers):
            core_num = num.rstrip("%xkmsGBM ")
            if core_num and core_num.lower() not in lower_clean:
                missing_numbers.append(num)
        if missing_numbers:
            issues.append(f"Lost or altered numeric values: {missing_numbers[:4]}")

        # Verify entity tokens without demanding identical adjacent phrasing
        entity_tokens: set[str] = set()
        for ent in required_entities:
            for part in ent.split():
                if len(part) >= 2 and part not in _ENTITY_IGNORE:
                    entity_tokens.add(part)
        missing_entities = [
            tok for tok in sorted(entity_tokens) if tok.lower() not in lower_clean
        ]
        if missing_entities and len(missing_entities) > max(0, len(entity_tokens) // 2):
            issues.append(f"Lost key entities: {missing_entities[:4]}")

        return issues

    @staticmethod
    def evidence_items_to_selected(
        evidence_items: list[EvidenceItem],
        *,
        conflicts: list[ClaimConflict] | None = None,
        estimator: BaseContextTokenEstimator | None = None,
    ) -> list[SelectedEvidenceRecord]:
        """Helper to wrap raw EvidenceItem objects into SelectedEvidenceRecord inputs for /compress."""
        from web_intelligence.app.context.models import EvidenceScoreBreakdown

        tok_est = estimator or ConservativeHeuristicEstimator()
        conflict_ids_by_ev: dict[str, list[str]] = {}
        for cf in conflicts or []:
            conflict_ids_by_ev.setdefault(cf.claim_a.evidence_id, []).append(cf.conflict_id)
            conflict_ids_by_ev.setdefault(cf.claim_b.evidence_id, []).append(cf.conflict_id)

        records: list[SelectedEvidenceRecord] = []
        for idx, item in enumerate(evidence_items, start=1):
            c_ids = conflict_ids_by_ev.get(
                item.evidence_id, list(item.corroboration.conflicting_claim_ids)
            )
            toks = tok_est.estimate(item.passage_text).tokens + 12
            records.append(
                SelectedEvidenceRecord(
                    evidence_id=item.evidence_id,
                    citation_label=f"[E{idx}]",
                    source_id=item.source_id,
                    source_url=item.source_url,
                    source_domain=item.source_domain,
                    source_title=item.source_title,
                    passage_text=item.passage_text,
                    original_token_count=toks,
                    allocated_token_count=toks,
                    selection_reason="direct_compression_input",
                    is_conflict_evidence=bool(c_ids),
                    conflict_ids=c_ids,
                    covered_subtopics=[],
                    scores=EvidenceScoreBreakdown(
                        relevance=item.relevance.combined_score,
                        topic_coverage=0.0,
                        source_independence=1.0,
                        provenance=0.5,
                        claim_support=0.5,
                        conflict_priority=1.0 if c_ids else 0.0,
                        recency=0.5,
                        token_cost_penalty=0.0,
                        final_score=item.relevance.combined_score,
                    ),
                )
            )
        return records
