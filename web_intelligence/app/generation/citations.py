"""Citation extraction, source mapping, and separate structural grounding validation."""

from __future__ import annotations

import re
from typing import Iterable

from web_intelligence.app.schemas.generation import (
    CitationEntry,
    CitationValidationReport,
    UncitedStatementWarning,
    UnsupportedCitationError,
)

# Matches [S1], [S2], [S1, S2], [S1; S3], or Stage-5 [E1] variants
_CITATION_GROUP_RE = re.compile(r"\[((?:[SE]\d+)(?:\s*[,;]\s*[SE]\d+)*)\]")
_SINGLE_TAG_RE = re.compile(r"([SE]\d+)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_CODE_FENCE_RE = re.compile(r"^\s*```")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[])|\n+")
_CITED_TAG_ANYWHERE_RE = re.compile(r"\[(?:[SE]\d+)(?:\s*[,;]\s*[SE]\d+)*\]")
_HEDGING_GAP_RE = re.compile(
    r"\b(?:does not confirm|do not confirm|not mentioned|no retrieved evidence|"
    r"insufficient evidence|cannot be verified|unclear from the sources|"
    r"conflict(?:s|ing)? between sources)\b",
    re.IGNORECASE,
)


def _strip_code_blocks_preserve_offsets(text: str) -> str:
    """Mask markdown fenced code blocks with spaces so spans remain exact."""
    chars = list(text)
    in_fence = False
    offset = 0
    for line in text.splitlines(keepends=True):
        if _CODE_FENCE_RE.match(line):
            in_fence = not in_fence
            offset += len(line)
            continue
        if in_fence:
            for idx in range(offset, offset + len(line)):
                if chars[idx] not in ("\n", "\r"):
                    chars[idx] = " "
        offset += len(line)
    return "".join(chars)


class CitationResolver:
    """Extracts [S1]/[S2]/... citations from generated text and maps spans to source metadata."""

    def resolve(
        self,
        text: str,
        citation_map: dict[str, CitationEntry],
        *,
        stage5_to_citation_id: dict[str, str] | None = None,
    ) -> tuple[list[CitationEntry], list[UnsupportedCitationError], list[str]]:
        """Resolve citation markers in `text` to source records with exact character spans."""
        spans_by_id: dict[str, list[tuple[int, int]]] = {}
        ordered_ids: list[str] = []
        unsupported: list[UnsupportedCitationError] = []
        stage5_alias = stage5_to_citation_id or {}

        for group_match in _CITATION_GROUP_RE.finditer(text):
            group_span = (group_match.start(), group_match.end())
            inner = group_match.group(1)
            for tag_match in _SINGLE_TAG_RE.finditer(inner):
                raw_id = tag_match.group(1)
                resolved_id = raw_id
                if resolved_id.startswith("E") and resolved_id in stage5_alias:
                    resolved_id = stage5_alias[resolved_id]

                if resolved_id not in citation_map:
                    unsupported.append(
                        UnsupportedCitationError(
                            raw_tag=f"[{raw_id}]",
                            span=group_span,
                            reason=(
                                f"Citation identifier [{raw_id}] does not correspond to any "
                                "source included in the generation context."
                            ),
                        )
                    )
                    continue

                if resolved_id not in spans_by_id:
                    spans_by_id[resolved_id] = []
                    ordered_ids.append(resolved_id)
                spans_by_id[resolved_id].append(group_span)

        resolved_entries: list[CitationEntry] = []
        for cid in ordered_ids:
            base = citation_map[cid]
            spans = spans_by_id[cid]
            resolved_entries.append(
                base.model_copy(
                    update={
                        "spans": spans,
                        "occurrence_count": len(spans),
                    }
                )
            )
        return resolved_entries, unsupported, ordered_ids


class CitationValidator:
    """Separate structural grounding check: verifies citation IDs and flags uncited claims.

    Note: Validating citation IDs never guarantees objective factual accuracy; it strictly checks
    whether generated assertions reference real sources provided in the context window.
    """

    def __init__(self, resolver: CitationResolver | None = None) -> None:
        self._resolver = resolver or CitationResolver()

    def validate(
        self,
        text: str,
        citation_map: dict[str, CitationEntry],
        *,
        stage5_to_citation_id: dict[str, str] | None = None,
        require_citations: bool = True,
        json_mode: bool = False,
    ) -> tuple[list[CitationEntry], CitationValidationReport]:
        """Validate citations and flag unsupported IDs or uncited factual statements."""
        resolved, unsupported, used_ids = self._resolver.resolve(
            text,
            citation_map,
            stage5_to_citation_id=stage5_to_citation_id,
        )
        known_ids = list(citation_map.keys())
        unused_ids = [cid for cid in known_ids if cid not in used_ids]
        uncited: list[UncitedStatementWarning] = []
        warnings: list[str] = []

        if text.strip() and known_ids and not json_mode:
            masked = _strip_code_blocks_preserve_offsets(text)
            idx = 0
            cursor = 0
            for raw_chunk in _SENTENCE_SPLIT_RE.split(masked):
                if not raw_chunk:
                    continue
                pos = masked.find(raw_chunk, cursor)
                if pos == -1:
                    pos = cursor
                cursor = pos + len(raw_chunk)

                stripped = raw_chunk.strip()
                if not stripped or len(stripped) < 28:
                    continue
                if _HEADING_RE.match(stripped):
                    continue
                if stripped.startswith(("|", "---", "===", "{", "}", "[", "]")):
                    continue
                if stripped.lstrip("-*0123456789. ").strip() == "":
                    continue
                # Skip explicit epistemic gap disclosures ("The retrieved evidence does not confirm...")
                if _HEDGING_GAP_RE.search(stripped):
                    continue

                if not _CITED_TAG_ANYWHERE_RE.search(stripped):
                    # Flag declarative statements lacking citation markers when evidence exists
                    words = [w for w in stripped.split() if any(ch.isalnum() for ch in w)]
                    if len(words) >= 6:
                        uncited.append(
                            UncitedStatementWarning(
                                statement_index=idx,
                                text=stripped[:240],
                                span=(pos, pos + len(raw_chunk)),
                                reason=(
                                    "Statement makes an assertion without an inline [S#] source reference."
                                ),
                            )
                        )
                        idx += 1

        if unsupported:
            bad_tags = ", ".join(sorted({item.raw_tag for item in unsupported}))
            warnings.append(f"Unsupported citation identifier(s) detected: {bad_tags}.")
        if require_citations and known_ids and text.strip() and not used_ids:
            warnings.append(
                "Response contains no valid [S#] citations despite retrieved source evidence."
            )
        if uncited:
            warnings.append(
                f"{len(uncited)} factual statement(s) appear without an inline source citation."
            )

        passed = (
            len(unsupported) == 0
            and (
                not require_citations
                or not known_ids
                or not text.strip()
                or (len(used_ids) > 0 and len(uncited) == 0)
            )
        )

        report = CitationValidationReport(
            passed=passed,
            require_citations=require_citations,
            known_citation_ids=known_ids,
            used_citation_ids=used_ids,
            unused_citation_ids=unused_ids,
            unsupported_citations=unsupported,
            uncited_statements=uncited,
            warnings=warnings,
        )
        return resolved, report


def merge_citations_preserving_order(
    citations_lists: Iterable[list[CitationEntry]],
) -> tuple[list[CitationEntry], dict[str, CitationEntry]]:
    """Combine section citation lists deduplicating by citation_id while preserving order."""
    merged: dict[str, CitationEntry] = {}
    for group in citations_lists:
        for item in group:
            existing = merged.get(item.citation_id)
            if existing is None:
                merged[item.citation_id] = item
            else:
                merged[item.citation_id] = existing.model_copy(
                    update={
                        "spans": existing.spans + item.spans,
                        "occurrence_count": existing.occurrence_count + item.occurrence_count,
                    }
                )
    ordered = list(merged.values())
    return ordered, merged
