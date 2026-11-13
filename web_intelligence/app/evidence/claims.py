"""Deterministic claim extraction, conflict detection, and replaceable semantic interface."""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod

from web_intelligence.app.evidence.chunker import split_sentences
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    EvidenceItem,
    ExtractedClaim,
)

_MONTHS = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}

_DATE_ACTION_RE = re.compile(
    r"\b(?P<action>released|launched|announced|published|introduced|founded|created|updated|started)"
    r"(?:\s+(?:in|on|during))?\s+"
    r"(?P<date>"
    r"\d{4}-\d{2}-\d{2}|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+\d{1,2},?\s+\d{4}|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+\d{4}|"
    r"\d{4}"
    r")\b",
    re.IGNORECASE,
)

_QUANTITY_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "throughput",
        "tokens/sec",
        re.compile(
            r"\b(?P<num>\d+(?:\.\d+)?)\s*(?:tokens\s+per\s+second|tokens/sec|tokens/s|tok/s|tps)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "memory_ram",
        "gb",
        re.compile(
            r"\b(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>gb|gigabytes?|mb|megabytes?)\s*(?:of\s+)?(?:ram|memory|vram)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "context_window",
        "tokens",
        re.compile(
            r"\b(?P<num>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?P<scale>k|m)?\s*(?:token\s+)?context\s*(?:window|length)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "parameter_count",
        "B",
        re.compile(
            r"\b(?P<num>\d+(?:\.\d+)?)\s*(?:b|billion)\s*(?:parameter|parameters|params)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "latency",
        "ms",
        re.compile(
            r"\blatency(?:\s+of|\s+is|\s+was|\s+at)?\s+(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>ms|milliseconds?|s|seconds?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "price",
        "usd",
        re.compile(
            r"\$(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "percentage",
        "%",
        re.compile(
            r"\b(?P<num>\d+(?:\.\d+)?)\s*(?:%|percent)\b",
            re.IGNORECASE,
        ),
    ),
]

_NEGATION_SUPPORT_RE = re.compile(
    r"\b(?:does\s+not|doesn't|cannot|can't|never|without|no)\s+(?P<verb>support|require|include|allow|use)\s+(?P<obj>[a-z0-9][a-z0-9\s\-]{2,35})",
    re.IGNORECASE,
)
_AFFIRM_SUPPORT_RE = re.compile(
    r"\b(?P<verb>supports|requires|includes|allows|uses)\s+(?P<obj>[a-z0-9][a-z0-9\s\-]{2,35})",
    re.IGNORECASE,
)

# Model/project identifiers like Aurora-SLM-4B, Llama-3, GPT-4o, SQLite3
_IDENT_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b")
_GENERIC_SUBJECT_RE = re.compile(
    r"\b(?:the\s+)?(?:flagship\s+|open[- ]source\s+|local\s+|new\s+)?(?P<subj>[A-Za-z0-9][A-Za-z0-9\-]{2,30})\s+"
    r"(?:was|is|were|are|achieves|reached|consumes|requires|supports|does\s+not|cannot|has)\b",
    re.IGNORECASE,
)

_ACTION_TO_ATTR = {
    "released": "release_date",
    "launched": "release_date",
    "announced": "announcement_date",
    "published": "publication_date",
    "introduced": "release_date",
    "founded": "founded_date",
    "created": "created_date",
    "updated": "updated_date",
    "started": "start_date",
}


class ClaimComparator(ABC):
    """Replaceable interface for comparing extracted claims (allows future local LLM comparison)."""

    @abstractmethod
    async def compare(
        self,
        claims: list[ExtractedClaim],
    ) -> tuple[list[ClaimConflict], dict[str, list[ExtractedClaim]]]:
        """Identify conflicting claim pairs and corroborated claim groups."""


def normalize_date_literal(raw: str) -> str:
    """Convert matched date expressions into canonical ISO-prefix form."""
    cleaned = raw.strip().replace(",", "")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
        return cleaned
    if re.fullmatch(r"\d{4}", cleaned):
        return cleaned

    tokens = cleaned.lower().replace(".", "").split()
    if len(tokens) == 2 and tokens[0] in _MONTHS and tokens[1].isdigit():
        return f"{tokens[1]}-{_MONTHS[tokens[0]]}"
    if len(tokens) == 3 and tokens[0] in _MONTHS and tokens[1].isdigit() and tokens[2].isdigit():
        return f"{tokens[2]}-{_MONTHS[tokens[0]]}-{int(tokens[1]):02d}"
    return cleaned.lower()


def _dates_conflict(val_a: str, val_b: str) -> bool:
    """Return True if two normalized date strings conflict at their shared precision."""
    if val_a == val_b:
        return False
    # Prefix match (e.g. '2026-01' vs '2026-01-15' or '2026' vs '2026-03') is refinement, not conflict
    if val_a.startswith(val_b) or val_b.startswith(val_a):
        return False
    return True


def _extract_entity(sentence: str, *, fallback_title: str | None = None) -> str:
    """Identify the primary entity or subject for a sentence."""
    ident = _IDENT_ENTITY_RE.search(sentence)
    if ident:
        return re.sub(r"[-_]+", " ", ident.group(0)).strip().lower()

    subj = _GENERIC_SUBJECT_RE.search(sentence)
    if subj:
        candidate = subj.group("subj").strip().lower()
        if candidate not in {"it", "this", "that", "they", "which", "who", "what"}:
            return candidate

    if fallback_title:
        title_ident = _IDENT_ENTITY_RE.search(fallback_title)
        if title_ident:
            return re.sub(r"[-_]+", " ", title_ident.group(0)).strip().lower()
    return "subject"


class RuleBasedClaimExtractor:
    """Deterministic extractor for date, numerical quantity, and polarity assertions."""

    def extract_claims(self, evidence_items: list[EvidenceItem]) -> list[ExtractedClaim]:
        """Extract structured assertions from ranked evidence passages."""
        claims: list[ExtractedClaim] = []
        seen_keys: set[tuple[str, str, str, str, str]] = set()

        for item in evidence_items:
            sentences = split_sentences(item.passage_text)
            for sentence in sentences:
                entity = _extract_entity(sentence, fallback_title=item.source_title)

                # 1. Structured date claims
                for match in _DATE_ACTION_RE.finditer(sentence):
                    action = match.group("action").lower()
                    attr = _ACTION_TO_ATTR.get(action, f"{action}_date")
                    norm_date = normalize_date_literal(match.group("date"))
                    dedup_key = (item.source_id, entity, attr, "date", norm_date)
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        claims.append(
                            ExtractedClaim(
                                claim_id=str(uuid.uuid4()),
                                source_id=item.source_id,
                                evidence_id=item.evidence_id,
                                source_url=item.source_url,
                                source_domain=item.source_domain,
                                entity=entity,
                                attribute=attr,
                                statement_text=sentence,
                                value_type="date",
                                normalized_value=norm_date,
                            )
                        )

                # 2. Structured quantity claims
                for attr, default_unit, pattern in _QUANTITY_PATTERNS:
                    for match in pattern.finditer(sentence):
                        raw_num = match.group("num").replace(",", "")
                        try:
                            val = float(raw_num)
                        except ValueError:
                            continue

                        unit = default_unit
                        group_dict = match.groupdict()
                        scale = (group_dict.get("scale") or "").lower()
                        raw_unit = (group_dict.get("unit") or "").lower()

                        if scale == "k":
                            val *= 1000.0
                        elif scale == "m":
                            val *= 1_000_000.0

                        if raw_unit.startswith("mb") or raw_unit.startswith("megabyte"):
                            val /= 1024.0
                            unit = "gb"
                        elif raw_unit.startswith("s") and attr == "latency":
                            val *= 1000.0
                            unit = "ms"

                        # Include model-size context if memory refers to e.g. '7B'
                        scoped_entity = entity
                        if attr == "memory_ram":
                            size_match = re.search(r"\b(\d+b)\b", sentence, re.I)
                            if size_match:
                                scoped_entity = f"{size_match.group(1).lower()} model"

                        norm_val = f"{val:g}"
                        dedup_key = (item.source_id, scoped_entity, attr, "quantity", norm_val)
                        if dedup_key not in seen_keys:
                            seen_keys.add(dedup_key)
                            claims.append(
                                ExtractedClaim(
                                    claim_id=str(uuid.uuid4()),
                                    source_id=item.source_id,
                                    evidence_id=item.evidence_id,
                                    source_url=item.source_url,
                                    source_domain=item.source_domain,
                                    entity=scoped_entity,
                                    attribute=attr,
                                    statement_text=sentence,
                                    value_type="quantity",
                                    normalized_value=norm_val,
                                    unit=unit,
                                )
                            )

                # 3. Polarity claims (supports/requires vs does not support/without)
                for pol, pattern in (
                    ("negative", _NEGATION_SUPPORT_RE),
                    ("affirmative", _AFFIRM_SUPPORT_RE),
                ):
                    for match in pattern.finditer(sentence):
                        verb = match.group("verb").lower().rstrip("s")
                        obj_words = re.findall(r"[a-z0-9]+", match.group("obj").lower())[:3]
                        if not obj_words:
                            continue
                        attr = f"{verb}_{'_'.join(obj_words)}"
                        dedup_key = (item.source_id, entity, attr, "polarity", pol)
                        if dedup_key not in seen_keys:
                            seen_keys.add(dedup_key)
                            claims.append(
                                ExtractedClaim(
                                    claim_id=str(uuid.uuid4()),
                                    source_id=item.source_id,
                                    evidence_id=item.evidence_id,
                                    source_url=item.source_url,
                                    source_domain=item.source_domain,
                                    entity=entity,
                                    attribute=attr,
                                    statement_text=sentence,
                                    value_type="polarity",
                                    normalized_value=pol,
                                    polarity=pol,
                                )
                            )

        return claims


class DeterministicClaimComparator(ClaimComparator):
    """Compares structured date, numerical, and polarity claims across independent sources."""

    def __init__(self, *, numeric_relative_tolerance: float = 0.05) -> None:
        self._rel_tol = numeric_relative_tolerance

    async def compare(
        self,
        claims: list[ExtractedClaim],
    ) -> tuple[list[ClaimConflict], dict[str, list[ExtractedClaim]]]:
        grouped: dict[tuple[str, str, str], list[ExtractedClaim]] = {}
        for claim in claims:
            key = (claim.entity, claim.attribute, claim.value_type)
            grouped.setdefault(key, []).append(claim)

        conflicts: list[ClaimConflict] = []
        corroborated_groups: dict[str, list[ExtractedClaim]] = {}
        seen_pairs: set[tuple[str, str]] = set()

        for (entity, attribute, value_type), bucket in grouped.items():
            # Group corroborating claims with matching values across independent domains
            by_norm_val: dict[str, list[ExtractedClaim]] = {}
            for c in bucket:
                by_norm_val.setdefault(c.normalized_value, []).append(c)

            for norm_val, matching_claims in by_norm_val.items():
                unique_domains = {c.source_domain for c in matching_claims}
                if len(unique_domains) >= 2:
                    corroborated_groups[f"{entity}|{attribute}|{norm_val}"] = matching_claims

            if len(bucket) < 2:
                continue

            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    c_a = bucket[i]
                    c_b = bucket[j]
                    # Compare only across distinct independent domains/sources
                    if c_a.source_id == c_b.source_id or c_a.source_domain == c_b.source_domain:
                        continue

                    pair_key = tuple(sorted((c_a.claim_id, c_b.claim_id)))
                    if pair_key in seen_pairs:
                        continue

                    conflict_type: str | None = None
                    if value_type == "date" and _dates_conflict(c_a.normalized_value, c_b.normalized_value):
                        conflict_type = "date_mismatch"
                    elif value_type == "quantity" and c_a.unit == c_b.unit:
                        try:
                            num_a = float(c_a.normalized_value)
                            num_b = float(c_b.normalized_value)
                            denom = max(abs(num_a), abs(num_b), 1e-9)
                            if abs(num_a - num_b) / denom > self._rel_tol:
                                conflict_type = "numeric_mismatch"
                        except ValueError:
                            pass
                    elif value_type == "polarity" and c_a.normalized_value != c_b.normalized_value:
                        conflict_type = "polarity_mismatch"

                    if conflict_type is not None:
                        seen_pairs.add(pair_key)
                        unit_suffix_a = f" {c_a.unit}" if c_a.unit else ""
                        unit_suffix_b = f" {c_b.unit}" if c_b.unit else ""
                        summary = (
                            f"Conflicting {attribute} for '{entity}': "
                            f"{c_a.source_domain} reports '{c_a.normalized_value}{unit_suffix_a}' "
                            f"whereas {c_b.source_domain} reports '{c_b.normalized_value}{unit_suffix_b}'."
                        )
                        conflicts.append(
                            ClaimConflict(
                                conflict_id=str(uuid.uuid4()),
                                topic=f"{entity} {attribute}".strip(),
                                entity=entity,
                                attribute=attribute,
                                conflict_type=conflict_type,
                                claim_a=c_a,
                                claim_b=c_b,
                                summary=summary,
                            )
                        )

        return conflicts, corroborated_groups
