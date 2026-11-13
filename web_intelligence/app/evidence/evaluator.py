"""Source provenance evaluation kept strictly separate from relevance and factual truth."""

from __future__ import annotations

from urllib.parse import urlparse

from web_intelligence.app.schemas.documents import (
    SourceClassification,
    SourceProvenance,
    SourceRecord,
)
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CorroborationSignals,
    CorroborationStatus,
    EvidenceItem,
    ExtractedClaim,
)

_REFERENCE_DOMAINS = (
    "wikipedia.org",
    "wikimedia.org",
    "wikidata.org",
    "britannica.com",
    "stanford.edu",  # Note: encyclopedia.stanford.edu handled below too
)

_PRIMARY_TLDS = (".gov", ".mil", ".gov.uk", ".edu")
_PRIMARY_HOST_PREFIXES = ("docs.", "developer.", "developers.", "spec.", "arxiv.", "github.")
_PRIMARY_PATH_HINTS = ("/docs/", "/documentation/", "/spec/", "/papers/", "/rfc/", "/release")
_SECONDARY_PATH_HINTS = ("/news/", "/blog/", "/review", "/article", "/opinion/", "/post/")
_SECONDARY_DOMAINS = ("medium.com", "substack.com", "dev.to")


def classify_source_provenance(source: SourceRecord) -> tuple[SourceClassification, str]:
    """Classify source role structurally without equating domain with factual correctness."""
    domain = source.domain.lower()
    path = (urlparse(source.final_url or source.original_url).path or "").lower()
    publisher = (source.source_metadata.publisher or "").lower()

    if any(domain.endswith(ref) for ref in ("wikipedia.org", "wikimedia.org", "wikidata.org", "britannica.com")):
        return SourceClassification.REFERENCE, "Encyclopedic/reference domain structure."

    if domain.startswith("docs.") or domain.startswith("spec.") or "official docs" in publisher:
        return SourceClassification.PRIMARY, "Official/technical documentation host or publisher cue."

    if any(domain.endswith(tld) for tld in _PRIMARY_TLDS) or "arxiv" in domain:
        return SourceClassification.PRIMARY, "Institutional, academic, or public-sector primary domain."

    if domain in {"github.com", "gitlab.com"} or any(domain.startswith(p) for p in _PRIMARY_HOST_PREFIXES):
        return SourceClassification.PRIMARY, "Repository or developer specification source."

    if any(hint in path for hint in _PRIMARY_PATH_HINTS):
        return SourceClassification.PRIMARY, "Technical specification or primary paper URL path."

    if any(domain.endswith(sec) for sec in _SECONDARY_DOMAINS) or any(
        hint in path for hint in _SECONDARY_PATH_HINTS
    ) or source.author:
        return (
            SourceClassification.SECONDARY,
            "Editorial, review, news, or authored commentary source.",
        )

    return SourceClassification.UNKNOWN, "Insufficient structural indicators to classify primary vs secondary."


class SourceProvenanceEvaluator:
    """Evaluates inspectable provenance and cross-source corroboration signals."""

    def build_provenance(self, source: SourceRecord) -> SourceProvenance:
        """Build separated SourceProvenance for a retrieved source."""
        classification, reason = classify_source_provenance(source)
        return SourceProvenance(
            domain=source.domain,
            publisher=source.source_metadata.publisher or source.domain,
            source_classification=classification,
            classification_reason=reason,
            author=source.author,
            published_at=source.published_at,
            original_url=source.original_url,
            canonical_url=source.canonical_url,
        )

    def evaluate_corroboration(
        self,
        evidence_items: list[EvidenceItem],
        sources: list[SourceRecord],
        claims: list[ExtractedClaim],
        conflicts: list[ClaimConflict],
        corroborated_groups: dict[str, list[ExtractedClaim]],
    ) -> list[EvidenceItem]:
        """Attach cross-source support, mirror-cluster, and conflict flags to each EvidenceItem."""
        # Map primary source -> duplicate alias source IDs/URLs (same underlying source cluster)
        cluster_by_source: dict[str, list[str]] = {}
        for src in sources:
            if src.is_duplicate and src.duplicate_of_source_id:
                cluster_by_source.setdefault(src.duplicate_of_source_id, []).append(src.original_url)
            elif src.alias_urls:
                cluster_by_source.setdefault(src.source_id, []).extend(src.alias_urls)

        conflicts_by_evidence: dict[str, list[str]] = {}
        for conflict in conflicts:
            conflicts_by_evidence.setdefault(conflict.claim_a.evidence_id, []).append(conflict.conflict_id)
            conflicts_by_evidence.setdefault(conflict.claim_b.evidence_id, []).append(conflict.conflict_id)

        support_sources_by_evidence: dict[str, set[str]] = {}
        support_domains_by_evidence: dict[str, set[str]] = {}
        for group_claims in corroborated_groups.values():
            unique_domains = {c.source_domain for c in group_claims}
            if len(unique_domains) < 2:
                continue
            all_sids = {c.source_id for c in group_claims}
            for claim in group_claims:
                support_sources_by_evidence.setdefault(claim.evidence_id, set()).update(
                    all_sids - {claim.source_id}
                )
                support_domains_by_evidence.setdefault(claim.evidence_id, set()).update(
                    unique_domains - {claim.source_domain}
                )

        claims_present_ev = {c.evidence_id for c in claims}

        updated: list[EvidenceItem] = []
        for item in evidence_items:
            conflict_ids = sorted(set(conflicts_by_evidence.get(item.evidence_id, [])))
            other_sids = sorted(support_sources_by_evidence.get(item.evidence_id, set()))
            other_domains = sorted(support_domains_by_evidence.get(item.evidence_id, set()))
            total_independent = 1 + len(other_domains) if other_domains else (
                1 if item.evidence_id in claims_present_ev else 0
            )

            if conflict_ids:
                status = CorroborationStatus.CONTESTED
            elif len(other_domains) >= 1:
                status = CorroborationStatus.CORROBORATED
            elif item.evidence_id in claims_present_ev:
                status = CorroborationStatus.SINGLE_SOURCE
            else:
                status = CorroborationStatus.UNVERIFIED

            cluster_urls = sorted(set(cluster_by_source.get(item.source_id, [])))
            corroboration = CorroborationSignals(
                corroboration_status=status,
                independent_supporting_sources=total_independent,
                supporting_source_ids=other_sids,
                supporting_domains=other_domains,
                conflicting_claim_ids=conflict_ids,
                same_underlying_source_cluster=cluster_urls,
            )
            updated.append(item.model_copy(update={"corroboration": corroboration}))

        return updated
