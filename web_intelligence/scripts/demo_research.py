"""End-to-end CLI demonstration of the Phase 1 (Stages 2-4) Web Intelligence Engine."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from web_intelligence.app.core.config import Settings
from web_intelligence.app.dependencies import build_container
from web_intelligence.app.providers.fixtures import (
    build_fixture_transport,
    fixture_dns_resolver,
)
from web_intelligence.app.schemas.research import ResearchRequest


async def main() -> None:
    query = (
        "Aurora-SLM-4B release date and small language models benchmarks "
        "and privacy boundaries and neuromorphic hardware support"
    )
    with TemporaryDirectory() as tmpdir:
        settings = Settings(
            database_path=Path(tmpdir) / "demo.db",
            search_providers=["mock"],
            enable_mock_provider=True,
            per_host_min_interval_seconds=0.0,
        )
        container = build_container(
            settings,
            transport_factory=build_fixture_transport,
            dns_resolver=fixture_dns_resolver,
        )
        await container.database.initialize()

        package = await container.orchestrator.run_research(
            ResearchRequest(query=query, max_results_per_query=6, max_fetch_urls=8)
        )

        print(f"Research ID: {package.research_id}")
        print(f"Status: {package.research_status.value}")
        print(f"Metadata: {json.dumps(package.metadata.model_dump(mode='json'), indent=2)}")
        print("\n--- Sources (Deduplication & Provenance) ---")
        for src in package.sources:
            dup = f"DUPLICATE({src.duplicate_reason})" if src.is_duplicate else "UNIQUE"
            cls = src.provenance.source_classification.value if src.provenance else "unknown"
            pub = src.published_at.date().isoformat() if src.published_at else "Not reported"
            print(f"  [{dup}] [{cls}] {src.domain} | pub={pub} | {src.title}")

        print("\n--- Top 3 Ranked Evidence Passages ---")
        for ev in package.evidence[:3]:
            print(
                f"  score={ev.relevance.combined_score:.3f} "
                f"(bm25={ev.relevance.bm25_score:.2f}, kw={ev.relevance.keyword_match_score:.2f}) "
                f"[{ev.corroboration.corroboration_status.value}] "
                f"{ev.source_domain} :: {ev.section_heading}"
            )
            print(f"    {ev.passage_text[:180]}...")

        print("\n--- Conflicting Claims ---")
        for c in package.conflicts:
            print(f"  [{c.conflict_type}] {c.summary}")

        print("\n--- Research Coverage ---")
        print(f"  Status: {package.coverage.coverage_status.value}")
        print(f"  Covered: {package.coverage.covered_topics}")
        print(f"  Partial: {package.coverage.partial_topics}")
        print(f"  Missing: {package.coverage.missing_topics}")
        print(f"  Unanswered Gaps: {package.coverage.unanswered_questions}")


if __name__ == "__main__":
    asyncio.run(main())
