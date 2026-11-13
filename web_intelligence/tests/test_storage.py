"""Integration tests for local SQLite storage, retrieval, and retention policies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from web_intelligence.app.schemas.documents import ExtractionStatus, SourceProvenance, SourceRecord
from web_intelligence.app.schemas.research import (
    ResearchMetadata,
    ResearchPackage,
    ResearchStatus,
)
from web_intelligence.app.storage.database import DatabaseManager
from web_intelligence.app.storage.repositories import ResearchRepository


async def test_sqlite_save_load_delete_and_retention(tmp_path: Path) -> None:
    db_path = tmp_path / "test_research.db"
    db = DatabaseManager(db_path)
    await db.initialize()
    repo = ResearchRepository(db, retention_days=10)

    now = datetime.now(timezone.utc)
    source = SourceRecord(
        source_id="src-1",
        original_url="https://example.org/article",
        canonical_url="https://example.org/article",
        final_url="https://example.org/article",
        domain="example.org",
        title="Example Article",
        retrieved_at=now,
        main_content="Normalized text content.",
        content_hash="abc123",
        extraction_status=ExtractionStatus.SUCCESS,
        provenance=SourceProvenance(
            domain="example.org",
            original_url="https://example.org/article",
            canonical_url="https://example.org/article",
        ),
    )
    pkg = ResearchPackage(
        research_id="res-1",
        original_query="test storage query",
        research_status=ResearchStatus.COMPLETED,
        sources=[source],
        metadata=ResearchMetadata(retrieved_sources=1, processed_documents=1, retrieved_at=now),
    )

    await repo.save_research_package(pkg)

    loaded = await repo.get_research_package("res-1")
    assert loaded is not None
    assert loaded.research_id == "res-1"
    assert len(loaded.sources) == 1

    loaded_source = await repo.get_source_by_id("src-1")
    assert loaded_source is not None
    assert loaded_source.title == "Example Article"

    runs = await repo.list_research_runs()
    assert len(runs) == 1
    assert runs[0].research_id == "res-1"

    # Purge with future reference time should delete expired record + cascaded sources
    purged = await repo.purge_expired_runs(reference_time=now + timedelta(days=11))
    assert purged == 1
    assert await repo.get_research_package("res-1") is None
    assert await repo.get_source_by_id("src-1") is None
