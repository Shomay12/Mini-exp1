"""Full end-to-end FastAPI integration tests using MockSearchProvider and fixture HTML."""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import status

from web_intelligence.app.core.config import Settings
from web_intelligence.app.main import create_app
from web_intelligence.app.providers.fixtures import (
    build_fixture_transport,
    fixture_dns_resolver,
)
from web_intelligence.app.schemas.research import ResearchStatus


async def test_full_api_and_stage4_research_workflow(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "integration.db",
        search_providers=["mock"],
        enable_mock_provider=True,
        brave_api_key="BSA-secret-should-never-leak",
        per_host_min_interval_seconds=0.0,
    )
    app = create_app(
        settings,
        transport_factory=build_fixture_transport,
        dns_resolver=fixture_dns_resolver,
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. UI and static assets served cleanly
            ui_resp = await client.get("/")
            assert ui_resp.status_code == 200
            assert "WEB INTELLIGENCE ENGINE" in ui_resp.text
            assert (await client.get("/static/styles.css")).status_code == 200
            assert (await client.get("/static/app.js")).status_code == 200

            # 2. Health & safe config
            health = (await client.get("/api/v1/health")).json()
            assert health["status"] == "ok"
            assert health["database_ready"] is True
            assert health["cloud_llm_required"] is False

            cfg_resp = await client.get("/api/v1/config")
            assert cfg_resp.status_code == 200
            assert "BSA-secret-should-never-leak" not in cfg_resp.text
            assert cfg_resp.json()["config"]["configured_credentials"]["brave"] is True

            # 3. Search endpoint + sanitization + validation
            bad_search = await client.post("/api/v1/search", json={"query": "a"})
            assert bad_search.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

            search_resp = await client.post(
                "/api/v1/search",
                json={"query": "small language models for alice@example.com"},
            )
            assert search_resp.status_code == 200
            search_data = search_resp.json()
            assert "alice@example.com" not in search_data["sanitized_query"]
            assert "email_address" in search_data["redactions_applied"]
            assert search_data["total_results"] >= 3

            # 4. Complete Stage 3 + Stage 4 Research Pipeline
            res_resp = await client.post(
                "/api/v1/research",
                json={
                    "query": (
                        "Aurora-SLM-4B release date and small language models benchmarks "
                        "and quantum entanglement routing"
                    ),
                    "max_results_per_query": 6,
                    "max_fetch_urls": 8,
                },
            )
            assert res_resp.status_code == 200
            pkg = res_resp.json()
            assert pkg["research_status"] == ResearchStatus.COMPLETED.value

            meta = pkg["metadata"]
            assert meta["retrieved_sources"] >= 4
            assert meta["duplicate_documents"] >= 1  # mirror.news-syndicate copy detected!
            assert meta["processed_documents"] >= 3
            assert meta["evidence_items"] >= 3
            assert meta["conflicts_detected"] >= 2  # release date + M2 tok/s throughput

            # Missing publication date must stay None (never invented)
            arxiv_sources = [
                s for s in pkg["sources"] if "arxiv-notes.example.edu" in s["domain"]
            ]
            assert len(arxiv_sources) == 1
            assert arxiv_sources[0]["published_at"] is None

            # Conflicts must preserve both sides without selecting a winner
            conflict_types = {c["conflict_type"] for c in pkg["conflicts"]}
            assert "date_mismatch" in conflict_types
            assert "numeric_mismatch" in conflict_types

            # Coverage must flag missing quantum topic rather than claiming full coverage
            cov = pkg["coverage"]
            assert cov["coverage_status"] == "partial"
            assert any("quantum" in topic for topic in cov["missing_topics"])

            # 5. Persistence & inspection endpoints
            rid = pkg["research_id"]
            fetched_pkg = await client.get(f"/api/v1/research/{rid}")
            assert fetched_pkg.status_code == 200
            assert fetched_pkg.json()["research_id"] == rid

            runs_list = (await client.get("/api/v1/research")).json()
            assert any(r["research_id"] == rid for r in runs_list)

            first_source_id = pkg["sources"][0]["source_id"]
            src_resp = await client.get(f"/api/v1/sources/{first_source_id}")
            assert src_resp.status_code == 200
            assert src_resp.json()["source_id"] == first_source_id

            # 6. 404 & DELETE retention control
            assert (await client.get("/api/v1/research/does-not-exist")).status_code == 404
            assert (await client.get("/api/v1/sources/does-not-exist")).status_code == 404

            del_resp = await client.delete(f"/api/v1/research/{rid}")
            assert del_resp.status_code == status.HTTP_204_NO_CONTENT
            assert (await client.get(f"/api/v1/research/{rid}")).status_code == 404
