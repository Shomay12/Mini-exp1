"""Full end-to-end FastAPI integration tests using MockSearchProvider and fixture HTML."""

from __future__ import annotations

from pathlib import Path
import json

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
            # 1. UI and modular static assets served cleanly
            ui_resp = await client.get("/")
            assert ui_resp.status_code == 200
            assert "WEB INTELLIGENCE ENGINE" in ui_resp.text
            for asset_path in (
                "/static/styles.css",
                "/static/app.js",
                "/static/styles/tokens.css",
                "/static/styles/layout.css",
                "/static/styles/components.css",
                "/static/js/app.js",
                "/static/js/api.js",
                "/static/js/components/workspace.js",
                "/static/js/components/inspector.js",
                "/static/js/components/settings_sheet.js",
            ):
                assert (await client.get(asset_path)).status_code == 200

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


async def test_universal_llm_api_routes_sse_security_and_research_workflow(
    tmp_path: Path,
) -> None:
    secret_key = "api-llm-secret-should-never-leak"

    def mock_llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "Qwen3-4B",
                            "object": "model",
                            "owned_by": "vllm",
                            "max_model_len": 32768,
                            "supported_parameters": ["tools", "response_format"],
                        }
                    ],
                },
            )
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content.decode())
            if body.get("stream"):
                sse = (
                    'data: {"model":"Qwen3-4B","choices":[{"delta":{"content":"API "},"finish_reason":null}]}\n\n'
                    'data: {"model":"Qwen3-4B","choices":[{"delta":{"content":"Ready"},"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                )
                return httpx.Response(200, content=sse.encode("utf-8"))

            sys_prompt = body["messages"][0]["content"]
            if "subqueries" in sys_prompt:
                content = json.dumps(
                    {
                        "subqueries": [
                            {
                                "query": "Aurora-SLM-4B release date benchmarks",
                                "subtopic": "Aurora-SLM-4B",
                                "rationale": "benchmarks",
                            }
                        ]
                    }
                )
            else:
                content = (
                    "Evidence reports Aurora-SLM-4B performance metrics [E1], while conflicting reports "
                    "differ across benchmark sources [E2]."
                )
            return httpx.Response(
                200,
                json={
                    "model": "Qwen3-4B",
                    "choices": [
                        {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                },
            )
        return httpx.Response(404)

    settings = Settings(
        database_path=tmp_path / "llm_api.db",
        search_providers=["mock"],
        enable_mock_provider=True,
        per_host_min_interval_seconds=0.0,
    )
    app = create_app(
        settings,
        transport_factory=build_fixture_transport,
        dns_resolver=fixture_dns_resolver,
        llm_transport=httpx.MockTransport(mock_llm_handler),
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Verify new modular LLM UI component asset is served cleanly
            js_resp = await client.get("/static/js/components/llm_pane.js")
            assert js_resp.status_code == 200

            presets_resp = await client.get("/api/v1/llm/presets")
            assert presets_resp.status_code == 200
            preset_ids = {p["id"] for p in presets_resp.json()}
            assert {"ollama", "lm-studio", "llamacpp", "vllm", "localai"} <= preset_ids

            # Reject metadata SSRF and unapproved remote hosts
            ssrf_resp = await client.post(
                "/api/v1/llm/connections",
                json={
                    "name": "Bad Metadata",
                    "provider": "openai-compatible",
                    "base_url": "http://169.254.169.254/latest/meta-data",
                    "allow_remote_endpoint": True,
                },
            )
            assert ssrf_resp.status_code == 403

            remote_denied = await client.post(
                "/api/v1/llm/connections",
                json={
                    "name": "Remote No Exception",
                    "provider": "openai-compatible",
                    "base_url": "http://192.168.1.99:8000/v1",
                    "allow_remote_endpoint": False,
                },
            )
            assert remote_denied.status_code == 403

            # Create & activate local connection with optional API key
            create_resp = await client.post(
                "/api/v1/llm/connections",
                json={
                    "name": "vLLM Local",
                    "provider": "openai-compatible",
                    "preset": "vllm",
                    "base_url": "http://localhost:8000/v1",
                    "api_key": secret_key,
                    "model": "Qwen3-4B",
                    "context_length": 16384,
                    "max_output_tokens": 2048,
                    "streaming": True,
                    "activate": True,
                },
            )
            assert create_resp.status_code == 201
            conn = create_resp.json()
            conn_id = conn["id"]
            assert conn["has_api_key"] is True
            assert secret_key not in create_resp.text

            # List and health must never expose API key and must show active_llm
            list_resp = await client.get("/api/v1/llm/connections")
            assert list_resp.status_code == 200
            assert secret_key not in list_resp.text

            health_resp = await client.get("/api/v1/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["active_llm"]["model"] == "Qwen3-4B"

            # Test connectivity & model discovery & model selection
            test_resp = await client.post(f"/api/v1/llm/connections/{conn_id}/test")
            assert test_resp.status_code == 200
            test_data = test_resp.json()
            assert test_data["ok"] is True
            assert test_data["capabilities"]["max_context_tokens"] == 32768
            assert test_data["capabilities"]["sources"]["context_length"] == "verified"

            models_resp = await client.get(f"/api/v1/llm/connections/{conn_id}/models")
            assert models_resp.status_code == 200
            assert models_resp.json()["discovery_supported"] is True

            select_resp = await client.post(
                f"/api/v1/llm/connections/{conn_id}/model",
                json={"model": "Qwen3-4B"},
            )
            assert select_resp.status_code == 200
            assert select_resp.json()["model"] == "Qwen3-4B"

            # Streaming test prompt via SSE
            stream_resp = await client.post(
                f"/api/v1/llm/connections/{conn_id}/prompt",
                json={"prompt": "Confirm readiness", "stream": True},
            )
            assert stream_resp.status_code == 200
            assert "API " in stream_resp.text
            assert "Ready" in stream_resp.text
            assert "data: [DONE]" in stream_resp.text

            # Full research workflow using active local LLM
            res_resp = await client.post(
                "/api/v1/research",
                json={
                    "query": "Aurora-SLM-4B release date and small language models benchmarks",
                    "use_local_llm": True,
                },
            )
            assert res_resp.status_code == 200
            pkg = res_resp.json()
            assert pkg["research_status"] == ResearchStatus.COMPLETED.value
            assert pkg["synthesis"] is not None
            assert "[E1]" in pkg["synthesis"]["answer"]
            assert pkg["context_allocation"]["max_context_tokens"] == 32768
            assert secret_key not in res_resp.text

            # Disconnect active LLM -> returns to clean Zero-LLM mode
            disc_resp = await client.post("/api/v1/llm/disconnect")
            assert disc_resp.status_code == 200
            health_after = await client.get("/api/v1/health")
            assert health_after.json()["active_llm"] is None

            # Delete connection
            del_conn = await client.delete(f"/api/v1/llm/connections/{conn_id}")
            assert del_conn.status_code == status.HTTP_204_NO_CONTENT
