"""API route and end-to-end local model workflow integration tests for Stage 5 Context Engine."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi import status

from web_intelligence.app.context.context_builder import ContextAdapterConverter
from web_intelligence.app.context.models import ContextPreviewResponse
from web_intelligence.app.core.config import Settings
from web_intelligence.app.main import create_app
from web_intelligence.app.providers.fixtures import (
    build_fixture_transport,
    fixture_dns_resolver,
)


async def test_context_api_endpoints_and_end_to_end_local_model_workflow(
    tmp_path: Path,
) -> None:
    recorded_chat_requests: list[dict] = []

    def mock_llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "local-research-4k",
                            "object": "model",
                            "owned_by": "vllm",
                            "max_model_len": 4096,
                            "max_output_tokens": 1024,
                            "supported_parameters": ["tools", "response_format"],
                        }
                    ],
                },
            )
        if request.url.path == "/v1/chat/completions":
            payload = json.loads(request.content.decode("utf-8"))
            recorded_chat_requests.append(payload)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-stage5-demo",
                    "model": "local-research-4k",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Grounded local synthesis referencing [E1].",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 220,
                        "completion_tokens": 12,
                        "total_tokens": 232,
                    },
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    llm_transport = httpx.MockTransport(mock_llm_handler)

    async def loopback_resolver(host: str, port: int) -> list[str]:
        del host, port
        return ["127.0.0.1"]

    settings = Settings(
        database_path=tmp_path / "stage5_workflow.db",
        search_providers=["mock"],
        enable_mock_provider=True,
        per_host_min_interval_seconds=0.0,
    )
    app = create_app(
        settings,
        transport_factory=build_fixture_transport,
        dns_resolver=fixture_dns_resolver,
        llm_transport=llm_transport,
        llm_resolver=loopback_resolver,
    )

    async with app.router.lifespan_context(app):
        asgi_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=asgi_transport, base_url="http://testserver"
        ) as client:
            # 1. Verify UI includes Stage 5 Context Inspector assets
            ui_resp = await client.get("/")
            assert ui_resp.status_code == 200
            assert 'data-inspector-tab="context"' in ui_resp.text
            assert (
                await client.get("/static/js/components/context_inspector.js")
            ).status_code == 200

            # 2. Register & activate a local OpenAI-compatible mock server (4K window)
            create_conn = await client.post(
                "/api/v1/llm/connections",
                json={
                    "name": "Local Mock vLLM",
                    "provider": "openai-compatible",
                    "preset": "vllm",
                    "base_url": "http://127.0.0.1:8001/v1",
                    "model": "local-research-4k",
                    "context_length": 4096,
                    "max_output_tokens": 1024,
                    "streaming": True,
                },
            )
            assert create_conn.status_code == status.HTTP_201_CREATED
            conn_id = create_conn.json()["id"]

            test_conn = await client.post(f"/api/v1/llm/connections/{conn_id}/test")
            assert test_conn.status_code == 200
            assert test_conn.json()["ok"] is True

            activate_resp = await client.post(f"/api/v1/llm/connections/{conn_id}/activate")
            assert activate_resp.status_code == 200

            # 3. GET /api/v1/context/capabilities inspects active model limits & approximate tokenizer status
            caps_resp = await client.get("/api/v1/context/capabilities?refresh=true")
            assert caps_resp.status_code == 200
            caps_data = caps_resp.json()
            assert caps_data["capabilities"]["model_name"] == "local-research-4k"
            assert caps_data["capabilities"]["max_context_tokens"] == 4096
            assert caps_data["capabilities"]["tokenizer_available"] is False
            assert caps_data["capabilities"]["estimation_is_approximate"] is True
            assert "deep_research" in caps_data["available_profiles"]

            # 4. GET & PUT /api/v1/context/budget + verify limit enforcement (422)
            budget_resp = await client.get("/api/v1/context/budget")
            assert budget_resp.status_code == 200
            assert budget_resp.json()["effective_allocation"]["effective_context_limit"] == 4096

            bad_budget = await client.put(
                "/api/v1/context/budget",
                json={"user_context_limit": 16384},  # exceeds verified 4096 limit
            )
            assert bad_budget.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

            update_budget = await client.put(
                "/api/v1/context/budget",
                json={
                    "profile": "deep_research",
                    "user_context_limit": 3072,
                    "compression_strategy": "structured",
                },
            )
            assert update_budget.status_code == 200
            upd_data = update_budget.json()
            assert upd_data["config"]["profile"] == "deep_research"
            assert upd_data["effective_allocation"]["effective_context_limit"] == 3072

            # 5. Execute offline Stage 3+4 research workflow via POST /api/v1/research
            res_resp = await client.post(
                "/api/v1/research",
                json={
                    "query": "What does PEP 703 free-threaded CPython change for local concurrency?",
                    "max_results_per_query": 4,
                    "max_fetch_urls": 4,
                },
            )
            assert res_resp.status_code == 200
            research_pkg = res_resp.json()
            research_id = research_pkg["research_id"]

            # 6. Add user-controlled persistent workspace note & session conversation turns
            note_resp = await client.post(
                "/api/v1/context/notes",
                json={
                    "title": "Evaluation Criterion",
                    "content": "Highlight GIL removal compatibility and memory overhead metrics.",
                },
            )
            assert note_resp.status_code == status.HTTP_201_CREATED
            note_id = note_resp.json()["note_id"]

            turn_resp = await client.post(
                f"/api/v1/context/history/{research_id}",
                json={
                    "role": "user",
                    "content": "We decided to focus on CPython 3.13+ benchmarks. Unresolved: What overhead remains single-threaded?",
                },
            )
            assert turn_resp.status_code == status.HTTP_201_CREATED

            # 7. GET /api/v1/context/{research_id}
            ctx_run_resp = await client.get(f"/api/v1/context/{research_id}")
            assert ctx_run_resp.status_code == 200
            preview = ContextPreviewResponse.model_validate(ctx_run_resp.json())
            assert preview.validation.is_valid is True
            assert preview.context_package.total_with_output_tokens <= 3072
            assert len(preview.context_package.source_references) == len(
                preview.context_package.selected_evidence
            )
            assert len(preview.context_package.history.workspace_notes) == 1

            # 8. POST /api/v1/context/preview (standalone custom budget preview + adapter messages)
            preview_resp = await client.post(
                "/api/v1/context/preview",
                json={
                    "research_id": research_id,
                    "profile": "quick_answer",
                    "compression_strategy": "deduplication",
                    "conversation_history": [
                        {
                            "role": "user",
                            "scope": "current_turn",
                            "content": "Give concise verified summary with citations.",
                        }
                    ],
                },
            )
            assert preview_resp.status_code == 200
            prev_json = preview_resp.json()
            assert prev_json["context_package"]["compression"]["strategy_applied"] == "deduplication"
            assert prev_json["validation"]["is_valid"] is True
            assert len(prev_json["adapter_messages"]) == 2

            # 9. POST /api/v1/context/validate (reject overflow cleanly)
            overflow_pkg = dict(prev_json["context_package"])
            overflow_pkg["total_input_tokens"] = 9999
            bad_val = await client.post(
                "/api/v1/context/validate",
                json={"context_package": overflow_pkg},
            )
            assert bad_val.status_code == 200
            val_report = bad_val.json()
            assert val_report["is_valid"] is False
            assert any(i["code"] == "CONTEXT_LIMIT_EXCEEDED" for i in val_report["issues"])

            # 10. POST /api/v1/context/compress preserves original evidence inspectably
            comp_resp = await client.post(
                "/api/v1/context/compress",
                json={
                    "research_id": research_id,
                    "strategy": "structured",
                },
            )
            assert comp_resp.status_code == 200
            comp_json = comp_resp.json()
            assert comp_json["compression"]["strategy_applied"] == "structured"
            assert comp_json["compression"]["provenance_preserved"] is True
            assert len(comp_json["original_evidence"]) == len(research_pkg["evidence"])

            # 11. End-to-end demo: convert validated ContextPackage & send via active LocalModelAdapter
            adapter = await app.state.container.llm_manager.get_active_adapter()
            inference_req = ContextAdapterConverter.to_inference_request(
                preview.context_package,
                prefer_json=False,
            )
            llm_out = await adapter.chat(inference_req)
            assert "[E1]" in llm_out.content
            assert len(recorded_chat_requests) == 1
            assert recorded_chat_requests[0]["max_tokens"] == preview.context_package.output_reservation_tokens

            # 12. Toggle & cleanup workspace note + 404 check on missing research run
            patch_note = await client.patch(
                f"/api/v1/context/notes/{note_id}",
                json={"enabled": False},
            )
            assert patch_note.status_code == 200
            assert patch_note.json()["enabled"] is False

            del_note = await client.delete(f"/api/v1/context/notes/{note_id}")
            assert del_note.status_code == status.HTTP_204_NO_CONTENT

            missing_ctx = await client.get("/api/v1/context/non-existent-run-id")
            assert missing_ctx.status_code == status.HTTP_404_NOT_FOUND
