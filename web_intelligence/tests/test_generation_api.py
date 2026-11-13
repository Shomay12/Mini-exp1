"""API route and end-to-end research -> context -> local LLM generation integration tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import status

from web_intelligence.app.core.config import Settings
from web_intelligence.app.main import create_app
from web_intelligence.app.providers.fixtures import (
    build_fixture_transport,
    fixture_dns_resolver,
)


def _parse_sse(raw_text: str) -> list[dict]:
    events: list[dict] = []
    current_event = "message"
    data_lines: list[str] = []
    for line in raw_text.splitlines():
        if not line:
            if data_lines:
                joined = "\n".join(data_lines)
                if joined != "[DONE]":
                    payload = json.loads(joined)
                    payload.setdefault("event", current_event)
                    events.append(payload)
            current_event = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if data_lines:
        joined = "\n".join(data_lines)
        if joined != "[DONE]":
            payload = json.loads(joined)
            payload.setdefault("event", current_event)
            events.append(payload)
    return events


async def _loopback_resolver(host: str, port: int) -> list[str]:
    del host, port
    return ["127.0.0.1"]


@pytest.mark.asyncio
async def test_generation_and_model_registry_end_to_end_api(tmp_path: Path) -> None:
    def llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "Qwen/Qwen3-4B",
                            "object": "model",
                            "owned_by": "vllm",
                            "max_model_len": 8192,
                            "max_output_tokens": 1024,
                            "supported_parameters": ["tools", "response_format"],
                        }
                    ],
                },
            )
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content.decode("utf-8"))
            if body.get("stream"):
                chunks = [
                    'data: {"model":"Qwen/Qwen3-4B","choices":[{"index":0,"delta":{"content":"Local hybrid search uses SQLite FTS5 [S1]. "}}]}\n\n',
                    'data: {"model":"Qwen/Qwen3-4B","choices":[{"index":0,"delta":{"content":"Retrieved evidence stays strictly cited [S2]."},"finish_reason":"stop"}],"usage":{"prompt_tokens":310,"completion_tokens":18,"total_tokens":328}}\n\n',
                    "data: [DONE]\n\n",
                ]
                return httpx.Response(
                    200,
                    content="".join(chunks).encode("utf-8"),
                    headers={"Content-Type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-api-1",
                    "model": "Qwen/Qwen3-4B",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Grounded non-stream answer citing [S1] and [S2].",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 300, "completion_tokens": 14, "total_tokens": 314},
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    settings = Settings(
        database_path=tmp_path / "gen_api.db",
        search_providers=["mock"],
        enable_mock_provider=True,
        per_host_min_interval_seconds=0.0,
    )
    app = create_app(
        settings,
        transport_factory=build_fixture_transport,
        dns_resolver=fixture_dns_resolver,
        llm_transport=httpx.MockTransport(llm_handler),
        llm_resolver=_loopback_resolver,
    )

    async with app.router.lifespan_context(app):
        asgi = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=asgi, base_url="http://testserver") as client:
            # Verify Stage 6 UI assets and Generation Inspector tab are served
            ui_resp = await client.get("/")
            assert ui_resp.status_code == status.HTTP_200_OK
            assert 'data-inspector-tab="generation"' in ui_resp.text
            assert (
                await client.get("/static/js/components/generation_workspace.js")
            ).status_code == status.HTTP_200_OK
            assert (
                await client.get("/static/js/components/generation_inspector.js")
            ).status_code == status.HTTP_200_OK

            # 1. Run mock web research first
            res_resp = await client.post(
                "/api/v1/research",
                json={"query": "SQLite FTS5 BM25 hybrid retrieval local LLM"},
            )
            assert res_resp.status_code == status.HTTP_200_OK
            research_id = res_resp.json()["research_id"]

            # 2. Generating without configured local model returns 409 Conflict
            no_model_resp = await client.post(
                "/api/v1/generation",
                json={"research_id": research_id},
            )
            assert no_model_resp.status_code == status.HTTP_409_CONFLICT

            # 3. Missing research run returns 404 even on stream preflight
            missing_stream = await client.post(
                "/api/v1/generation/stream",
                json={"research_id": "res-does-not-exist"},
            )
            assert missing_stream.status_code == status.HTTP_404_NOT_FOUND

            # 4. Create & test local LLM connection with secret API key
            conn_resp = await client.post(
                "/api/v1/llm/connections",
                json={
                    "name": "LocalvLLM",
                    "provider": "openai-compatible",
                    "base_url": "http://localhost:8000/v1",
                    "api_key": "sk-super-secret-api-test",
                    "model": "Qwen/Qwen3-4B",
                    "context_length": 8192,
                    "max_output_tokens": 1024,
                    "streaming": True,
                    "activate": True,
                },
            )
            assert conn_resp.status_code == status.HTTP_201_CREATED
            conn_id = conn_resp.json()["id"]
            assert "sk-super-secret-api-test" not in conn_resp.text

            test_resp = await client.post(f"/api/v1/llm/connections/{conn_id}/test")
            assert test_resp.status_code == status.HTTP_200_OK

            # 5. ModelRegistry API shows discovered model + supports manual model + role PATCH (including /)
            reg_list = await client.get("/api/v1/llm/registry")
            assert reg_list.status_code == status.HTTP_200_OK
            assert "sk-super-secret-api-test" not in reg_list.text
            entries = reg_list.json()
            assert any(e["model_id"] == "Qwen/Qwen3-4B" and e["is_active"] for e in entries)

            manual_resp = await client.post(
                "/api/v1/llm/registry",
                json={
                    "connection_id": conn_id,
                    "model_id": "custom/draft-2b",
                    "display_name": "Draft Outline 2B",
                    "context_limit": 4096,
                    "max_output_limit": 512,
                    "task_roles": ["outline"],
                },
            )
            assert manual_resp.status_code == status.HTTP_201_CREATED
            manual_entry_id = manual_resp.json()["entry_id"]

            role_resp = await client.patch(
                f"/api/v1/llm/registry/{manual_entry_id}/roles",
                json={"task_roles": ["outline", "synthesis"]},
            )
            assert role_resp.status_code == status.HTTP_200_OK
            assert role_resp.json()["task_roles"] == ["outline", "synthesis"]

            # 6. Standard generation endpoint works + persists run
            gen_resp = await client.post(
                "/api/v1/generation",
                json={
                    "research_id": research_id,
                    "template_type": "quick_answer",
                    "output_format": "markdown",
                    "answer_length": "short",
                },
            )
            assert gen_resp.status_code == status.HTTP_200_OK
            gen_data = gen_resp.json()
            assert gen_data["status"] == "completed"
            assert gen_data["citation_validation"]["passed"] is True
            assert [c["citation_id"] for c in gen_data["citations"]] == ["S1", "S2"]
            assert gen_data["usage"]["total_tokens"] == 314

            # 7. Real SSE streaming endpoint works cleanly
            stream_resp = await client.post(
                "/api/v1/generation/stream",
                json={
                    "research_id": research_id,
                    "template_type": "detailed_explanation",
                    "output_format": "markdown",
                },
            )
            assert stream_resp.status_code == status.HTTP_200_OK
            assert "text/event-stream" in stream_resp.headers["content-type"]
            sse_events = _parse_sse(stream_resp.text)
            assert sse_events[0]["event"] == "start"
            assert any(e["event"] == "delta" for e in sse_events)
            assert any(e["event"] == "citation" for e in sse_events)
            complete_ev = next(e for e in sse_events if e["event"] == "complete")
            stream_gen_id = complete_ev["generation_id"]
            assert complete_ev["generation"]["status"] == "completed"

            # 8. Persistence & history endpoints retrieve exact GenerationResult
            get_resp = await client.get(f"/api/v1/generation/{stream_gen_id}")
            assert get_resp.status_code == status.HTTP_200_OK
            assert get_resp.json()["generation_id"] == stream_gen_id

            hist_resp = await client.get(
                "/api/v1/generation", params={"research_id": research_id}
            )
            assert hist_resp.status_code == status.HTTP_200_OK
            assert len(hist_resp.json()) >= 2


@pytest.mark.asyncio
async def test_generation_api_live_cancel_and_long_form_stream(tmp_path: Path) -> None:
    gate = asyncio.Event()
    mode = "slow_stream"

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"model":"qwen","choices":[{"index":0,"delta":{"content":"First clause [S1] "}}]}\n\n'
            gate.set()
            await asyncio.sleep(5)
            yield b'data: {"model":"qwen","choices":[{"index":0,"delta":{"content":"never emitted"}}]}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen", "max_model_len": 8192}]})
        if mode == "slow_stream":
            return httpx.Response(
                200,
                stream=SlowStream(),
                headers={"Content-Type": "text/event-stream"},
            )
        chunks = [
            'data: {"model":"qwen","choices":[{"index":0,"delta":{"content":"Section text citing [S1]."}}]}\n\n',
            'data: {"model":"qwen","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":80,"completion_tokens":8,"total_tokens":88}}\n\n',
            "data: [DONE]\n\n",
        ]
        return httpx.Response(
            200,
            content="".join(chunks).encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    settings = Settings(
        database_path=tmp_path / "gen_cancel_api.db",
        search_providers=["mock"],
        enable_mock_provider=True,
        per_host_min_interval_seconds=0.0,
    )
    app = create_app(
        settings,
        transport_factory=build_fixture_transport,
        dns_resolver=fixture_dns_resolver,
        llm_transport=httpx.MockTransport(handler),
        llm_resolver=_loopback_resolver,
    )

    async with app.router.lifespan_context(app):
        asgi = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=asgi, base_url="http://testserver") as client:
            res_resp = await client.post("/api/v1/research", json={"query": "SQLite FTS5"})
            research_id = res_resp.json()["research_id"]

            await client.post(
                "/api/v1/llm/connections",
                json={
                    "name": "LocalSlow",
                    "provider": "openai-compatible",
                    "base_url": "http://localhost:8000/v1",
                    "model": "qwen",
                    "context_length": 8192,
                    "max_output_tokens": 512,
                    "streaming": True,
                    "activate": True,
                },
            )

            # Cancel mid-stream cleanly
            collected_events: list[dict] = []
            gen_id: str | None = None

            async def run_stream() -> None:
                nonlocal gen_id
                async with client.stream(
                    "POST",
                    "/api/v1/generation/stream",
                    json={"research_id": research_id},
                ) as resp:
                    assert resp.status_code == 200
                    current_ev = "message"
                    async for raw_line in resp.aiter_lines():
                        if raw_line.startswith("event:"):
                            current_ev = raw_line[6:].strip()
                        elif raw_line.startswith("data:"):
                            data_s = raw_line[5:].strip()
                            if data_s == "[DONE]":
                                continue
                            payload = json.loads(data_s)
                            payload.setdefault("event", current_ev)
                            collected_events.append(payload)
                            if payload.get("generation_id"):
                                gen_id = payload["generation_id"]

            task = asyncio.create_task(run_stream())
            await asyncio.wait_for(gate.wait(), timeout=2.0)
            # Poll REST generation list while ASGI stream is mid-flight to discover active run ID
            active_gen_id: str | None = None
            for _ in range(40):
                list_resp = await client.get(
                    "/api/v1/generation", params={"research_id": research_id}
                )
                assert list_resp.status_code == status.HTTP_200_OK
                streaming_runs = [
                    r for r in list_resp.json() if r.get("status") == "streaming"
                ]
                if streaming_runs:
                    active_gen_id = streaming_runs[0]["generation_id"]
                    break
                await asyncio.sleep(0.01)
            assert active_gen_id is not None

            cancel_resp = await client.post(f"/api/v1/generation/{active_gen_id}/cancel")
            assert cancel_resp.status_code == status.HTTP_200_OK
            await asyncio.wait_for(task, timeout=2.0)
            assert gen_id == active_gen_id

            cancelled_ev = next(e for e in collected_events if e["event"] == "cancelled")
            assert cancelled_ev["generation"]["partial"] is True
            assert "First clause" in cancelled_ev["generation"]["content"]

            # Long-form streaming emits outline, section_start, section_complete, complete
            mode = "long_form"
            lf_resp = await client.post(
                "/api/v1/generation/stream",
                json={
                    "research_id": research_id,
                    "generation_mode": "long_form_report",
                    "template_type": "research_report",
                    "long_form_config": {"max_sections": 3, "section_max_output_tokens": 128},
                },
            )
            assert lf_resp.status_code == status.HTTP_200_OK
            lf_events = _parse_sse(lf_resp.text)
            assert any(e["event"] == "outline" for e in lf_events)
            assert any(e["event"] == "section_start" for e in lf_events)
            assert any(e["event"] == "section_complete" for e in lf_events)
            lf_complete = next(e for e in lf_events if e["event"] == "complete")
            assert lf_complete["generation"]["mode"] == "long_form_report"
            assert len(lf_complete["generation"]["long_form"]["sections"]) == 3
