"""Comprehensive tests for OpenAICompatibleAdapter, OllamaAdapter, Registry, and ConnectionManager."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from web_intelligence.app.llm.base import (
    LLMAuthError,
    LLMResponseError,
    LLMTimeoutError,
    UniversalLocalModelAdapter,
)
from web_intelligence.app.llm.manager import (
    ConnectionNotFoundError,
    LLMConnectionManager,
    NoActiveConnectionError,
)
from web_intelligence.app.llm.registry import AdapterRegistry, UnsupportedProviderError
from web_intelligence.app.llm.security import LLMConnectionSecurityError
from web_intelligence.app.schemas.llm import (
    CapabilitySource,
    ChatMessage,
    ConnectionStatus,
    DiscoveredModel,
    LLMConnectionConfig,
    LLMConnectionCreate,
    LLMConnectionUpdate,
    LLMInferenceRequest,
    LLMInferenceResponse,
    LLMStreamChunk,
    ModelCapabilities,
    TestPromptRequest,
    ToolDefinition,
)
from web_intelligence.app.storage.database import DatabaseManager
from web_intelligence.app.storage.repositories import ResearchRepository


@pytest.mark.asyncio
async def test_openai_compatible_discovery_chat_streaming_and_strict_capabilities() -> None:
    seen_requests: list[tuple[str, str, dict | None, dict[str, str]]] = []
    secret_key = "vllm-super-secret-abc123"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode()) if request.content else None
        seen_requests.append((request.method, str(request.url), body, dict(request.headers)))

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
                        },
                        {
                            "id": "minimal-gguf",
                            "object": "model",
                            "owned_by": "llama.cpp",
                        },
                    ],
                },
            )

        if request.url.path == "/v1/chat/completions":
            if body and body.get("stream"):
                sse_body = (
                    ": keep-alive\n\n"
                    'data: {"id":"1","model":"Qwen3-4B","choices":[{"index":0,"delta":{"content":"Local "},"finish_reason":null}]}\n\n'
                    'data: {"id":"1","model":"Qwen3-4B","choices":[{"index":0,"delta":{"content":"evidence."},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}}\n\n'
                    "data: [DONE]\n\n"
                )
                return httpx.Response(
                    200,
                    content=sse_body.encode("utf-8"),
                    headers={"Content-Type": "text/event-stream"},
                )

            # Simulate server rejecting unsupported response_format on plain model
            if body and body.get("response_format") and body.get("model") == "minimal-gguf":
                return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})

            if body and body.get("tools"):
                return httpx.Response(
                    200,
                    json={
                        "model": "Qwen3-4B",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "search_web",
                                                "arguments": '{"query":"local ai"}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    },
                )

            return httpx.Response(
                200,
                json={
                    "model": body.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": '{"subqueries": ["local ai retrieval"]}',
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29},
                },
            )

        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    registry = AdapterRegistry()
    cfg = LLMConnectionConfig(
        id="vllm-1",
        name="Local vLLM",
        provider="openai-compatible",
        base_url="http://localhost:8000/v1",
        api_key=SecretStr(secret_key),
        model="Qwen3-4B",
        context_length=16384,
        max_output_tokens=2048,
        streaming=True,
    )
    adapter = registry.create(cfg, transport=transport)

    ok, msg, models, caps = await adapter.test_connection()
    assert ok is True
    assert len(models) == 2
    assert caps.max_context_tokens == 32768
    assert caps.sources.context_length == CapabilitySource.VERIFIED
    assert caps.supports_tool_calling is True
    assert caps.sources.tool_calling == CapabilitySource.VERIFIED
    assert caps.supports_json_mode is True
    assert caps.sources.json_mode == CapabilitySource.VERIFIED
    assert caps.supports_vision is False
    assert caps.sources.vision == CapabilitySource.UNKNOWN

    # Minimal model without metadata MUST NOT invent OpenAI features
    minimal_caps = await adapter.inspect_model_capabilities("minimal-gguf")
    assert minimal_caps.max_context_tokens == 16384
    assert minimal_caps.sources.context_length == CapabilitySource.CONFIGURED
    assert minimal_caps.supports_tool_calling is False
    assert minimal_caps.sources.tool_calling == CapabilitySource.UNKNOWN
    assert minimal_caps.supports_json_mode is False
    assert minimal_caps.sources.json_mode == CapabilitySource.UNKNOWN

    # Authorization header was sent cleanly without proxy interference
    assert seen_requests[0][3].get("authorization") == f"Bearer {secret_key}"

    # Tool-call parsing
    tool_res = await adapter.chat(
        LLMInferenceRequest(
            messages=[ChatMessage(role="user", content="Search local ai")],
            tools=[ToolDefinition(name="search_web", parameters={"type": "object"})],
        )
    )
    assert len(tool_res.tool_calls) == 1
    assert tool_res.tool_calls[0].name == "search_web"
    assert tool_res.tool_calls[0].arguments == {"query": "local ai"}

    # Streaming SSE
    chunks = [
        chunk
        async for chunk in adapter.stream_chat(
            LLMInferenceRequest(
                messages=[ChatMessage(role="user", content="Hi")],
                stream=True,
            )
        )
    ]
    assert "".join(c.delta for c in chunks) == "Local evidence."
    assert chunks[-1].done is True

    # JSON helper fallback when minimal model rejects response_format
    adapter_minimal = registry.create(
        cfg.model_copy(
            update={
                "model": "minimal-gguf",
                "capabilities": ModelCapabilities(
                    model_name="minimal-gguf",
                    backend="openai-compatible",
                    supports_json_mode=True,
                ),
            }
        ),
        transport=transport,
    )
    parsed = await adapter_minimal.generate_json([ChatMessage(role="user", content="JSON please")])
    assert parsed == {"subqueries": ["local ai retrieval"]}


@pytest.mark.asyncio
async def test_openai_compatible_manual_model_when_discovery_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(404, text="Not Found")
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    registry = AdapterRegistry()
    cfg = LLMConnectionConfig(
        id="manual-1",
        name="Minimal Server",
        provider="openai-compatible",
        base_url="http://127.0.0.1:8080",
        model="custom-finetune",
        context_length=8192,
        max_output_tokens=1024,
    )
    adapter = registry.create(cfg, transport=httpx.MockTransport(handler))
    ok, msg, models, caps = await adapter.test_connection()
    assert ok is True
    assert models == []
    assert "manually" in msg.lower()
    assert caps.model_name == "custom-finetune"
    assert caps.sources.context_length == CapabilitySource.CONFIGURED


@pytest.mark.asyncio
async def test_ollama_discovery_verified_capabilities_chat_and_ndjson_streaming() -> None:
    captured_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen3:4b",
                            "model": "qwen3:4b",
                            "details": {
                                "family": "qwen3",
                                "parameter_size": "4.0B",
                                "quantization_level": "Q4_K_M",
                            },
                        }
                    ]
                },
            )
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "details": {"family": "qwen3", "parameter_size": "4.0B"},
                    "model_info": {
                        "general.architecture": "qwen3",
                        "qwen3.context_length": 40960,
                    },
                    "capabilities": ["completion", "tools"],
                },
            )
        if request.url.path == "/api/chat":
            body = json.loads(request.content.decode())
            captured_payloads.append(body)
            if body.get("stream"):
                ndjson = (
                    '{"model":"qwen3:4b","message":{"role":"assistant","content":"Hello "},"done":false}\n'
                    '{"model":"qwen3:4b","message":{"role":"assistant","content":"world!"},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":3}\n'
                )
                return httpx.Response(200, content=ndjson.encode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "model": "qwen3:4b",
                    "message": {
                        "role": "assistant",
                        "content": '{"answer": "verified", "confidence": 0.9}',
                    },
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 30,
                    "eval_count": 12,
                },
            )
        return httpx.Response(404)

    registry = AdapterRegistry()
    cfg = LLMConnectionConfig(
        id="ollama-1",
        name="Ollama Local",
        provider="ollama",
        base_url="http://localhost:11434/v1",  # should normalize cleanly for native endpoints
        model="qwen3:4b",
        context_length=16384,
        max_output_tokens=1024,
    )
    adapter = registry.create(cfg, transport=httpx.MockTransport(handler))

    ok, msg, models, caps = await adapter.test_connection()
    assert ok is True
    assert len(models) == 1
    assert models[0].parameter_size == "4.0B"
    assert caps.max_context_tokens == 40960
    assert caps.sources.context_length == CapabilitySource.VERIFIED
    assert caps.supports_tool_calling is True
    assert caps.sources.tool_calling == CapabilitySource.VERIFIED
    assert caps.supports_vision is False
    assert caps.sources.vision == CapabilitySource.VERIFIED

    res = await adapter.chat(
        LLMInferenceRequest(
            messages=[ChatMessage(role="user", content="Return JSON")],
            response_format="json_object",
        )
    )
    assert res.parsed_json == {"answer": "verified", "confidence": 0.9}
    assert captured_payloads[-1]["format"] == "json"
    assert captured_payloads[-1]["options"]["num_ctx"] == 16384

    streamed = [
        chunk
        async for chunk in adapter.stream_chat(
            LLMInferenceRequest(messages=[ChatMessage(role="user", content="Hi")], stream=True)
        )
    ]
    assert "".join(c.delta for c in streamed) == "Hello world!"
    assert streamed[-1].done is True
    assert streamed[-1].usage is not None and streamed[-1].usage.total_tokens == 13


@pytest.mark.asyncio
async def test_adapter_error_handling_redirects_timeouts_and_secret_redaction() -> None:
    secret_val = "sk-local-secret-998877"

    def redirect_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest"})

    cfg = LLMConnectionConfig(
        id="err-1",
        name="Error Test",
        provider="openai-compatible",
        base_url="http://localhost:8000/v1",
        api_key=SecretStr(secret_val),
        model="test-model",
    )
    registry = AdapterRegistry()

    with pytest.raises(LLMConnectionSecurityError):
        await registry.create(cfg, transport=httpx.MockTransport(redirect_handler)).chat(
            LLMInferenceRequest(messages=[ChatMessage(role="user", content="hi")])
        )

    def auth_err_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"Invalid key {secret_val}"}})

    with pytest.raises(LLMAuthError) as exc_info:
        await registry.create(cfg, transport=httpx.MockTransport(auth_err_handler)).list_models()
    assert secret_val not in str(exc_info.value)

    def timeout_handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(LLMTimeoutError):
        await registry.create(cfg, transport=httpx.MockTransport(timeout_handler)).chat(
            LLMInferenceRequest(messages=[ChatMessage(role="user", content="hi")])
        )

    def bad_json_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(LLMResponseError):
        await registry.create(cfg, transport=httpx.MockTransport(bad_json_handler)).chat(
            LLMInferenceRequest(messages=[ChatMessage(role="user", content="hi")])
        )


def test_registry_allows_future_custom_adapters() -> None:
    class DummyCustomAdapter(UniversalLocalModelAdapter):
        provider_id = "custom-grpc"

        async def test_connection(self):
            return True, "ok", [], self.capabilities()

        async def list_models(self):
            return [], False, "manual"

        async def inspect_model_capabilities(self, model_name=None):
            return self.capabilities()

        async def chat(self, request: LLMInferenceRequest) -> LLMInferenceResponse:
            return LLMInferenceResponse(content="custom-ok", model=self.config.model)

        async def stream_chat(self, request: LLMInferenceRequest) -> AsyncIterator[LLMStreamChunk]:
            yield LLMStreamChunk(delta="custom-ok", done=True)

    registry = AdapterRegistry()
    registry.register("custom-grpc", DummyCustomAdapter)
    assert registry.supports("custom-grpc") is True

    with pytest.raises(UnsupportedProviderError):
        registry.create(
            LLMConnectionConfig(
                id="x",
                name="Unknown",
                provider="nonexistent",
                base_url="http://localhost:9000",
            )
        )


@pytest.mark.asyncio
async def test_connection_manager_lifecycle_prompt_and_secret_redaction(tmp_path: Path) -> None:
    secret_key = "secret-lmstudio-key-777"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            if request.headers.get("x-fail"):
                return httpx.Response(500, text=f"crash with {secret_key}")
            return httpx.Response(
                200,
                json={"data": [{"id": "qwen3-8b", "context_length": 32768}]},
            )
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content.decode())
            if body.get("stream"):
                return httpx.Response(
                    200,
                    content=(
                        b'data: {"model":"qwen3-8b","choices":[{"delta":{"content":"Test "},"finish_reason":null}]}\n\n'
                        b'data: {"model":"qwen3-8b","choices":[{"delta":{"content":"passed"},"finish_reason":"stop"}]}\n\n'
                        b"data: [DONE]\n\n"
                    ),
                )
            return httpx.Response(
                200,
                json={
                    "model": "qwen3-8b",
                    "choices": [{"message": {"content": "Ready to assist."}, "finish_reason": "stop"}],
                },
            )
        return httpx.Response(404)

    db = DatabaseManager(tmp_path / "mgr.db")
    await db.initialize()
    repo = ResearchRepository(db)
    manager = LLMConnectionManager(repo, transport=httpx.MockTransport(handler))
    await manager.initialize()

    with pytest.raises(NoActiveConnectionError):
        await manager.get_adapter(None)

    created = await manager.create_connection(
        LLMConnectionCreate(
            name="LM Studio",
            provider="openai-compatible",
            preset="lm-studio",
            base_url="http://localhost:1234/v1",
            api_key=SecretStr(secret_key),
            model="",
            context_length=16384,
            max_output_tokens=2048,
            activate=True,
        )
    )
    assert created.has_api_key is True
    assert secret_key not in created.model_dump_json()

    test_res = await manager.test_connection(created.id)
    assert test_res.ok is True
    assert test_res.connection.status == ConnectionStatus.CONNECTED
    assert len(test_res.discovered_models) == 1

    selected = await manager.select_model(created.id, "qwen3-8b")
    assert selected.model == "qwen3-8b"
    assert selected.capabilities is not None
    assert selected.capabilities.sources.context_length == CapabilitySource.VERIFIED
    assert selected.capabilities.max_context_tokens == 32768

    prompt_res = await manager.run_test_prompt(
        created.id, TestPromptRequest(prompt="Say ready")
    )
    assert prompt_res.content == "Ready to assist."

    chunks = [
        chunk
        async for chunk in manager.stream_test_prompt(
            created.id, TestPromptRequest(prompt="Stream test", stream=True)
        )
    ]
    assert "".join(c.delta for c in chunks) == "Test passed"

    updated = await manager.update_connection(
        created.id, LLMConnectionUpdate(clear_api_key=True, max_output_tokens=4096)
    )
    assert updated.has_api_key is False
    assert updated.max_output_tokens == 4096

    await manager.disconnect_active()
    assert await manager.get_active_config() is None

    await manager.delete_connection(created.id)
    with pytest.raises(ConnectionNotFoundError):
        await manager.get_connection(created.id)
