"""Tests for persistent Stage 6 ModelRegistry across multiple local servers and capabilities."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from web_intelligence.app.llm.manager import LLMConnectionManager
from web_intelligence.app.llm.model_registry import ModelRegistry, sanitize_endpoint_for_registry
from web_intelligence.app.schemas.generation import ModelRegistryRegisterRequest, ModelTaskRole
from web_intelligence.app.schemas.llm import CapabilitySource, ConnectionStatus, LLMConnectionCreate
from web_intelligence.app.storage.database import DatabaseManager
from web_intelligence.app.storage.repositories import ResearchRepository


@pytest.mark.asyncio
async def test_model_registry_multi_server_sync_capabilities_roles_and_zero_secret_exposure(
    tmp_path: Path,
) -> None:
    secret_vllm = "sk-local-vllm-secret-999"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith("http://localhost:8000/v1/models"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "Qwen3-4B",
                            "max_model_len": 32768,
                            "supported_parameters": ["tools", "response_format"],
                        },
                        {
                            "id": "Qwen3-1.7B",
                            "max_model_len": 16384,
                        },
                    ],
                },
            )
        if str(request.url).startswith("http://localhost:11434/api/tags"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "llama3.2:3b",
                            "details": {
                                "family": "llama",
                                "parameter_size": "3B",
                                "quantization_level": "Q4_K_M",
                            },
                        }
                    ]
                },
            )
        if str(request.url).startswith("http://localhost:11434/api/show"):
            return httpx.Response(
                200,
                json={
                    "details": {
                        "family": "llama",
                        "parameter_size": "3B",
                        "quantization_level": "Q4_K_M",
                    },
                    "model_info": {"llama.context_length": 131072},
                },
            )
        if str(request.url).startswith("http://localhost:8080/v1/models"):
            return httpx.Response(404)
        if str(request.url).startswith("http://localhost:8080/health"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    db = DatabaseManager(tmp_path / "registry.db")
    await db.initialize()
    repo = ResearchRepository(db)
    registry = ModelRegistry(repo)
    manager = LLMConnectionManager(
        repo,
        model_registry=registry,
        transport=httpx.MockTransport(handler),
    )
    await manager.initialize()

    vllm_conn = await manager.create_connection(
        LLMConnectionCreate(
            name="vLLM Primary",
            provider="openai-compatible",
            preset="vllm",
            base_url="http://localhost:8000/v1",
            api_key=SecretStr(secret_vllm),
            model="Qwen3-4B",
            context_length=16384,
            max_output_tokens=2048,
            activate=True,
        )
    )
    ollama_conn = await manager.create_connection(
        LLMConnectionCreate(
            name="Ollama Local",
            provider="ollama",
            preset="ollama",
            base_url="http://localhost:11434",
            model="llama3.2:3b",
            context_length=8192,
            max_output_tokens=1024,
            activate=False,
        )
    )
    llamacpp_conn = await manager.create_connection(
        LLMConnectionCreate(
            name="llama.cpp Manual",
            provider="openai-compatible",
            preset="llamacpp",
            base_url="http://localhost:8080/v1",
            model="gemma-manual",
            context_length=8192,
            max_output_tokens=1024,
            activate=False,
        )
    )

    test_vllm = await manager.test_connection(vllm_conn.id)
    assert test_vllm.ok is True
    test_ol = await manager.test_connection(ollama_conn.id)
    assert test_ol.ok is True
    test_cpp = await manager.test_connection(llamacpp_conn.id)
    assert test_cpp.ok is True

    entries = await registry.list_models()
    assert len(entries) == 4

    # Secrets must never appear anywhere in ModelRegistry serialized payloads
    serialized_all = json.dumps([e.model_dump(mode="json") for e in entries])
    assert secret_vllm not in serialized_all

    # Check verified vs configured capability provenance
    qwen4b = next(e for e in entries if e.model_id == "Qwen3-4B")
    assert qwen4b.context_limit == 32768
    assert qwen4b.capability_provenance.context_length == CapabilitySource.VERIFIED
    assert qwen4b.supports_json_mode is True
    assert qwen4b.supports_tool_calling is True
    assert qwen4b.connection_status == ConnectionStatus.CONNECTED
    assert qwen4b.last_connected_at is not None
    assert qwen4b.is_active is True

    llama3 = next(e for e in entries if e.model_id == "llama3.2:3b")
    assert llama3.model_metadata.get("family") == "llama"
    assert llama3.model_metadata.get("quantization_level") == "Q4_K_M"
    assert llama3.is_active is False

    gemma_manual = next(e for e in entries if e.model_id == "gemma-manual")
    assert gemma_manual.capability_provenance.context_length == CapabilitySource.CONFIGURED
    assert gemma_manual.is_discovered is False

    # Manual registration on non-discovery server + per-task role routing
    cfg_cpp = await repo.get_llm_connection(llamacpp_conn.id)
    assert cfg_cpp is not None
    added_manual = await registry.register_manual_model(
        cfg_cpp,
        ModelRegistryRegisterRequest(
            connection_id=llamacpp_conn.id,
            model_id="phi4-outline",
            display_name="Phi4 Outline Specialist",
            context_limit=16384,
            max_output_limit=2048,
            task_roles=[ModelTaskRole.OUTLINE],
        ),
    )
    assert added_manual.context_limit == 16384
    assert ModelTaskRole.OUTLINE in added_manual.task_roles

    resolved_outline = await registry.resolve_target(task_role=ModelTaskRole.OUTLINE)
    assert resolved_outline is not None
    assert resolved_outline.model_id == "phi4-outline"

    resolved_active = await registry.resolve_target()
    assert resolved_active is not None
    assert resolved_active.model_id == "Qwen3-4B"

    # Assign role to llama3 and resolve by role
    await registry.assign_task_roles(llama3.entry_id, [ModelTaskRole.EXTRACTION])
    resolved_extract = await registry.resolve_target(task_role=ModelTaskRole.EXTRACTION)
    assert resolved_extract is not None
    assert resolved_extract.model_id == "llama3.2:3b"

    # Endpoint sanitizer strips embedded URL userinfo/query
    assert (
        sanitize_endpoint_for_registry("http://user:pass@localhost:8000/v1?token=abc")
        == "http://localhost:8000/v1"
    )

    # Deleting connection removes associated registry records cleanly
    await manager.delete_connection(ollama_conn.id)
    remaining = await registry.list_models()
    assert all(e.connection_id != ollama_conn.id for e in remaining)
