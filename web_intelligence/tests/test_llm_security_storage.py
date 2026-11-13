"""Tests for LLM endpoint security boundary, secret protection, and SQLite connection storage."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import SecretStr

from web_intelligence.app.core.logging import (
    SecretRedactingFilter,
    redact_text,
    register_secret,
)
from web_intelligence.app.llm.security import (
    LLMConnectionSecurityError,
    LLMURLValidationError,
    redact_llm_error,
    validate_api_endpoint_path,
    validate_llm_endpoint_async,
    validate_llm_endpoint_url,
)
from web_intelligence.app.schemas.llm import (
    CapabilityProvenance,
    CapabilitySource,
    ConnectionStatus,
    LLMConnectionConfig,
    LLMConnectionCreate,
    ModelCapabilities,
)
from web_intelligence.app.storage.database import DatabaseManager
from web_intelligence.app.storage.repositories import ResearchRepository


def test_loopback_endpoints_allowed_with_approval() -> None:
    assert (
        validate_llm_endpoint_url("http://localhost:11434/")
        == "http://localhost:11434"
    )
    assert (
        validate_llm_endpoint_url("http://127.0.0.1:8000/v1/")
        == "http://127.0.0.1:8000/v1"
    )
    assert (
        validate_llm_endpoint_url("http://[::1]:1234/v1")
        == "http://[::1]:1234/v1"
    )


def test_unapproved_or_malformed_urls_blocked() -> None:
    with pytest.raises(LLMConnectionSecurityError):
        validate_llm_endpoint_url("http://localhost:11434", approved_by_user=False)

    for bad_url in [
        "",
        "ftp://localhost:11434",
        "file:///etc/passwd",
        "http://user:pass@localhost:8000/v1",
        "http://localhost:8000/v1?token=leak",
        "http://localhost:8000/v1#frag",
        "http://localhost:8000/\nheader",
    ]:
        with pytest.raises(LLMURLValidationError):
            validate_llm_endpoint_url(bad_url)


def test_remote_endpoints_require_explicit_exception() -> None:
    with pytest.raises(LLMConnectionSecurityError):
        validate_llm_endpoint_url("http://192.168.1.40:11434", allow_remote_endpoint=False)

    with pytest.raises(LLMConnectionSecurityError):
        validate_llm_endpoint_url("https://vllm.corp.example/v1", allow_remote_endpoint=False)

    assert (
        validate_llm_endpoint_url("http://192.168.1.40:11434", allow_remote_endpoint=True)
        == "http://192.168.1.40:11434"
    )
    assert (
        validate_llm_endpoint_url("https://vllm.corp.example/v1", allow_remote_endpoint=True)
        == "https://vllm.corp.example/v1"
    )


@pytest.mark.parametrize(
    "blocked_url",
    [
        "http://169.254.169.254/latest/meta-data",
        "http://metadata.google.internal/computeMetadata/v1",
        "http://metadata.goog/v1",
        "http://[::ffff:169.254.169.254]:8000",
        "http://[fe80::1]:8000/v1",
        "http://0.0.0.0:11434",
        "http://localhost:5432",
        "http://localhost:6379",
        "http://localhost:22",
    ],
)
def test_metadata_link_local_and_sensitive_ports_always_blocked(blocked_url: str) -> None:
    with pytest.raises(LLMConnectionSecurityError):
        validate_llm_endpoint_url(blocked_url, allow_remote_endpoint=True)


def test_api_endpoint_override_prevents_host_escape() -> None:
    assert validate_api_endpoint_path("/chat/completions") == "/chat/completions"
    assert validate_api_endpoint_path("   ") is None

    for bad_path in [
        "chat/completions",
        "//evil.com/v1/chat",
        "https://evil.com/v1/chat",
        "/v1/../../etc/passwd",
        "/v1/chat?override=1",
        "/v1/user@evil",
    ]:
        with pytest.raises(LLMURLValidationError):
            validate_api_endpoint_path(bad_path)


@pytest.mark.asyncio
async def test_dns_rebinding_checks_block_metadata_and_unapproved_remote() -> None:
    async def rebind_metadata(_host: str, _port: int) -> list[str]:
        return ["169.254.169.254"]

    with pytest.raises(LLMConnectionSecurityError):
        await validate_llm_endpoint_async(
            "http://remote-llm.example:8000/v1",
            allow_remote_endpoint=True,
            resolver=rebind_metadata,
        )

    async def rebind_lan(_host: str, _port: int) -> list[str]:
        return ["192.168.1.55"]

    with pytest.raises(LLMConnectionSecurityError):
        await validate_llm_endpoint_async(
            "http://sneaky.localhost:8000/v1",
            allow_remote_endpoint=False,
            resolver=rebind_lan,
        )

    async def good_loopback(_host: str, _port: int) -> list[str]:
        return ["127.0.0.1"]

    clean, ips = await validate_llm_endpoint_async(
        "http://localhost:11434",
        resolver=good_loopback,
    )
    assert clean == "http://localhost:11434"
    assert ips == ["127.0.0.1"]


def test_secret_protection_and_public_dto_strip() -> None:
    secret_val = "local-llm-secret-token-98765"
    register_secret(secret_val)

    cfg = LLMConnectionConfig(
        id="conn-1",
        name="Local vLLM",
        provider="openai-compatible",
        base_url="http://localhost:8000/v1",
        api_key=SecretStr(secret_val),
        model="Qwen3-4B",
        context_length=32768,
        max_output_tokens=4096,
        streaming=True,
    )
    public = cfg.to_public()
    dumped = public.model_dump_json()
    assert public.has_api_key is True
    assert secret_val not in dumped
    assert '"api_key"' not in dumped  # raw field is absent on LLMConnectionPublic

    redacted = redact_llm_error(f"Auth failed header Authorization: Bearer {secret_val}")
    assert secret_val not in redacted
    assert "[REDACTED_SECRET]" in redacted

    # Verify logging filter masks registered runtime secrets
    assert secret_val not in redact_text(f"Request error with key={secret_val}")
    log_filter = SecretRedactingFilter([secret_val])
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=f"Failed using {secret_val}",
        args=(),
        exc_info=None,
    )
    assert log_filter.filter(record) is True
    assert secret_val not in record.msg


def test_token_budget_validation_on_connection_schemas() -> None:
    with pytest.raises(ValueError):
        LLMConnectionCreate(
            name="Bad Budget",
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            context_length=2048,
            max_output_tokens=2048,
        )


@pytest.mark.asyncio
async def test_sqlite_llm_connection_crud_and_active_switching(tmp_path: Path) -> None:
    db = DatabaseManager(tmp_path / "llm_test.db")
    await db.initialize()
    repo = ResearchRepository(db)

    conn_a = LLMConnectionConfig(
        id="ollama-local",
        name="Ollama Local",
        provider="ollama",
        preset="ollama",
        base_url="http://localhost:11434",
        model="qwen3:4b",
        context_length=16384,
        max_output_tokens=2048,
        streaming=True,
        is_active=True,
        status=ConnectionStatus.CONNECTED,
        capabilities=ModelCapabilities(
            model_name="qwen3:4b",
            backend="ollama",
            max_context_tokens=16384,
            recommended_output_tokens=2048,
            supports_streaming=True,
            supports_tool_calling=True,
            supports_json_mode=True,
            sources=CapabilityProvenance(
                context_length=CapabilitySource.VERIFIED,
                streaming=CapabilitySource.VERIFIED,
                tool_calling=CapabilitySource.VERIFIED,
            ),
        ),
    )
    conn_b = LLMConnectionConfig(
        id="vllm-local",
        name="vLLM Local",
        provider="openai-compatible",
        preset="vllm",
        base_url="http://127.0.0.1:8000/v1",
        api_key=SecretStr("vllm-local-key-xyz"),
        model="Qwen3-4B",
        context_length=32768,
        max_output_tokens=4096,
        streaming=True,
        is_active=True,  # Saving active B must deactivate A
    )

    await repo.save_llm_connection(conn_a)
    await repo.save_llm_connection(conn_b)

    listed = await repo.list_llm_connections()
    assert len(listed) == 2
    active = await repo.get_active_llm_connection()
    assert active is not None
    assert active.id == "vllm-local"
    assert active.api_key is not None
    assert active.api_key.get_secret_value() == "vllm-local-key-xyz"

    reloaded_a = await repo.get_llm_connection("ollama-local")
    assert reloaded_a is not None
    assert reloaded_a.is_active is False
    assert reloaded_a.capabilities is not None
    assert reloaded_a.capabilities.sources.context_length == CapabilitySource.VERIFIED

    # Switch active back to conn_a
    switched = await repo.set_active_llm_connection("ollama-local")
    assert switched is not None and switched.id == "ollama-local" and switched.is_active is True
    assert (await repo.get_active_llm_connection()).id == "ollama-local"  # type: ignore[union-attr]

    # Disconnect all
    assert await repo.set_active_llm_connection(None) is None
    assert await repo.get_active_llm_connection() is None

    # Delete
    assert await repo.delete_llm_connection("ollama-local") is True
    assert await repo.get_llm_connection("ollama-local") is None
