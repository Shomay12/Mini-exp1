"""Connection lifecycle manager coordinating SQLite config, security policy, and adapters."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import httpx

from web_intelligence.app.core.logging import get_logger, register_secret, register_secrets
from web_intelligence.app.core.security import AsyncResolver
from web_intelligence.app.llm.base import UniversalLocalModelAdapter
from web_intelligence.app.llm.registry import AdapterRegistry, UnsupportedProviderError
from web_intelligence.app.llm.security import (
    LLMConnectionSecurityError,
    redact_llm_error,
    validate_api_endpoint_path,
    validate_llm_endpoint_url,
)
from web_intelligence.app.schemas.llm import (
    CapabilitySource,
    ChatMessage,
    ConnectionStatus,
    ConnectionTestResponse,
    LLMConnectionConfig,
    LLMConnectionCreate,
    LLMConnectionPublic,
    LLMConnectionSummary,
    LLMConnectionUpdate,
    LLMInferenceRequest,
    LLMStreamChunk,
    ModelDiscoveryResponse,
    TestPromptRequest,
    TestPromptResponse,
)
from web_intelligence.app.storage.repositories import ResearchRepository

logger = get_logger("llm.manager")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class ConnectionNotFoundError(LookupError):
    """Raised when a requested LLM connection ID does not exist."""


class NoActiveConnectionError(RuntimeError):
    """Raised when an operation requires an active LLM connection and none is attached."""


def _generate_connection_id(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")[:24] or "llm"
    return f"{slug}-{uuid.uuid4().hex[:8]}"


class LLMConnectionManager:
    """High-level service managing user-hosted LLM endpoint configs and runtime adapters."""

    def __init__(
        self,
        repository: ResearchRepository,
        *,
        registry: AdapterRegistry | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: AsyncResolver | None = None,
    ) -> None:
        self.repository = repository
        self.registry = registry or AdapterRegistry()
        self._transport = transport
        self._resolver = resolver

    async def initialize(self) -> None:
        """Load persisted connection secrets into log redaction filter at startup."""
        connections = await self.repository.list_llm_connections()
        secrets = [
            cfg.api_key.get_secret_value()
            for cfg in connections
            if cfg.api_key is not None and cfg.api_key.get_secret_value().strip()
        ]
        if secrets:
            register_secrets(secrets)

    def _validate_provider(self, provider: str) -> str:
        normalized = provider.strip().lower()
        if not self.registry.supports(normalized):
            raise UnsupportedProviderError(
                f"Unsupported LLM provider '{provider}'. "
                f"Supported: {', '.join(self.registry.supported_providers())}"
            )
        return normalized

    async def list_connections(self) -> list[LLMConnectionPublic]:
        """Return frontend-safe representations of all stored LLM connections."""
        configs = await self.repository.list_llm_connections()
        return [cfg.to_public() for cfg in configs]

    async def get_connection(self, connection_id: str) -> LLMConnectionPublic:
        """Return a single frontend-safe connection DTO or raise ConnectionNotFoundError."""
        cfg = await self.repository.get_llm_connection(connection_id)
        if cfg is None:
            raise ConnectionNotFoundError(f"LLM connection '{connection_id}' was not found.")
        return cfg.to_public()

    async def get_active_config(self) -> LLMConnectionConfig | None:
        """Return internal active connection configuration, if attached."""
        return await self.repository.get_active_llm_connection()

    async def get_active_summary(self) -> LLMConnectionSummary | None:
        """Return compact active LLM status summary for health checks and toolbar."""
        cfg = await self.repository.get_active_llm_connection()
        return cfg.to_summary() if cfg is not None else None

    async def get_adapter(
        self, connection_id: str | None = None
    ) -> tuple[LLMConnectionConfig, UniversalLocalModelAdapter]:
        """Resolve either the named connection or active connection into a ready adapter."""
        if connection_id:
            cfg = await self.repository.get_llm_connection(connection_id)
            if cfg is None:
                raise ConnectionNotFoundError(
                    f"LLM connection '{connection_id}' was not found."
                )
        else:
            cfg = await self.repository.get_active_llm_connection()
            if cfg is None:
                raise NoActiveConnectionError(
                    "No active local LLM connection is attached."
                )

        if not cfg.approved_by_user:
            raise LLMConnectionSecurityError(
                f"Connection '{cfg.name}' is not approved for outbound inference."
            )

        validate_llm_endpoint_url(
            cfg.base_url,
            allow_remote_endpoint=cfg.allow_remote_endpoint,
            approved_by_user=cfg.approved_by_user,
            api_endpoint=cfg.api_endpoint,
        )
        adapter = self.registry.create(
            cfg, transport=self._transport, resolver=self._resolver
        )
        return cfg, adapter

    async def get_active_adapter(self) -> UniversalLocalModelAdapter | None:
        """Return adapter for active connection or None when operating zero-LLM."""
        cfg = await self.repository.get_active_llm_connection()
        if cfg is None or not cfg.approved_by_user:
            return None
        validate_llm_endpoint_url(
            cfg.base_url,
            allow_remote_endpoint=cfg.allow_remote_endpoint,
            approved_by_user=cfg.approved_by_user,
            api_endpoint=cfg.api_endpoint,
        )
        return self.registry.create(cfg, transport=self._transport, resolver=self._resolver)

    async def create_connection(
        self, payload: LLMConnectionCreate
    ) -> LLMConnectionPublic:
        """Validate, register secret, and persist a new LLM endpoint connection."""
        provider = self._validate_provider(payload.provider)
        clean_url = validate_llm_endpoint_url(
            payload.base_url,
            allow_remote_endpoint=payload.allow_remote_endpoint,
            approved_by_user=payload.approved_by_user,
            api_endpoint=payload.api_endpoint,
        )
        clean_endpoint = validate_api_endpoint_path(payload.api_endpoint)
        conn_id = (payload.id or "").strip() or _generate_connection_id(payload.name)

        if payload.api_key is not None:
            raw_key = payload.api_key.get_secret_value().strip()
            if raw_key:
                register_secret(raw_key)

        now = datetime.now(timezone.utc)
        config = LLMConnectionConfig(
            id=conn_id,
            name=payload.name.strip(),
            provider=provider,
            preset=payload.preset,
            base_url=clean_url,
            api_endpoint=clean_endpoint,
            api_key=payload.api_key,
            model=payload.model.strip(),
            context_length=payload.context_length,
            max_output_tokens=payload.max_output_tokens,
            request_timeout_seconds=payload.request_timeout_seconds,
            streaming=payload.streaming,
            allow_remote_endpoint=payload.allow_remote_endpoint,
            approved_by_user=payload.approved_by_user,
            is_active=payload.activate,
            status=ConnectionStatus.CONFIGURED,
            created_at=now,
            updated_at=now,
        )
        saved = await self.repository.save_llm_connection(config)
        logger.info(
            "Saved LLM connection configuration",
            extra={
                "extra_data": {
                    "connection_id": saved.id,
                    "provider": saved.provider,
                    "remote": saved.allow_remote_endpoint,
                }
            },
        )
        return saved.to_public()

    async def update_connection(
        self, connection_id: str, payload: LLMConnectionUpdate
    ) -> LLMConnectionPublic:
        """Modify an existing connection with full security and token-budget validation."""
        existing = await self.repository.get_llm_connection(connection_id)
        if existing is None:
            raise ConnectionNotFoundError(f"LLM connection '{connection_id}' was not found.")

        provider = (
            self._validate_provider(payload.provider)
            if payload.provider is not None
            else existing.provider
        )
        allow_remote = (
            payload.allow_remote_endpoint
            if payload.allow_remote_endpoint is not None
            else existing.allow_remote_endpoint
        )
        approved = (
            payload.approved_by_user
            if payload.approved_by_user is not None
            else existing.approved_by_user
        )
        raw_base_url = payload.base_url if payload.base_url is not None else existing.base_url
        raw_api_endpoint = (
            payload.api_endpoint
            if "api_endpoint" in payload.model_fields_set
            else existing.api_endpoint
        )
        clean_url = validate_llm_endpoint_url(
            raw_base_url,
            allow_remote_endpoint=allow_remote,
            approved_by_user=approved,
            api_endpoint=raw_api_endpoint,
        )
        clean_endpoint = validate_api_endpoint_path(raw_api_endpoint)

        api_key = existing.api_key
        if payload.clear_api_key:
            api_key = None
        elif payload.api_key is not None:
            raw_key = payload.api_key.get_secret_value().strip()
            if raw_key:
                register_secret(raw_key)
                api_key = payload.api_key

        next_ctx = (
            payload.context_length
            if payload.context_length is not None
            else existing.context_length
        )
        next_out = (
            payload.max_output_tokens
            if payload.max_output_tokens is not None
            else existing.max_output_tokens
        )
        next_streaming = (
            payload.streaming if payload.streaming is not None else existing.streaming
        )
        next_model = payload.model.strip() if payload.model is not None else existing.model

        # Refresh configured capability fields without overwriting verified provenance
        next_caps = existing.capabilities
        if next_caps is not None:
            updates: dict[str, object] = {
                "model_name": next_model or "unselected",
                "backend": provider,
            }
            if next_caps.sources.context_length != CapabilitySource.VERIFIED:
                updates["max_context_tokens"] = next_ctx
            if next_caps.sources.max_output_tokens != CapabilitySource.VERIFIED:
                updates["recommended_output_tokens"] = min(next_out, next_ctx - 1)
            if next_caps.sources.streaming != CapabilitySource.VERIFIED:
                updates["supports_streaming"] = next_streaming
            next_caps = next_caps.model_copy(update=updates)

        updated = LLMConnectionConfig(
            id=existing.id,
            name=payload.name.strip() if payload.name is not None else existing.name,
            provider=provider,
            preset=payload.preset if "preset" in payload.model_fields_set else existing.preset,
            base_url=clean_url,
            api_endpoint=clean_endpoint,
            api_key=api_key,
            model=next_model,
            context_length=next_ctx,
            max_output_tokens=next_out,
            request_timeout_seconds=(
                payload.request_timeout_seconds
                if payload.request_timeout_seconds is not None
                else existing.request_timeout_seconds
            ),
            streaming=next_streaming,
            allow_remote_endpoint=allow_remote,
            approved_by_user=approved,
            is_active=existing.is_active and approved,
            status=existing.status,
            last_error=existing.last_error,
            last_checked_at=existing.last_checked_at,
            capabilities=next_caps,
            created_at=existing.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        saved = await self.repository.save_llm_connection(updated)
        return saved.to_public()

    async def delete_connection(self, connection_id: str) -> bool:
        """Delete a stored LLM connection."""
        deleted = await self.repository.delete_llm_connection(connection_id)
        if not deleted:
            raise ConnectionNotFoundError(f"LLM connection '{connection_id}' was not found.")
        return True

    async def activate_connection(self, connection_id: str) -> LLMConnectionPublic:
        """Mark a validated connection as the active runtime LLM."""
        existing = await self.repository.get_llm_connection(connection_id)
        if existing is None:
            raise ConnectionNotFoundError(f"LLM connection '{connection_id}' was not found.")
        validate_llm_endpoint_url(
            existing.base_url,
            allow_remote_endpoint=existing.allow_remote_endpoint,
            approved_by_user=existing.approved_by_user,
            api_endpoint=existing.api_endpoint,
        )
        activated = await self.repository.set_active_llm_connection(connection_id)
        assert activated is not None
        return activated.to_public()

    async def disconnect_active(self) -> None:
        """Detach active LLM so research runs cleanly in zero-LLM mode."""
        await self.repository.set_active_llm_connection(None)

    async def test_connection(self, connection_id: str) -> ConnectionTestResponse:
        """Probe connection health, discover models, inspect capabilities, and persist status."""
        started = time.monotonic()
        cfg, adapter = await self.get_adapter(connection_id)
        raw_secret = cfg.api_key.get_secret_value() if cfg.api_key else None
        now = datetime.now(timezone.utc)

        try:
            ok, message, models, caps = await adapter.test_connection()
        except Exception as exc:
            ok = False
            message = redact_llm_error(exc, secrets=[raw_secret])
            models = []
            caps = cfg.effective_capabilities()

        updated_cfg = cfg.model_copy(
            update={
                "status": ConnectionStatus.CONNECTED if ok else ConnectionStatus.ERROR,
                "last_error": None if ok else redact_llm_error(message, secrets=[raw_secret]),
                "last_checked_at": now,
                "capabilities": caps if ok else cfg.capabilities,
                "updated_at": now,
            }
        )
        saved = await self.repository.save_llm_connection(updated_cfg)
        elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
        return ConnectionTestResponse(
            connection=saved.to_public(),
            ok=ok,
            message=redact_llm_error(message, secrets=[raw_secret]),
            latency_ms=elapsed_ms,
            discovered_models=models,
            capabilities=caps if ok else saved.capabilities,
        )

    async def discover_models(self, connection_id: str) -> ModelDiscoveryResponse:
        """Discover models from a connection endpoint without assuming discovery always exists."""
        cfg, adapter = await self.get_adapter(connection_id)
        models, supported, message = await adapter.list_models()
        return ModelDiscoveryResponse(
            connection_id=cfg.id,
            provider=cfg.provider,
            models=models,
            discovery_supported=supported,
            message=message,
        )

    async def select_model(
        self, connection_id: str, model: str
    ) -> LLMConnectionPublic:
        """Select a model (discovered or manual) and update inspected capabilities."""
        cfg, adapter = await self.get_adapter(connection_id)
        target = model.strip()
        if not target:
            raise ValueError("Model identifier cannot be empty.")

        updated_cfg = cfg.model_copy(
            update={
                "model": target,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        refreshed_adapter = self.registry.create(
            updated_cfg, transport=self._transport, resolver=self._resolver
        )
        caps = await refreshed_adapter.inspect_model_capabilities(target)
        updated_cfg = updated_cfg.model_copy(update={"capabilities": caps})
        saved = await self.repository.save_llm_connection(updated_cfg)
        return saved.to_public()

    async def run_test_prompt(
        self, connection_id: str, payload: TestPromptRequest
    ) -> TestPromptResponse:
        """Execute a single non-streaming verification prompt on a configured connection."""
        cfg, adapter = await self.get_adapter(connection_id)
        messages: list[ChatMessage] = []
        if payload.system_prompt:
            messages.append(ChatMessage(role="system", content=payload.system_prompt))
        messages.append(ChatMessage(role="user", content=payload.prompt))

        response = await adapter.chat(
            LLMInferenceRequest(
                messages=messages,
                model=cfg.model or None,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens or cfg.max_output_tokens,
                stream=False,
            )
        )
        return TestPromptResponse(
            connection_id=cfg.id,
            model=response.model or cfg.model,
            content=response.content,
            latency_ms=response.latency_ms,
            usage=response.usage,
        )

    async def stream_test_prompt(
        self, connection_id: str, payload: TestPromptRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream token chunks for a verification prompt on a configured connection."""
        cfg, adapter = await self.get_adapter(connection_id)
        messages: list[ChatMessage] = []
        if payload.system_prompt:
            messages.append(ChatMessage(role="system", content=payload.system_prompt))
        messages.append(ChatMessage(role="user", content=payload.prompt))

        req = LLMInferenceRequest(
            messages=messages,
            model=cfg.model or None,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens or cfg.max_output_tokens,
            stream=True,
        )
        async for chunk in adapter.stream_chat(req):
            yield chunk
