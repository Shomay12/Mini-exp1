"""Persistent ModelRegistry tracking multi-server local models, verified capabilities, and task roles."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

from web_intelligence.app.llm.security import redact_llm_error
from web_intelligence.app.schemas.generation import (
    ModelRegistryEntry,
    ModelRegistryRegisterRequest,
    ModelTaskRole,
)
from web_intelligence.app.schemas.llm import (
    CapabilityProvenance,
    CapabilitySource,
    ConnectionStatus,
    DiscoveredModel,
    LLMConnectionConfig,
    ModelCapabilities,
)

if TYPE_CHECKING:
    from web_intelligence.app.storage.repositories import ResearchRepository


def make_model_entry_id(connection_id: str, model_id: str) -> str:
    """Build deterministic registry entry ID for a (connection_id, model_id) pair."""
    return f"{connection_id.strip()}::{model_id.strip()}"


def sanitize_endpoint_for_registry(base_url: str, api_endpoint: str | None = None) -> str:
    """Return display-safe HTTP endpoint without credentials, query strings, or secrets."""
    parsed = urlparse(base_url.strip())
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = netloc.split("@", 1)[-1]
    clean_base = urlunparse((parsed.scheme, netloc, parsed.path.rstrip("/"), "", "", ""))
    if api_endpoint:
        suffix = "/" + api_endpoint.lstrip("/")
        if clean_base.endswith("/v1") and suffix.startswith("/v1/"):
            root = urlunparse((parsed.scheme, netloc, "", "", "", ""))
            return f"{root}{suffix}"
        return f"{clean_base}{suffix}"
    return clean_base


def _metadata_from_discovered(model: DiscoveredModel) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if model.family:
        meta["family"] = model.family
    if model.parameter_size:
        meta["parameter_size"] = model.parameter_size
    if model.quantization_level:
        meta["quantization_level"] = model.quantization_level
    if model.capabilities and model.capabilities.available_metadata:
        for k, v in model.capabilities.available_metadata.items():
            meta.setdefault(k, v)
    return meta


def _build_entry(
    *,
    entry_id: str,
    config: LLMConnectionConfig,
    model_id: str,
    display_name: str,
    server_endpoint: str,
    caps: ModelCapabilities,
    effective_status: ConnectionStatus,
    effective_last_conn: datetime | None,
    safe_error: str | None,
    metadata: dict[str, Any],
    is_discovered: bool,
    prev: ModelRegistryEntry | None,
    now: datetime,
) -> ModelRegistryEntry:
    return ModelRegistryEntry(
        entry_id=entry_id,
        connection_id=config.id,
        connection_name=config.name,
        provider_id=config.provider,
        preset=config.preset,
        server_endpoint=server_endpoint,
        api_endpoint=config.api_endpoint,
        model_id=model_id,
        display_name=display_name,
        context_limit=caps.max_context_tokens,
        max_output_limit=caps.recommended_output_tokens,
        supports_streaming=caps.supports_streaming,
        supports_json_mode=caps.supports_json_mode,
        supports_tool_calling=caps.supports_tool_calling,
        supports_system_prompt=bool(
            caps.available_metadata.get("supports_system_prompt", True)
        ),
        supports_stop_sequences=bool(
            caps.available_metadata.get("supports_stop_sequences", True)
        ),
        capability_provenance=caps.sources,
        connection_status=effective_status,
        last_connected_at=effective_last_conn,
        last_error=safe_error,
        model_metadata=metadata,
        is_active=config.is_active and config.model.strip() == model_id,
        is_discovered=is_discovered,
        task_roles=prev.task_roles if prev else [],
        created_at=prev.created_at if prev else now,
        updated_at=now,
    )


class ModelRegistry:
    """Tracks registered and discovered local LLM models across servers without storing API secrets."""

    def __init__(self, repository: ResearchRepository) -> None:
        self.repository = repository

    async def initialize_from_connections(self) -> list[ModelRegistryEntry]:
        """Backfill registry entries from persisted connections on startup when absent."""
        existing = await self.repository.list_model_registry_entries()
        existing_conns = {entry.connection_id for entry in existing}
        connections = await self.repository.list_llm_connections()
        synced: list[ModelRegistryEntry] = []
        for cfg in connections:
            if cfg.id not in existing_conns and cfg.model.strip():
                entries = await self.sync_connection(cfg)
                synced.extend(entries)
        return existing + synced

    async def sync_connection(
        self,
        config: LLMConnectionConfig,
        *,
        discovered_models: list[DiscoveredModel] | None = None,
        ok: bool | None = None,
        last_connected_at: datetime | None = None,
        last_error: str | None = None,
    ) -> list[ModelRegistryEntry]:
        """Synchronize configured and discovered models for a server connection into SQLite."""
        now = datetime.now(timezone.utc)
        existing_entries = {
            e.entry_id: e
            for e in await self.repository.list_model_registry_entries(config.id)
        }
        server_endpoint = sanitize_endpoint_for_registry(config.base_url, config.api_endpoint)
        raw_secret = config.api_key.get_secret_value() if config.api_key else None
        safe_error = (
            redact_llm_error(last_error, secrets=[raw_secret])
            if last_error
            else (
                redact_llm_error(config.last_error, secrets=[raw_secret])
                if config.last_error
                else None
            )
        )
        effective_status = config.status
        if ok is True:
            effective_status = ConnectionStatus.CONNECTED
            safe_error = None
        elif ok is False:
            effective_status = ConnectionStatus.ERROR

        saved: list[ModelRegistryEntry] = []
        seen_ids: set[str] = set()

        # Register/update each discovered model when model discovery returned entries
        configured_model = config.model.strip()
        for dm in discovered_models or []:
            model_id = dm.id.strip()
            if not model_id:
                continue
            entry_id = make_model_entry_id(config.id, model_id)
            seen_ids.add(entry_id)
            prev = existing_entries.get(entry_id)
            caps: ModelCapabilities = dm.capabilities or config.effective_capabilities()
            if (
                model_id == configured_model
                and config.capabilities is not None
                and config.capabilities.sources.context_length == CapabilitySource.VERIFIED
            ):
                caps = config.capabilities
            prev_last_conn = prev.last_connected_at if prev else None
            effective_last_conn = last_connected_at or (
                now if effective_status == ConnectionStatus.CONNECTED else prev_last_conn
            )
            display_name = (
                dm.name.strip()
                if dm.name and dm.name.strip() != model_id
                else f"{config.name} · {model_id}"
            )
            metadata = _metadata_from_discovered(dm)
            if caps.available_metadata:
                for k, v in caps.available_metadata.items():
                    metadata.setdefault(k, v)
            entry = _build_entry(
                entry_id=entry_id,
                config=config,
                model_id=model_id,
                display_name=display_name,
                server_endpoint=server_endpoint,
                caps=caps,
                effective_status=effective_status,
                effective_last_conn=effective_last_conn,
                safe_error=safe_error,
                metadata=metadata,
                is_discovered=True,
                prev=prev,
                now=now,
            )
            saved.append(await self.repository.save_model_registry_entry(entry))

        # Ensure configured default model on connection is always present when set
        if configured_model:
            entry_id = make_model_entry_id(config.id, configured_model)
            if entry_id not in seen_ids:
                seen_ids.add(entry_id)
                prev = existing_entries.get(entry_id)
                caps = config.effective_capabilities()
                prev_last_conn = prev.last_connected_at if prev else None
                effective_last_conn = last_connected_at or (
                    now if effective_status == ConnectionStatus.CONNECTED else prev_last_conn
                )
                entry = _build_entry(
                    entry_id=entry_id,
                    config=config,
                    model_id=configured_model,
                    display_name=f"{config.name} · {configured_model}",
                    server_endpoint=server_endpoint,
                    caps=caps,
                    effective_status=effective_status,
                    effective_last_conn=effective_last_conn,
                    safe_error=safe_error,
                    metadata=caps.available_metadata or (prev.model_metadata if prev else {}),
                    is_discovered=prev.is_discovered if prev else False,
                    prev=prev,
                    now=now,
                )
                saved.append(await self.repository.save_model_registry_entry(entry))

        # Keep existing manual entries updated on connection status/active flags without deleting them
        for eid, prev in existing_entries.items():
            if eid in seen_ids:
                continue
            updated = prev.model_copy(
                update={
                    "connection_name": config.name,
                    "provider_id": config.provider,
                    "preset": config.preset,
                    "server_endpoint": server_endpoint,
                    "api_endpoint": config.api_endpoint,
                    "connection_status": effective_status,
                    "last_connected_at": last_connected_at or prev.last_connected_at,
                    "last_error": safe_error,
                    "is_active": config.is_active and configured_model == prev.model_id,
                    "updated_at": now,
                }
            )
            saved.append(await self.repository.save_model_registry_entry(updated))

        return saved

    async def refresh_active_flags(self) -> None:
        """Synchronize is_active flags across all registry entries from current connections."""
        connections = {cfg.id: cfg for cfg in await self.repository.list_llm_connections()}
        entries = await self.repository.list_model_registry_entries()
        now = datetime.now(timezone.utc)
        for entry in entries:
            cfg = connections.get(entry.connection_id)
            next_active = bool(
                cfg is not None
                and cfg.is_active
                and cfg.model.strip() == entry.model_id
            )
            if entry.is_active != next_active:
                await self.repository.save_model_registry_entry(
                    entry.model_copy(update={"is_active": next_active, "updated_at": now})
                )

    async def register_manual_model(
        self,
        config: LLMConnectionConfig,
        payload: ModelRegistryRegisterRequest,
    ) -> ModelRegistryEntry:
        """Manually register a model against a connection when discovery is unavailable."""
        now = datetime.now(timezone.utc)
        model_id = payload.model_id.strip()
        entry_id = make_model_entry_id(config.id, model_id)
        existing = await self.repository.get_model_registry_entry(entry_id)

        ctx_limit = payload.context_limit or config.context_length
        out_limit = min(payload.max_output_limit or config.max_output_tokens, max(64, ctx_limit - 1))
        provenance = CapabilityProvenance(
            context_length=CapabilitySource.CONFIGURED,
            max_output_tokens=CapabilitySource.CONFIGURED,
            streaming=CapabilitySource.CONFIGURED,
            tool_calling=CapabilitySource.UNKNOWN,
            json_mode=CapabilitySource.UNKNOWN,
            vision=CapabilitySource.UNKNOWN,
        )
        entry = ModelRegistryEntry(
            entry_id=entry_id,
            connection_id=config.id,
            connection_name=config.name,
            provider_id=config.provider,
            preset=config.preset,
            server_endpoint=sanitize_endpoint_for_registry(config.base_url, config.api_endpoint),
            api_endpoint=config.api_endpoint,
            model_id=model_id,
            display_name=(payload.display_name or f"{config.name} · {model_id}").strip(),
            context_limit=ctx_limit,
            max_output_limit=out_limit,
            supports_streaming=config.streaming,
            supports_json_mode=False,
            supports_tool_calling=False,
            supports_system_prompt=True,
            supports_stop_sequences=True,
            capability_provenance=provenance,
            connection_status=config.status,
            last_connected_at=config.last_checked_at if config.status == ConnectionStatus.CONNECTED else None,
            last_error=None,
            model_metadata=existing.model_metadata if existing else {},
            is_active=config.is_active and config.model.strip() == model_id,
            is_discovered=False,
            task_roles=payload.task_roles or (existing.task_roles if existing else []),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        return await self.repository.save_model_registry_entry(entry)

    async def assign_task_roles(
        self, entry_id: str, task_roles: list[ModelTaskRole]
    ) -> ModelRegistryEntry | None:
        """Assign per-task routing roles (synthesis, outline, etc.) to a registered model."""
        existing = await self.repository.get_model_registry_entry(entry_id)
        if existing is None:
            return None
        deduped = list(dict.fromkeys(task_roles))
        updated = existing.model_copy(
            update={
                "task_roles": deduped,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        return await self.repository.save_model_registry_entry(updated)

    async def list_models(
        self,
        *,
        connection_id: str | None = None,
        task_role: ModelTaskRole | None = None,
    ) -> list[ModelRegistryEntry]:
        """Return registered models ordered active-first, optionally filtered."""
        entries = await self.repository.list_model_registry_entries(connection_id=connection_id)
        if task_role is not None:
            entries = [e for e in entries if task_role in e.task_roles]
        entries.sort(
            key=lambda e: (
                not e.is_active,
                e.connection_status != ConnectionStatus.CONNECTED,
                e.connection_name.lower(),
                e.model_id.lower(),
            )
        )
        return entries

    async def get_model(self, entry_id: str) -> ModelRegistryEntry | None:
        """Load one ModelRegistryEntry by ID."""
        return await self.repository.get_model_registry_entry(entry_id)

    async def resolve_target(
        self,
        *,
        connection_id: str | None = None,
        model_id: str | None = None,
        task_role: ModelTaskRole | None = None,
    ) -> ModelRegistryEntry | None:
        """Resolve best registered model target honoring explicit override, task role, or active model."""
        entries = await self.list_models()
        if connection_id and model_id:
            target = next(
                (
                    e
                    for e in entries
                    if e.connection_id == connection_id and e.model_id == model_id
                ),
                None,
            )
            if target is not None:
                return target

        if connection_id:
            conn_entries = [e for e in entries if e.connection_id == connection_id]
            if task_role is not None:
                role_match = next((e for e in conn_entries if task_role in e.task_roles), None)
                if role_match is not None:
                    return role_match
            if conn_entries:
                return conn_entries[0]

        if model_id:
            by_model = [e for e in entries if e.model_id == model_id]
            if by_model:
                return by_model[0]

        if task_role is not None:
            by_role = [e for e in entries if task_role in e.task_roles]
            if by_role:
                return by_role[0]

        return next((e for e in entries if e.is_active), entries[0] if entries else None)

    async def remove_connection_models(self, connection_id: str) -> int:
        """Delete all registered models belonging to a deleted connection."""
        return await self.repository.delete_model_registry_entries_for_connection(connection_id)
