"""FastAPI routes for Stage 6 Universal Local LLM Generation and ModelRegistry management."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from web_intelligence.app.context.engine import ResearchRunNotFoundError
from web_intelligence.app.dependencies import AppContainer
from web_intelligence.app.generation.executor import (
    ContextBudgetBlockedError,
    GenerationError,
    ModelNotConfiguredError,
)
from web_intelligence.app.llm.manager import (
    ConnectionNotFoundError,
    NoActiveConnectionError,
)
from web_intelligence.app.llm.security import (
    LLMConnectionSecurityError,
    LLMURLValidationError,
    redact_llm_error,
)
from web_intelligence.app.schemas.generation import (
    GenerationRequest,
    GenerationResult,
    GenerationStreamRequest,
    ModelRegistryEntry,
    ModelRegistryRegisterRequest,
    ModelRegistryRoleUpdate,
    ModelTaskRole,
)
from web_intelligence.app.schemas.llm import ConnectionStatus

router = APIRouter(prefix="/api/v1/generation", tags=["generation"])
registry_router = APIRouter(prefix="/api/v1/llm/registry", tags=["llm", "models"])


def _raise_http_for_generation_error(exc: Exception) -> None:
    detail = redact_llm_error(exc)
    if isinstance(exc, (ResearchRunNotFoundError, ConnectionNotFoundError)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail) from exc
    if isinstance(exc, (ModelNotConfiguredError, NoActiveConnectionError)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail) from exc
    if isinstance(exc, LLMConnectionSecurityError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail) from exc
    if isinstance(exc, (ContextBudgetBlockedError, LLMURLValidationError, ValueError)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail
        ) from exc
    if isinstance(exc, GenerationError):
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected error processing local LLM generation request.",
    ) from exc


def _sanitize_sse_payload(event_dict: dict[str, Any]) -> tuple[str, str]:
    event_name = str(event_dict.get("event") or "message")
    clean = {k: v for k, v in event_dict.items() if k != "result_obj"}
    return event_name, json.dumps(clean, default=str)


@router.post("", response_model=GenerationResult)
async def create_generation(
    payload: GenerationRequest, request: Request
) -> GenerationResult:
    """Execute complete grounded answer/report generation using the selected local LLM."""
    container: AppContainer = request.app.state.container
    assert container.answer_generator is not None
    try:
        return await container.answer_generator.generate(payload)
    except Exception as exc:
        _raise_http_for_generation_error(exc)


@router.post("/stream")
async def stream_generation(
    payload: GenerationStreamRequest, request: Request
) -> StreamingResponse:
    """Stream real-time local LLM generation progress, token deltas, citations, and completion."""
    container: AppContainer = request.app.state.container
    assert container.answer_generator is not None

    gen_iter = container.answer_generator.stream_generate(
        payload,
        is_disconnected=request.is_disconnected,
    )
    try:
        first_event = await anext(gen_iter)
    except StopAsyncIteration as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Generation stream ended unexpectedly before starting.",
        ) from exc
    except Exception as exc:
        _raise_http_for_generation_error(exc)

    async def event_stream() -> AsyncIterator[str]:
        try:
            ev_name, data_json = _sanitize_sse_payload(first_event)
            yield f"event: {ev_name}\ndata: {data_json}\n\n"
            async for event_dict in gen_iter:
                ev_name, data_json = _sanitize_sse_payload(event_dict)
                yield f"event: {ev_name}\ndata: {data_json}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            err_detail = redact_llm_error(exc)
            err_payload = json.dumps(
                {"event": "error", "detail": err_detail, "error_code": "stream_error"}
            )
            yield f"event: error\ndata: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("", response_model=list[GenerationResult])
async def list_generations(
    request: Request,
    research_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[GenerationResult]:
    """List persisted local LLM generation runs ordered newest first."""
    container: AppContainer = request.app.state.container
    assert container.answer_generator is not None
    return await container.answer_generator.list_generations(
        research_id=research_id, limit=limit
    )


@router.get("/{generation_id}", response_model=GenerationResult)
async def get_generation(generation_id: str, request: Request) -> GenerationResult:
    """Retrieve a single persisted generation run (including partial cancelled output)."""
    container: AppContainer = request.app.state.container
    assert container.answer_generator is not None
    result = await container.answer_generator.get_generation(generation_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Generation run '{generation_id}' was not found.",
        )
    return result


@router.post("/{generation_id}/cancel", response_model=GenerationResult)
async def cancel_generation(generation_id: str, request: Request) -> GenerationResult:
    """Request cooperative cancellation of an active generation run."""
    container: AppContainer = request.app.state.container
    assert container.answer_generator is not None
    result = await container.answer_generator.cancel_generation(generation_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Generation run '{generation_id}' was not found.",
        )
    return result


@registry_router.get("", response_model=list[ModelRegistryEntry])
async def list_registered_models(
    request: Request,
    connection_id: str | None = Query(default=None),
    only_connected: bool = Query(default=False),
    task_role: ModelTaskRole | None = Query(default=None),
) -> list[ModelRegistryEntry]:
    """List multi-server models registered locally with verified capabilities (no secrets)."""
    container: AppContainer = request.app.state.container
    assert container.model_registry is not None
    entries = await container.model_registry.list_models(
        connection_id=connection_id,
        task_role=task_role,
    )
    if only_connected:
        entries = [e for e in entries if e.connection_status == ConnectionStatus.CONNECTED]
    return entries


@registry_router.post(
    "",
    response_model=ModelRegistryEntry,
    status_code=status.HTTP_201_CREATED,
)
async def register_manual_model(
    payload: ModelRegistryRegisterRequest, request: Request
) -> ModelRegistryEntry:
    """Manually register a model ID against a configured connection when discovery is unavailable."""
    container: AppContainer = request.app.state.container
    assert container.model_registry is not None
    cfg = await container.repository.get_llm_connection(payload.connection_id)
    if cfg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LLM connection '{payload.connection_id}' was not found.",
        )
    try:
        return await container.model_registry.register_manual_model(cfg, payload)
    except Exception as exc:
        _raise_http_for_generation_error(exc)


@registry_router.patch("/{entry_id:path}/roles", response_model=ModelRegistryEntry)
async def update_model_task_roles(
    entry_id: str, payload: ModelRegistryRoleUpdate, request: Request
) -> ModelRegistryEntry:
    """Assign task routing roles (synthesis, long_form_outline, etc.) to a registered model."""
    container: AppContainer = request.app.state.container
    assert container.model_registry is not None
    updated = await container.model_registry.assign_task_roles(entry_id, payload.task_roles)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Registered model entry '{entry_id}' was not found.",
        )
    return updated
