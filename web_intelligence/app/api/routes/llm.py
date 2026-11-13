"""FastAPI routes for universal local/self-hosted LLM connection management and test inference."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from web_intelligence.app.dependencies import AppContainer
from web_intelligence.app.llm.base import (
    LLMAuthError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
)
from web_intelligence.app.llm.manager import (
    ConnectionNotFoundError,
    NoActiveConnectionError,
)
from web_intelligence.app.llm.registry import UnsupportedProviderError
from web_intelligence.app.llm.security import (
    LLMConnectionSecurityError,
    LLMURLValidationError,
    redact_llm_error,
)
from web_intelligence.app.schemas.llm import (
    ConnectionTestResponse,
    LLMConnectionCreate,
    LLMConnectionPublic,
    LLMConnectionUpdate,
    ModelDiscoveryResponse,
    ProviderPreset,
    TestPromptRequest,
    TestPromptResponse,
)

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])


class ModelSelectRequest(BaseModel):
    """Payload for selecting a discovered or manually entered model ID."""

    model: str = Field(..., min_length=1, max_length=200)


def _raise_http_for_llm_error(exc: Exception) -> None:
    detail = redact_llm_error(exc)
    if isinstance(exc, ConnectionNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail) from exc
    if isinstance(exc, NoActiveConnectionError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail) from exc
    if isinstance(exc, LLMConnectionSecurityError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail) from exc
    if isinstance(exc, (LLMURLValidationError, UnsupportedProviderError, ValueError)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail
        ) from exc
    if isinstance(exc, LLMTimeoutError):
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=detail) from exc
    if isinstance(exc, (LLMAuthError, LLMResponseError, LLMError)):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from exc
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected error handling LLM request.",
    ) from exc


@router.get("/presets", response_model=list[ProviderPreset])
async def list_llm_presets(request: Request) -> list[ProviderPreset]:
    """Return built-in local/self-hosted server presets."""
    container: AppContainer = request.app.state.container
    return container.llm_registry.presets()


@router.get("/connections", response_model=list[LLMConnectionPublic])
async def list_llm_connections(request: Request) -> list[LLMConnectionPublic]:
    """List saved LLM endpoint configurations with API keys excluded."""
    container: AppContainer = request.app.state.container
    return await container.llm_manager.list_connections()


@router.post(
    "/connections",
    response_model=LLMConnectionPublic,
    status_code=status.HTTP_201_CREATED,
)
async def create_llm_connection(
    payload: LLMConnectionCreate, request: Request
) -> LLMConnectionPublic:
    """Validate and save a new local or explicitly approved remote LLM connection."""
    container: AppContainer = request.app.state.container
    try:
        return await container.llm_manager.create_connection(payload)
    except Exception as exc:
        _raise_http_for_llm_error(exc)


@router.get("/connections/{connection_id}", response_model=LLMConnectionPublic)
async def get_llm_connection(
    connection_id: str, request: Request
) -> LLMConnectionPublic:
    """Fetch a single LLM connection configuration without secrets."""
    container: AppContainer = request.app.state.container
    try:
        return await container.llm_manager.get_connection(connection_id)
    except Exception as exc:
        _raise_http_for_llm_error(exc)


@router.patch("/connections/{connection_id}", response_model=LLMConnectionPublic)
async def update_llm_connection(
    connection_id: str, payload: LLMConnectionUpdate, request: Request
) -> LLMConnectionPublic:
    """Update fields on an existing LLM connection configuration."""
    container: AppContainer = request.app.state.container
    try:
        return await container.llm_manager.update_connection(connection_id, payload)
    except Exception as exc:
        _raise_http_for_llm_error(exc)


@router.delete(
    "/connections/{connection_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_llm_connection(connection_id: str, request: Request) -> None:
    """Delete a saved LLM connection configuration."""
    container: AppContainer = request.app.state.container
    try:
        await container.llm_manager.delete_connection(connection_id)
    except Exception as exc:
        _raise_http_for_llm_error(exc)


@router.post(
    "/connections/{connection_id}/activate", response_model=LLMConnectionPublic
)
async def activate_llm_connection(
    connection_id: str, request: Request
) -> LLMConnectionPublic:
    """Attach a saved connection as the active research inference model."""
    container: AppContainer = request.app.state.container
    try:
        return await container.llm_manager.activate_connection(connection_id)
    except Exception as exc:
        _raise_http_for_llm_error(exc)


@router.post("/disconnect")
async def disconnect_active_llm(request: Request) -> dict[str, object]:
    """Detach active LLM so the engine operates cleanly in zero-LLM mode."""
    container: AppContainer = request.app.state.container
    await container.llm_manager.disconnect_active()
    return {"status": "disconnected", "active_llm": None}


@router.post(
    "/connections/{connection_id}/test", response_model=ConnectionTestResponse
)
async def test_llm_connection(
    connection_id: str, request: Request
) -> ConnectionTestResponse:
    """Probe endpoint connectivity, discover models, and inspect capabilities."""
    container: AppContainer = request.app.state.container
    try:
        return await container.llm_manager.test_connection(connection_id)
    except Exception as exc:
        _raise_http_for_llm_error(exc)


@router.get(
    "/connections/{connection_id}/models", response_model=ModelDiscoveryResponse
)
async def discover_llm_models(
    connection_id: str, request: Request
) -> ModelDiscoveryResponse:
    """Discover available models from the connected inference server."""
    container: AppContainer = request.app.state.container
    try:
        return await container.llm_manager.discover_models(connection_id)
    except Exception as exc:
        _raise_http_for_llm_error(exc)


@router.post(
    "/connections/{connection_id}/model", response_model=LLMConnectionPublic
)
async def select_llm_model(
    connection_id: str, payload: ModelSelectRequest, request: Request
) -> LLMConnectionPublic:
    """Select a discovered or manual model identifier and refresh inspected capabilities."""
    container: AppContainer = request.app.state.container
    try:
        return await container.llm_manager.select_model(connection_id, payload.model)
    except Exception as exc:
        _raise_http_for_llm_error(exc)


@router.post("/connections/{connection_id}/prompt", response_model=None)
async def run_llm_prompt(
    connection_id: str, payload: TestPromptRequest, request: Request
) -> TestPromptResponse | StreamingResponse:
    """Execute a verification prompt (JSON or SSE stream) against the configured server."""
    container: AppContainer = request.app.state.container
    if not payload.stream:
        try:
            return await container.llm_manager.run_test_prompt(connection_id, payload)
        except Exception as exc:
            _raise_http_for_llm_error(exc)

    # Validate connection exists and passes security policy before starting SSE stream
    try:
        cfg, _ = await container.llm_manager.get_adapter(connection_id)
    except Exception as exc:
        _raise_http_for_llm_error(exc)

    raw_secret = cfg.api_key.get_secret_value() if cfg.api_key else None

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for chunk in container.llm_manager.stream_test_prompt(
                connection_id, payload
            ):
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            err_msg = redact_llm_error(exc, secrets=[raw_secret])
            yield f"event: error\ndata: {json.dumps({'detail': err_msg})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
