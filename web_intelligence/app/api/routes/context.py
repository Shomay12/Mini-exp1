"""FastAPI routes for Stage 5 Adaptive Context Management Engine."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from web_intelligence.app.context.engine import (
    ResearchRunNotFoundError,
    WorkspaceNoteNotFoundError,
)
from web_intelligence.app.context.models import (
    ContextBudgetResponse,
    ContextBudgetUpdateRequest,
    ContextCapabilitiesResponse,
    ContextCompressRequest,
    ContextCompressResponse,
    ContextConfigurationError,
    ContextLimitExceededError,
    ContextPreviewRequest,
    ContextPreviewResponse,
    ContextValidateRequest,
    ContextValidationReport,
    ConversationTurn,
    WorkspaceNote,
    WorkspaceNoteCreate,
    WorkspaceNoteUpdate,
)
from web_intelligence.app.dependencies import AppContainer
from web_intelligence.app.llm.manager import ConnectionNotFoundError
from web_intelligence.app.llm.security import redact_llm_error

router = APIRouter(prefix="/api/v1/context", tags=["context"])


class ConversationTurnCreate(BaseModel):
    """Payload to append a turn to research session conversation history."""

    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    scope: str = "session"
    decisions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


def _raise_http_for_context_error(exc: Exception) -> None:
    detail = redact_llm_error(exc)
    if isinstance(
        exc,
        (
            ResearchRunNotFoundError,
            WorkspaceNoteNotFoundError,
            ConnectionNotFoundError,
        ),
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail) from exc
    if isinstance(exc, (ContextLimitExceededError, ContextConfigurationError, ValueError)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected error processing context request.",
    ) from exc


@router.get("/capabilities", response_model=ContextCapabilitiesResponse)
async def get_context_capabilities(
    request: Request,
    connection_id: str | None = Query(default=None),
    refresh: bool = Query(default=False),
) -> ContextCapabilitiesResponse:
    """Inspect active or specified local model context capabilities and estimator precision."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    try:
        return await container.context_engine.get_capabilities_response(
            connection_id=connection_id,
            refresh_live=refresh,
        )
    except Exception as exc:
        _raise_http_for_context_error(exc)


@router.get("/budget", response_model=ContextBudgetResponse)
async def get_context_budget(
    request: Request,
    connection_id: str | None = Query(default=None),
) -> ContextBudgetResponse:
    """Retrieve active context budget configuration and dynamic 7-bucket token allocation."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    try:
        return await container.context_engine.get_budget_response(
            connection_id=connection_id
        )
    except Exception as exc:
        _raise_http_for_context_error(exc)


@router.put("/budget", response_model=ContextBudgetResponse)
async def update_context_budget(
    payload: ContextBudgetUpdateRequest,
    request: Request,
    connection_id: str | None = Query(default=None),
) -> ContextBudgetResponse:
    """Validate and persist updated context budget configuration within model limits."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    try:
        return await container.context_engine.update_budget(
            payload, connection_id=connection_id
        )
    except Exception as exc:
        _raise_http_for_context_error(exc)


@router.post("/preview", response_model=ContextPreviewResponse)
async def preview_context(
    payload: ContextPreviewRequest,
    request: Request,
) -> ContextPreviewResponse:
    """Assemble, compress, convert, and pre-validate a context package before inference."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    try:
        return await container.context_engine.preview_context(payload)
    except Exception as exc:
        _raise_http_for_context_error(exc)


@router.post("/validate", response_model=ContextValidationReport)
async def validate_context(
    payload: ContextValidateRequest,
    request: Request,
) -> ContextValidationReport:
    """Run pre-inference token, budget, provenance, and section checks without silent truncation."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    try:
        return await container.context_engine.validate_context(payload)
    except Exception as exc:
        _raise_http_for_context_error(exc)


@router.post("/compress", response_model=ContextCompressResponse)
async def compress_context_evidence(
    payload: ContextCompressRequest,
    request: Request,
) -> ContextCompressResponse:
    """Apply configured compression strategy while preserving original inspectable evidence."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    try:
        return await container.context_engine.compress_evidence(payload)
    except Exception as exc:
        _raise_http_for_context_error(exc)


@router.get("/notes", response_model=list[WorkspaceNote])
async def list_workspace_notes(
    request: Request,
    workspace_id: str = Query(default="default"),
) -> list[WorkspaceNote]:
    """List user-controlled persistent workspace research notes."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    return await container.context_engine.list_workspace_notes(workspace_id)


@router.post("/notes", response_model=WorkspaceNote, status_code=status.HTTP_201_CREATED)
async def create_workspace_note(
    payload: WorkspaceNoteCreate,
    request: Request,
) -> WorkspaceNote:
    """Create a persistent user-controlled workspace research note."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    return await container.context_engine.create_workspace_note(payload)


@router.patch("/notes/{note_id}", response_model=WorkspaceNote)
async def update_workspace_note(
    note_id: str,
    payload: WorkspaceNoteUpdate,
    request: Request,
) -> WorkspaceNote:
    """Update content or enabled toggle on a workspace research note."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    try:
        return await container.context_engine.update_workspace_note(note_id, payload)
    except Exception as exc:
        _raise_http_for_context_error(exc)


@router.delete("/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace_note(note_id: str, request: Request) -> None:
    """Delete a persistent workspace research note."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    deleted = await container.context_engine.delete_workspace_note(note_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace note '{note_id}' not found.",
        )


@router.post("/history/{research_id}", response_model=ConversationTurn, status_code=status.HTTP_201_CREATED)
async def append_conversation_turn(
    research_id: str,
    payload: ConversationTurnCreate,
    request: Request,
) -> ConversationTurn:
    """Append a conversation turn to a research session."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    try:
        return await container.context_engine.add_conversation_turn(
            research_id=research_id,
            role=payload.role,
            content=payload.content,
            scope=payload.scope,
            decisions=payload.decisions,
            unresolved_questions=payload.unresolved_questions,
        )
    except Exception as exc:
        _raise_http_for_context_error(exc)


@router.get("/{research_id}", response_model=ContextPreviewResponse)
async def get_research_context(
    research_id: str,
    request: Request,
    connection_id: str | None = Query(default=None),
) -> ContextPreviewResponse:
    """Inspect assembled ContextPackage, validation report, and messages for a research run."""
    container: AppContainer = request.app.state.container
    assert container.context_engine is not None
    try:
        return await container.context_engine.preview_context(
            ContextPreviewRequest(
                research_id=research_id,
                connection_id=connection_id,
            )
        )
    except Exception as exc:
        _raise_http_for_context_error(exc)
