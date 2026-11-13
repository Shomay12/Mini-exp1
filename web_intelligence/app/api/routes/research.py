"""Research workflow execution, retrieval, listing, and deletion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from web_intelligence.app.dependencies import AppContainer
from web_intelligence.app.schemas.research import (
    ResearchPackage,
    ResearchRequest,
    ResearchRunSummary,
)

router = APIRouter(prefix="/api/v1", tags=["research"])


@router.post("/research", response_model=ResearchPackage)
async def execute_research(payload: ResearchRequest, request: Request) -> ResearchPackage:
    """Execute the full end-to-end Stage 3 + Stage 4 web intelligence workflow."""
    container: AppContainer = request.app.state.container
    package = await container.orchestrator.run_research(payload)
    return package


@router.get("/research", response_model=list[ResearchRunSummary])
async def list_research_runs(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[ResearchRunSummary]:
    """List locally persisted research runs."""
    container: AppContainer = request.app.state.container
    return await container.repository.list_research_runs(limit=limit)


@router.get("/research/{research_id}", response_model=ResearchPackage)
async def get_research_by_id(research_id: str, request: Request) -> ResearchPackage:
    """Retrieve full status and structured evidence package for a research run."""
    container: AppContainer = request.app.state.container
    package = await container.repository.get_research_package(research_id)
    if package is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research run '{research_id}' not found.",
        )
    return package


@router.delete("/research/{research_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_research_by_id(research_id: str, request: Request) -> None:
    """Delete a local research run and all cascaded sources, evidence, and conflicts."""
    container: AppContainer = request.app.state.container
    deleted = await container.repository.delete_research_run(research_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research run '{research_id}' not found.",
        )
