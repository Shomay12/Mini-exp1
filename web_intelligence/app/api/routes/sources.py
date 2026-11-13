"""Individual retrieved source inspection endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from web_intelligence.app.dependencies import AppContainer
from web_intelligence.app.schemas.documents import SourceRecord

router = APIRouter(prefix="/api/v1", tags=["sources"])


@router.get("/sources/{source_id}", response_model=SourceRecord)
async def get_source_by_id(source_id: str, request: Request) -> SourceRecord:
    """Retrieve complete source metadata, provenance, duplicate links, and extracted content."""
    container: AppContainer = request.app.state.container
    source = await container.repository.get_source_by_id(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source '{source_id}' not found.",
        )
    return source
