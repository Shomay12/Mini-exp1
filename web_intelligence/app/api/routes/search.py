"""Normalized search endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from web_intelligence.app.dependencies import AppContainer
from web_intelligence.app.schemas.search import SearchRequest, SearchResponse

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search_web(payload: SearchRequest, request: Request) -> SearchResponse:
    """Sanitize a search query and return normalized results across configured providers."""
    container: AppContainer = request.app.state.container
    results, disclosed, diagnostics, sanitized_q, redactions = await container.router.search(
        payload.query,
        max_results=payload.max_results,
        providers=payload.providers,
        failover_only=True,
    )
    if not sanitized_q:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query became empty after privacy sanitization.",
        )

    return SearchResponse(
        original_query=payload.query,
        sanitized_query=sanitized_q,
        redactions_applied=redactions,
        results=results,
        disclosed_queries=disclosed,
        provider_diagnostics=diagnostics,
        total_results=len(results),
    )
