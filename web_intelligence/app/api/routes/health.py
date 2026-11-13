"""Health and configuration endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from web_intelligence.app.dependencies import AppContainer
from web_intelligence.app.schemas.research import HealthResponse, SafeConfigResponse

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def get_health(request: Request) -> HealthResponse:
    """Return service health, SQLite readiness, and search provider availability."""
    container: AppContainer = request.app.state.container
    db_ready = False
    try:
        conn = await container.database.connect()
        try:
            await conn.execute("SELECT 1")
            db_ready = True
        finally:
            await conn.close()
    except Exception:
        db_ready = False

    providers = container.router.list_statuses()
    return HealthResponse(
        status="ok" if db_ready else "degraded",
        database_ready=db_ready,
        cloud_llm_required=False,
        providers=providers,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/config", response_model=SafeConfigResponse)
async def get_safe_config(request: Request) -> SafeConfigResponse:
    """Return safe configuration metadata with all secrets masked/excluded."""
    container: AppContainer = request.app.state.container
    return SafeConfigResponse(config=container.settings.to_safe_dict())
