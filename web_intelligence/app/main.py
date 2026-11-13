"""FastAPI application entrypoint for the Local-First Web Intelligence Engine."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from web_intelligence.app.api.routes.health import router as health_router
from web_intelligence.app.api.routes.llm import router as llm_router
from web_intelligence.app.api.routes.research import router as research_router
from web_intelligence.app.api.routes.search import router as search_router
from web_intelligence.app.api.routes.sources import router as sources_router
from web_intelligence.app.core.config import Settings, get_settings
from web_intelligence.app.core.logging import configure_logging, get_logger
from web_intelligence.app.core.security import AsyncResolver
from web_intelligence.app.dependencies import build_container

UI_DIR = Path(__file__).resolve().parent / "ui"
logger = get_logger("main")


def create_app(
    settings: Settings | None = None,
    *,
    transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
    dns_resolver: AsyncResolver | None = None,
    llm_transport: httpx.AsyncBaseTransport | None = None,
    llm_resolver: AsyncResolver | None = None,
) -> FastAPI:
    """Create and wire the FastAPI application instance."""
    runtime_settings = settings or get_settings()
    configure_logging(runtime_settings.log_level, runtime_settings.registered_secrets())
    container = build_container(
        runtime_settings,
        transport_factory=transport_factory,
        dns_resolver=dns_resolver,
        llm_transport=llm_transport,
        llm_resolver=llm_resolver,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await container.database.initialize()
        await container.llm_manager.initialize()
        purged = await container.repository.purge_expired_runs()
        if purged > 0:
            logger.info("Purged expired research runs", extra={"extra_data": {"count": purged}})
        yield

    application = FastAPI(
        title=runtime_settings.app_name,
        version=runtime_settings.app_version,
        description=(
            "Phase 1 Local-First Web Intelligence Engine: sanitized search, SSRF-safe retrieval, "
            "evidence deduplication, BM25 ranking, provenance, conflict detection, and coverage."
        ),
        lifespan=lifespan,
    )
    application.state.container = container

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled server error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error while processing request."},
        )

    application.include_router(health_router)
    application.include_router(llm_router)
    application.include_router(search_router)
    application.include_router(research_router)
    application.include_router(sources_router)

    if UI_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

        @application.get("/", include_in_schema=False)
        async def serve_ui() -> FileResponse:
            return FileResponse(UI_DIR / "index.html")

    return application


app = create_app()
