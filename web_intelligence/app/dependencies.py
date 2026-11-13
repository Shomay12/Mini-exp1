"""Application service container wiring settings, security, retrieval, storage, and orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from web_intelligence.app.core.config import Settings
from web_intelligence.app.core.security import AsyncResolver
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.providers.fixtures import (
    build_fixture_transport,
    fixture_dns_resolver,
)
from web_intelligence.app.providers.router import SearchProviderRouter
from web_intelligence.app.research.orchestrator import ResearchOrchestrator
from web_intelligence.app.storage.database import DatabaseManager
from web_intelligence.app.storage.repositories import ResearchRepository


@dataclass
class AppContainer:
    """Dependency container attached to FastAPI app state."""

    settings: Settings
    database: DatabaseManager
    repository: ResearchRepository
    gateway: PrivacyGateway
    router: SearchProviderRouter
    orchestrator: ResearchOrchestrator


def build_container(
    settings: Settings,
    *,
    transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
    dns_resolver: AsyncResolver | None = None,
) -> AppContainer:
    """Construct wired application dependencies.

    When configured exclusively with `mock` provider and no explicit transport/resolver
    override, uses the controlled offline fixture transport and public-IP fixture DNS
    resolver so offline local evaluation executes end-to-end safely.
    """
    is_pure_mock_mode = (
        settings.search_providers == ["mock"]
        and settings.enable_mock_provider
        and transport_factory is None
        and dns_resolver is None
    )

    effective_transport = (
        build_fixture_transport if is_pure_mock_mode else transport_factory
    )
    effective_resolver = (
        fixture_dns_resolver if is_pure_mock_mode else dns_resolver
    )

    database = DatabaseManager(settings.database_path)
    repository = ResearchRepository(database, retention_days=settings.retention_days)
    gateway = PrivacyGateway(settings, transport_factory=effective_transport)
    router = SearchProviderRouter(settings, gateway)
    orchestrator = ResearchOrchestrator(
        settings,
        gateway,
        repository,
        router=router,
        dns_resolver=effective_resolver,
    )
    return AppContainer(
        settings=settings,
        database=database,
        repository=repository,
        gateway=gateway,
        router=router,
        orchestrator=orchestrator,
    )
