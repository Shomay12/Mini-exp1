"""Application service container wiring settings, security, retrieval, storage, and orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from web_intelligence.app.agent.orchestrator import AgentResearchOrchestrator
from web_intelligence.app.context.engine import ContextManagementEngine
from web_intelligence.app.core.config import Settings
from web_intelligence.app.core.security import AsyncResolver
from web_intelligence.app.generation.answer_generator import AnswerGenerator
from web_intelligence.app.generation.citations import CitationValidator
from web_intelligence.app.generation.executor import InferenceExecutor
from web_intelligence.app.generation.long_form import LongFormReportGenerator
from web_intelligence.app.generation.prompt_compiler import PromptCompiler
from web_intelligence.app.llm.manager import LLMConnectionManager
from web_intelligence.app.llm.model_registry import ModelRegistry
from web_intelligence.app.llm.registry import AdapterRegistry
from web_intelligence.app.privacy.gateway import PrivacyGateway
from web_intelligence.app.providers.fixtures import (
    build_fixture_transport,
    fixture_dns_resolver,
)
from web_intelligence.app.providers.router import SearchProviderRouter
from web_intelligence.app.research.orchestrator import ResearchOrchestrator
from web_intelligence.app.retrieval.search_engine import WebRetrievalEngine
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
    llm_registry: AdapterRegistry
    llm_manager: LLMConnectionManager
    orchestrator: ResearchOrchestrator
    context_engine: ContextManagementEngine | None = None
    model_registry: ModelRegistry | None = None
    inference_executor: InferenceExecutor | None = None
    answer_generator: AnswerGenerator | None = None
    agent_orchestrator: AgentResearchOrchestrator | None = None


def build_container(
    settings: Settings,
    *,
    transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
    dns_resolver: AsyncResolver | None = None,
    llm_transport: httpx.AsyncBaseTransport | None = None,
    llm_resolver: AsyncResolver | None = None,
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
    llm_registry = AdapterRegistry()
    model_registry = ModelRegistry(repository)
    llm_manager = LLMConnectionManager(
        repository,
        registry=llm_registry,
        model_registry=model_registry,
        transport=llm_transport,
        resolver=llm_resolver,
    )
    orchestrator = ResearchOrchestrator(
        settings,
        gateway,
        repository,
        router=router,
        dns_resolver=effective_resolver,
        llm_manager=llm_manager,
    )
    context_engine = ContextManagementEngine(
        repository=repository,
        llm_manager=llm_manager,
    )
    inference_executor = InferenceExecutor(llm_manager, model_registry)
    answer_generator = AnswerGenerator(
        repository=repository,
        context_engine=context_engine,
        executor=inference_executor,
    )
    prompt_compiler = PromptCompiler()
    citation_validator = CitationValidator()
    long_form_generator = LongFormReportGenerator(
        executor=inference_executor,
        prompt_compiler=prompt_compiler,
        citation_validator=citation_validator,
    )
    agent_retrieval = WebRetrievalEngine(
        settings=settings,
        gateway=gateway,
        router=router,
        dns_resolver=effective_resolver,
    )
    agent_orchestrator = AgentResearchOrchestrator(
        repository=repository,
        gateway=gateway,
        router=router,
        retrieval=agent_retrieval,
        context_engine=context_engine,
        llm_manager=llm_manager,
        inference_executor=inference_executor,
        long_form_generator=long_form_generator,
        prompt_compiler=prompt_compiler,
        citation_validator=citation_validator,
    )
    return AppContainer(
        settings=settings,
        database=database,
        repository=repository,
        gateway=gateway,
        router=router,
        llm_registry=llm_registry,
        llm_manager=llm_manager,
        orchestrator=orchestrator,
        context_engine=context_engine,
        model_registry=model_registry,
        inference_executor=inference_executor,
        answer_generator=answer_generator,
        agent_orchestrator=agent_orchestrator,
    )
