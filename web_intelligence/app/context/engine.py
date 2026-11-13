"""High-level Adaptive Context Management Engine orchestrating capabilities, budgets, selection, compression, history, building, and validation."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from web_intelligence.app.context.budget_manager import DynamicContextBudgetManager
from web_intelligence.app.context.context_builder import (
    ContextAdapterConverter,
    StructuredContextBuilder,
)
from web_intelligence.app.context.context_compressor import ContextCompressor
from web_intelligence.app.context.context_validator import ContextValidator
from web_intelligence.app.context.history_manager import ConversationHistoryManager
from web_intelligence.app.context.models import (
    AllocationProfile,
    BudgetAllocation,
    CompressionStrategy,
    ContextBudgetConfig,
    ContextBudgetResponse,
    ContextBudgetUpdateRequest,
    ContextCapabilitiesResponse,
    ContextCompressRequest,
    ContextCompressResponse,
    ContextConfigurationError,
    ContextLimitExceededError,
    ContextPackage,
    ContextPreviewRequest,
    ContextPreviewResponse,
    ContextValidateRequest,
    ContextValidationReport,
    ConversationTurn,
    ModelContextCapabilities,
    WorkspaceNote,
    WorkspaceNoteCreate,
    WorkspaceNoteUpdate,
)
from web_intelligence.app.context.token_estimator import (
    BaseContextTokenEstimator,
    normalize_model_context_capabilities,
    resolve_token_estimator,
)
from web_intelligence.app.llm.manager import (
    ConnectionNotFoundError,
    LLMConnectionManager,
    NoActiveConnectionError,
)
from web_intelligence.app.schemas.evidence import ClaimConflict, EvidenceItem, ExtractedClaim
from web_intelligence.app.schemas.llm import LLMConnectionConfig, ModelCapabilities

if TYPE_CHECKING:
    from web_intelligence.app.storage.repositories import ResearchRepository


class ResearchRunNotFoundError(LookupError):
    """Raised when a referenced research_id does not exist in local storage."""


class WorkspaceNoteNotFoundError(LookupError):
    """Raised when a referenced workspace note_id does not exist."""


class ContextManagementEngine:
    """Cohesive service facade implementing Stage 5 Adaptive Context Management."""

    def __init__(
        self,
        repository: ResearchRepository,
        llm_manager: LLMConnectionManager,
        *,
        default_profile: AllocationProfile = AllocationProfile.STANDARD_RESEARCH,
        default_context_limit: int = 8192,
        default_output_tokens: int = 1024,
        budget_manager: DynamicContextBudgetManager | None = None,
        builder: StructuredContextBuilder | None = None,
        compressor: ContextCompressor | None = None,
        history_manager: ConversationHistoryManager | None = None,
        validator: ContextValidator | None = None,
    ) -> None:
        self._repository = repository
        self._llm_manager = llm_manager
        self._default_profile = default_profile
        self._budget_manager = budget_manager or DynamicContextBudgetManager(
            default_context_limit=default_context_limit,
            default_output_tokens=default_output_tokens,
        )
        self._history_manager = history_manager or ConversationHistoryManager()
        self._compressor = compressor or ContextCompressor()
        self._builder = builder or StructuredContextBuilder(
            budget_manager=self._budget_manager,
            compressor=self._compressor,
            history_manager=self._history_manager,
        )
        self._validator = validator or ContextValidator()

    async def inspect_capabilities(
        self,
        *,
        connection_id: str | None = None,
        user_context_limit: int | None = None,
        refresh_live: bool = False,
    ) -> tuple[ModelContextCapabilities, Any, LLMConnectionConfig | None, ModelCapabilities | None]:
        """Inspect normalized capabilities alongside live adapter and token estimator."""
        config: LLMConnectionConfig | None = None
        adapter: Any = None
        raw_caps: ModelCapabilities | None = None

        if connection_id is not None:
            config, adapter = await self._llm_manager.get_adapter(connection_id)
        else:
            config = await self._llm_manager.get_active_config()
            if config is not None:
                try:
                    adapter = await self._llm_manager.get_active_adapter()
                except NoActiveConnectionError:
                    adapter = None

        extra_warnings: list[str] = []
        if adapter is not None:
            if refresh_live:
                inspect_fn = getattr(adapter, "inspect_model_capabilities", None)
                if callable(inspect_fn):
                    try:
                        maybe = inspect_fn(config.model if config else None)
                        raw_caps = await maybe if inspect.isawaitable(maybe) else maybe
                    except Exception as exc:
                        extra_warnings.append(
                            f"Live capability refresh failed ({type(exc).__name__}); using saved capabilities."
                        )
            if raw_caps is None:
                caps_fn = getattr(adapter, "capabilities", None)
                if callable(caps_fn):
                    try:
                        maybe = caps_fn()
                        raw_caps = await maybe if inspect.isawaitable(maybe) else maybe
                    except Exception as exc:
                        extra_warnings.append(
                            f"Adapter capability check failed ({type(exc).__name__}); using configured limits."
                        )

        if raw_caps is None and config is not None:
            raw_caps = config.capabilities

        normalized = normalize_model_context_capabilities(
            raw_caps,
            config=config,
            user_context_limit=user_context_limit,
            adapter=adapter,
            default_context_limit=self._budget_manager._default_context_limit,
            default_output_tokens=self._budget_manager._default_output_tokens,
        )
        if extra_warnings:
            normalized = normalized.model_copy(
                update={"warnings": [*normalized.warnings, *extra_warnings]}
            )
        return normalized, adapter, config, raw_caps

    async def get_capabilities_response(
        self,
        *,
        connection_id: str | None = None,
        refresh_live: bool = False,
    ) -> ContextCapabilitiesResponse:
        """Build response payload for GET /api/v1/context/capabilities."""
        caps, *_ = await self.inspect_capabilities(
            connection_id=connection_id,
            refresh_live=refresh_live,
        )
        return ContextCapabilitiesResponse(
            capabilities=caps,
            available_profiles=list(AllocationProfile),
            compression_strategies=list(CompressionStrategy),
            default_profile=self._default_profile,
        )

    async def get_or_default_budget_config(self) -> ContextBudgetConfig:
        """Return persisted ContextBudgetConfig or default instance."""
        saved = await self._repository.get_context_budget_config("default")
        if saved is not None:
            return saved
        return ContextBudgetConfig(profile=self._default_profile)

    async def get_budget_response(
        self, *, connection_id: str | None = None
    ) -> ContextBudgetResponse:
        """Return active config, model capabilities, and computed allocation."""
        config = await self.get_or_default_budget_config()
        caps, adapter, _conn, raw_caps = await self.inspect_capabilities(
            connection_id=connection_id
        )
        estimator = resolve_token_estimator(adapter, raw_caps)

        # Ensure read path stays resilient even if user switched to a smaller model than saved config
        safe_config = config
        runtime_warnings: list[str] = []
        if (
            config.user_context_limit is not None
            and config.user_context_limit > caps.max_context_tokens
        ):
            runtime_warnings.append(
                f"Saved user_context_limit ({config.user_context_limit}) exceeds active model limit "
                f"({caps.max_context_tokens}); displaying clamped allocation."
            )
            safe_config = config.model_copy(
                update={"user_context_limit": caps.max_context_tokens}
            )
        if (
            safe_config.output_reservation_tokens is not None
            and safe_config.output_reservation_tokens >= caps.max_context_tokens - 128
        ):
            clamped_out = max(64, min(caps.max_output_tokens, caps.max_context_tokens // 4))
            runtime_warnings.append(
                f"Saved output_reservation_tokens ({safe_config.output_reservation_tokens}) adjusted to "
                f"{clamped_out} for active model window."
            )
            safe_config = safe_config.model_copy(update={"output_reservation_tokens": clamped_out})

        allocation = self._budget_manager.plan_budget(
            caps,
            safe_config,
            estimator=estimator,
        )
        if runtime_warnings:
            allocation = allocation.model_copy(
                update={"warnings": [*allocation.warnings, *runtime_warnings]}
            )

        return ContextBudgetResponse(
            config=config,
            capabilities=caps,
            effective_allocation=allocation,
        )

    async def update_budget(
        self,
        update: ContextBudgetUpdateRequest,
        *,
        connection_id: str | None = None,
    ) -> ContextBudgetResponse:
        """Validate budget updates strictly against active model limits and persist."""
        current = await self.get_or_default_budget_config()
        caps, *_ = await self.inspect_capabilities(connection_id=connection_id)
        updated_config, allocation = self._budget_manager.validate_and_apply_update(
            current, update, caps
        )
        await self._repository.save_context_budget_config(updated_config)
        return ContextBudgetResponse(
            config=updated_config,
            capabilities=caps,
            effective_allocation=allocation,
        )

    async def preview_context(
        self, request: ContextPreviewRequest
    ) -> ContextPreviewResponse:
        """Build structured ContextPackage, adapter messages, and validation report."""
        package = request.research_package
        if package is None and request.research_id:
            package = await self._repository.get_research_package(request.research_id)
            if package is None:
                raise ResearchRunNotFoundError(
                    f"Research run '{request.research_id}' not found."
                )

        base_config = await self.get_or_default_budget_config()
        override_patch: dict[str, Any] = {}
        if request.profile is not None:
            override_patch["profile"] = request.profile
        if request.user_context_limit is not None:
            override_patch["user_context_limit"] = request.user_context_limit
        if request.output_reservation_tokens is not None:
            override_patch["output_reservation_tokens"] = request.output_reservation_tokens
        if request.safety_margin_tokens is not None:
            override_patch["safety_margin_tokens"] = request.safety_margin_tokens
        if request.custom_ratios is not None:
            override_patch["custom_ratios"] = request.custom_ratios
            if request.profile is None:
                override_patch["profile"] = AllocationProfile.CUSTOM
        if request.compression_strategy is not None:
            override_patch["compression_strategy"] = request.compression_strategy
        if request.allow_model_assisted_compression:
            override_patch["allow_model_assisted_compression"] = True
        if request.expected_output_length is not None:
            override_patch["expected_output_length"] = request.expected_output_length

        effective_config = (
            base_config.model_copy(update=override_patch) if override_patch else base_config
        )

        caps, adapter, _conn, raw_caps = await self.inspect_capabilities(
            connection_id=request.connection_id,
            user_context_limit=effective_config.user_context_limit,
        )
        estimator = resolve_token_estimator(adapter, raw_caps)

        session_key = (
            request.research_id
            or (package.research_id if package else None)
            or "global"
        )
        stored_turns = await self._repository.list_conversation_turns(session_key)
        inline_turns = [
            self._history_manager.new_turn(
                role=t.role,
                content=t.content,
                research_id=session_key,
                scope=t.scope,
                decisions=t.decisions,
                unresolved_questions=t.unresolved_questions,
            )
            for t in request.conversation_history
        ]
        all_turns = [*stored_turns, *inline_turns]

        notes = (
            await self._repository.list_workspace_notes("default")
            if request.include_workspace_notes
            else []
        )

        question = (
            request.question
            or (package.original_query if package else "")
            or ""
        )

        context_pkg = await self._builder.build_async(
            question=question,
            research_id=package.research_id if package else request.research_id,
            research_objective=request.research_objective,
            system_instructions=request.system_instructions,
            evidence_items=package.evidence if package else [],
            sources=package.sources if package else [],
            claims=package.claims if package else [],
            conflicts=package.conflicts if package else [],
            coverage=package.coverage if package else None,
            capabilities=caps,
            budget_config=effective_config,
            turns=all_turns,
            workspace_notes=notes,
            estimator=estimator,
            adapter=adapter,
            expected_output_length=request.expected_output_length,
        )
        adapter_messages = ContextAdapterConverter.to_messages(
            context_pkg, prefer_json=caps.supports_structured_output
        )
        report = self._validator.validate(
            context_pkg,
            adapter_messages=adapter_messages,
            estimator=estimator,
        )
        return ContextPreviewResponse(
            context_package=context_pkg,
            validation=report,
            adapter_messages=adapter_messages,
        )

    async def validate_context(
        self, request: ContextValidateRequest
    ) -> ContextValidationReport:
        """Validate either a provided ContextPackage or preview-built context."""
        if request.context_package is not None:
            pkg = request.context_package
            caps, adapter, _conn, raw_caps = await self.inspect_capabilities(
                connection_id=request.connection_id,
                user_context_limit=request.user_context_limit,
            )
            if request.connection_id or request.user_context_limit:
                pkg = pkg.model_copy(update={"capabilities": caps})
            estimator = resolve_token_estimator(adapter, raw_caps)
            msgs = ContextAdapterConverter.to_messages(pkg)
            return self._validator.validate(
                pkg,
                adapter_messages=msgs,
                estimator=estimator,
            )

        assert request.preview_request is not None
        preview_req = request.preview_request
        if request.connection_id and not preview_req.connection_id:
            preview_req = preview_req.model_copy(update={"connection_id": request.connection_id})
        if request.user_context_limit and not preview_req.user_context_limit:
            preview_req = preview_req.model_copy(update={"user_context_limit": request.user_context_limit})
        preview = await self.preview_context(preview_req)
        return preview.validation

    async def compress_evidence(
        self, request: ContextCompressRequest
    ) -> ContextCompressResponse:
        """Compress evidence items preserving original evidence and conflict attributions."""
        evidence_items: list[EvidenceItem]
        conflicts: list[ClaimConflict]
        claims: list[ExtractedClaim]
        query = ""

        if request.research_id:
            pkg = await self._repository.get_research_package(request.research_id)
            if pkg is None:
                raise ResearchRunNotFoundError(
                    f"Research run '{request.research_id}' not found."
                )
            evidence_items = pkg.evidence
            conflicts = pkg.conflicts
            claims = pkg.claims
            query = pkg.original_query
        else:
            evidence_items = request.evidence_items
            conflicts = request.conflicts
            claims = request.claims

        _caps, adapter, _conn, raw_caps = await self.inspect_capabilities(
            connection_id=request.connection_id
        )
        estimator = resolve_token_estimator(adapter, raw_caps)
        selected_inputs = ContextCompressor.evidence_items_to_selected(
            evidence_items, conflicts=conflicts, estimator=estimator
        )
        compressor = ContextCompressor(estimator=estimator)
        result = await compressor.compress_async(
            selected_inputs,
            strategy=request.strategy,
            query=query,
            target_tokens=request.target_tokens,
            claims=claims,
            conflicts=conflicts,
            adapter=adapter,
            allow_model_assisted=request.allow_model_assisted,
        )
        preserved_conflicts = [cf.conflict_id for cf in conflicts]
        return ContextCompressResponse(
            compression=result,
            original_evidence=evidence_items,
            preserved_conflict_ids=preserved_conflicts,
        )

    # --- Workspace Notes & Conversation History Management ---

    async def list_workspace_notes(self, workspace_id: str = "default") -> list[WorkspaceNote]:
        """List user-controlled workspace notes."""
        return await self._repository.list_workspace_notes(workspace_id)

    async def create_workspace_note(self, payload: WorkspaceNoteCreate) -> WorkspaceNote:
        """Create and persist a user-controlled workspace note."""
        note = self._history_manager.new_note(
            title=payload.title,
            content=payload.content,
            workspace_id=payload.workspace_id,
            enabled=payload.enabled,
        )
        return await self._repository.save_workspace_note(note)

    async def update_workspace_note(
        self, note_id: str, payload: WorkspaceNoteUpdate
    ) -> WorkspaceNote:
        """Update or toggle a user-controlled workspace note."""
        existing = await self._repository.get_workspace_note(note_id)
        if existing is None:
            raise WorkspaceNoteNotFoundError(f"Workspace note '{note_id}' not found.")
        from datetime import datetime, timezone

        patch: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if payload.title is not None:
            patch["title"] = payload.title.strip()
        if payload.content is not None:
            patch["content"] = payload.content.strip()
        if payload.enabled is not None:
            patch["enabled"] = payload.enabled
        updated = existing.model_copy(update=patch)
        return await self._repository.save_workspace_note(updated)

    async def delete_workspace_note(self, note_id: str) -> bool:
        """Delete a persisted workspace note."""
        return await self._repository.delete_workspace_note(note_id)

    async def add_conversation_turn(
        self,
        *,
        research_id: str,
        role: str,
        content: str,
        scope: str = "session",
        decisions: list[str] | None = None,
        unresolved_questions: list[str] | None = None,
    ) -> ConversationTurn:
        """Persist a conversation turn for session context tracking."""
        turn = self._history_manager.new_turn(
            role=role,
            content=content,
            research_id=research_id,
            scope=scope,
            decisions=decisions,
            unresolved_questions=unresolved_questions,
        )
        return await self._repository.save_conversation_turn(turn)
