"""High-level AnswerGenerator coordinating Stage 5 context, prompt compilation, local inference, citations, and persistence."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any

from web_intelligence.app.context.engine import (
    ContextManagementEngine,
    ResearchRunNotFoundError,
)
from web_intelligence.app.context.models import (
    AllocationProfile,
    ContextConfigurationError,
    ContextPreviewRequest,
    ContextPreviewResponse,
    ExpectedOutputLength,
)
from web_intelligence.app.context.token_estimator import resolve_token_estimator
from web_intelligence.app.generation.citations import CitationValidator
from web_intelligence.app.generation.executor import (
    ContextBudgetBlockedError,
    GenerationError,
    InferenceCancelledError,
    InferenceExecutor,
    ResolvedInferenceTarget,
)
from web_intelligence.app.generation.long_form import LongFormReportGenerator
from web_intelligence.app.generation.prompt_compiler import PromptCompiler
from web_intelligence.app.llm.base import extract_json_payload
from web_intelligence.app.schemas.generation import (
    GenerationContextStats,
    GenerationFormat,
    GenerationLength,
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    PromptTemplateType,
)
from web_intelligence.app.schemas.research import ResearchPackage
from web_intelligence.app.storage.repositories import ResearchRepository


def _resolve_profile(request: GenerationRequest) -> AllocationProfile:
    if request.allocation_profile is not None:
        return AllocationProfile(request.allocation_profile)
    if request.generation_mode == GenerationMode.LONG_FORM_REPORT:
        return AllocationProfile.LONG_FORM
    if request.template_type == PromptTemplateType.QUICK_ANSWER:
        return AllocationProfile.QUICK_ANSWER
    if request.template_type == PromptTemplateType.RESEARCH_REPORT:
        return AllocationProfile.DEEP_RESEARCH
    if request.answer_length == GenerationLength.LONG:
        return AllocationProfile.DEEP_RESEARCH
    return AllocationProfile.STANDARD_RESEARCH


def _resolve_expected_length(answer_length: GenerationLength) -> ExpectedOutputLength:
    if answer_length == GenerationLength.SHORT:
        return ExpectedOutputLength.SHORT
    if answer_length == GenerationLength.LONG:
        return ExpectedOutputLength.LONG
    return ExpectedOutputLength.MEDIUM


def _build_context_stats(preview: ContextPreviewResponse) -> GenerationContextStats:
    pkg = preview.context_package
    val_warnings = [
        issue.message
        for issue in preview.validation.issues
        if issue.severity == "warning"
    ]
    return GenerationContextStats(
        context_limit=pkg.capabilities.effective_context_limit,
        reserved_output_tokens=pkg.output_reservation_tokens,
        usable_input_budget=pkg.budget_allocation.total_planned_tokens
        - pkg.output_reservation_tokens,
        used_input_tokens=pkg.total_input_tokens,
        evidence_items_included=len(pkg.selected_evidence),
        evidence_items_dropped=len(pkg.excluded_evidence),
        compression_applied=pkg.compression.strategy_applied.value != "none",
        validation_passed=preview.validation.is_valid,
        validation_warnings=val_warnings,
    )


class AnswerGenerator:
    """Orchestrates grounded answer and multi-stage report generation against user-hosted local LLMs."""

    def __init__(
        self,
        *,
        repository: ResearchRepository,
        context_engine: ContextManagementEngine,
        executor: InferenceExecutor,
        prompt_compiler: PromptCompiler | None = None,
        citation_validator: CitationValidator | None = None,
        long_form_generator: LongFormReportGenerator | None = None,
    ) -> None:
        self._repository = repository
        self._context_engine = context_engine
        self._executor = executor
        self._compiler = prompt_compiler or PromptCompiler()
        self._validator = citation_validator or CitationValidator()
        self._long_form = long_form_generator or LongFormReportGenerator(
            executor=executor,
            prompt_compiler=self._compiler,
            citation_validator=self._validator,
        )

    async def get_generation(self, generation_id: str) -> GenerationResult | None:
        """Load one persisted generation run by ID."""
        return await self._repository.get_generation_result(generation_id)

    async def list_generations(
        self, research_id: str | None = None, limit: int = 20
    ) -> list[GenerationResult]:
        """List recent generation runs for a research session or workspace."""
        return await self._repository.list_generation_results(
            research_id=research_id, limit=limit
        )

    async def cancel_generation(self, generation_id: str) -> GenerationResult | None:
        """Request cooperative cancellation on an active run and return current persisted state."""
        signalled = self._executor.job_registry.cancel(generation_id)
        existing = await self._repository.get_generation_result(generation_id)
        if existing is None:
            return None
        if existing.status in {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        }:
            return existing
        # If worker hasn't finished transitioning yet, record intent cleanly when inactive
        if not signalled:
            updated = existing.model_copy(
                update={
                    "status": GenerationStatus.CANCELLED,
                    "partial": bool(existing.content),
                    "completed_at": datetime.now(timezone.utc),
                }
            )
            return await self._repository.save_generation_result(updated)
        return existing

    async def _prepare_inputs(
        self, request: GenerationRequest
    ) -> tuple[ResearchPackage, ResolvedInferenceTarget, ContextPreviewResponse]:
        package = await self._repository.get_research_package(request.research_id)
        if package is None:
            raise ResearchRunNotFoundError(
                f"Research run '{request.research_id}' not found."
            )

        target = await self._executor.resolve_target(
            connection_id=request.connection_id,
            model_id=request.model_id,
            task_role=request.task_role,
        )
        system_directives = self._compiler.build_system_directives(
            template_type=request.template_type,
            output_format=request.output_format,
            answer_length=request.answer_length,
            require_citations=request.require_citations,
            json_schema_description=request.json_schema_description,
            custom_instruction=request.custom_instruction,
        )
        # Keep effective context limit grounded strictly to selected target capabilities
        cap_out = min(
            request.max_output_tokens or target.capabilities.recommended_output_tokens,
            max(64, target.capabilities.max_context_tokens - 128),
        )
        try:
            preview = await self._context_engine.preview_context(
                ContextPreviewRequest(
                    research_id=package.research_id,
                    research_package=package,
                    question=package.original_query,
                    system_instructions=system_directives,
                    connection_id=target.connection_id,
                    profile=_resolve_profile(request),
                    expected_output_length=_resolve_expected_length(request.answer_length),
                    user_context_limit=target.capabilities.max_context_tokens,
                    output_reservation_tokens=cap_out,
                    include_workspace_notes=request.include_workspace_notes,
                )
            )
        except ContextConfigurationError as exc:
            raise ContextBudgetBlockedError(
                str(exc),
                error_code="context_configuration_blocked",
                status_code=422,
            ) from exc
        if not preview.validation.is_valid:
            errs = "; ".join(
                f"[{i.code}] {i.message}"
                for i in preview.validation.issues
                if i.severity == "error"
            )
            raise ContextBudgetBlockedError(
                f"Stage 5 context validation failed: {errs or 'invalid context'}",
                error_code="context_validation_failed",
                status_code=422,
            )
        return package, target, preview

    def _finalize_standard_result(
        self,
        *,
        generation_id: str,
        created_at: datetime,
        started_monotonic: float,
        request: GenerationRequest,
        target: ResolvedInferenceTarget,
        preview: ContextPreviewResponse,
        compiled: Any,
        content: str,
        status: GenerationStatus,
        partial: bool,
        finish_reason: str | None,
        usage: Any,
        warnings: list[str],
        error: str | None = None,
        error_code: str | None = None,
    ) -> GenerationResult:
        is_json = request.output_format == GenerationFormat.JSON
        resolved_citations, val_report = self._validator.validate(
            content,
            compiled.citation_map,
            stage5_to_citation_id=compiled.stage5_to_citation_id,
            require_citations=request.require_citations,
            json_mode=is_json,
        )
        parsed_json = None
        if is_json and content.strip() and status == GenerationStatus.COMPLETED:
            parsed_json = extract_json_payload(content)
            if parsed_json is None:
                warnings.append(
                    "Model response for JSON output format could not be parsed as valid JSON."
                )

        truncated = finish_reason == "length"
        if truncated:
            warnings.append(
                f"Output reached max_output_tokens limit ({compiled.reserved_output_tokens}) and may be truncated."
            )

        dedup_warnings = list(dict.fromkeys([*compiled.warnings, *warnings]))
        return GenerationResult(
            generation_id=generation_id,
            research_id=request.research_id,
            status=status,
            partial=partial,
            mode=request.generation_mode,
            template_type=request.template_type,
            output_format=request.output_format,
            connection_id=target.connection_id,
            provider_id=target.provider_id,
            server_endpoint=target.server_endpoint,
            model_id=target.model_id,
            content=content,
            parsed_json=parsed_json,
            finish_reason=finish_reason,
            truncated=truncated,
            citations=resolved_citations,
            source_map=compiled.citation_map,
            citation_validation=val_report,
            usage=usage,
            estimated_prompt_tokens=compiled.estimated_prompt_tokens,
            context_stats=_build_context_stats(preview),
            error=error,
            error_code=error_code,
            warnings=dedup_warnings,
            duration_ms=max(0, int((time.monotonic() - started_monotonic) * 1000)),
            created_at=created_at,
            completed_at=datetime.now(timezone.utc),
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute complete generation run (standard or long-form report) and persist result."""
        if request.generation_mode == GenerationMode.LONG_FORM_REPORT:
            final_res: GenerationResult | None = None
            async for ev in self.stream_generate(request):
                if ev.get("event") in {"complete", "cancelled"} and ev.get("result_obj"):
                    final_res = ev["result_obj"]
                elif ev.get("event") == "error":
                    raise GenerationError(
                        ev.get("detail", "Generation failed."),
                        error_code=ev.get("error_code", "generation_error"),
                        status_code=ev.get("status_code", 500),
                        partial_content=ev.get("partial_content", ""),
                    )
            assert final_res is not None
            return final_res

        started = time.monotonic()
        now = datetime.now(timezone.utc)
        generation_id = f"gen-{uuid.uuid4().hex[:12]}"
        package, target, preview = await self._prepare_inputs(request)

        estimator = resolve_token_estimator(target.adapter)
        compiled = self._compiler.compile(
            preview.context_package,
            template_type=request.template_type,
            output_format=request.output_format,
            answer_length=request.answer_length,
            require_citations=request.require_citations,
            reserved_output_tokens=preview.context_package.output_reservation_tokens,
            json_schema_description=request.json_schema_description,
            custom_instruction=request.custom_instruction,
            token_estimator=estimator,
        )

        initial = GenerationResult(
            generation_id=generation_id,
            research_id=package.research_id,
            status=GenerationStatus.PENDING,
            mode=request.generation_mode,
            template_type=request.template_type,
            output_format=request.output_format,
            connection_id=target.connection_id,
            provider_id=target.provider_id,
            server_endpoint=target.server_endpoint,
            model_id=target.model_id,
            source_map=compiled.citation_map,
            estimated_prompt_tokens=compiled.estimated_prompt_tokens,
            context_stats=_build_context_stats(preview),
            created_at=now,
        )
        await self._repository.save_generation_result(initial)
        cancel_event = self._executor.job_registry.register(generation_id)

        try:
            resp, exec_warnings = await self._executor.execute(
                target=target,
                context_package=preview.context_package,
                compiled_prompt=compiled,
                context_validation=preview.validation,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                stop_sequences=request.stop_sequences,
                timeout_seconds=request.timeout_seconds,
                cancel_event=cancel_event,
            )
            result = self._finalize_standard_result(
                generation_id=generation_id,
                created_at=now,
                started_monotonic=started,
                request=request,
                target=target,
                preview=preview,
                compiled=compiled,
                content=resp.content,
                status=GenerationStatus.COMPLETED,
                partial=False,
                finish_reason=resp.finish_reason,
                usage=resp.usage,
                warnings=exec_warnings,
            )
            return await self._repository.save_generation_result(result)
        except InferenceCancelledError as exc:
            cancelled = self._finalize_standard_result(
                generation_id=generation_id,
                created_at=now,
                started_monotonic=started,
                request=request,
                target=target,
                preview=preview,
                compiled=compiled,
                content=exc.partial_content,
                status=GenerationStatus.CANCELLED,
                partial=True,
                finish_reason="cancelled",
                usage=exc.usage,
                warnings=["Generation cancelled before completion."],
                error=str(exc),
                error_code=exc.error_code,
            )
            return await self._repository.save_generation_result(cancelled)
        except GenerationError as exc:
            failed = self._finalize_standard_result(
                generation_id=generation_id,
                created_at=now,
                started_monotonic=started,
                request=request,
                target=target,
                preview=preview,
                compiled=compiled,
                content=exc.partial_content,
                status=GenerationStatus.FAILED,
                partial=bool(exc.partial_content),
                finish_reason="error",
                usage=exc.usage,
                warnings=[],
                error=str(exc),
                error_code=exc.error_code,
            )
            await self._repository.save_generation_result(failed)
            raise
        finally:
            self._executor.job_registry.unregister(generation_id)

    async def stream_generate(
        self,
        request: GenerationRequest,
        *,
        generation_id: str | None = None,
        is_disconnected: Callable[[], Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream live generation events (delta, section, citation, error, cancelled, complete)."""
        started = time.monotonic()
        now = datetime.now(timezone.utc)
        gen_id = generation_id or f"gen-{uuid.uuid4().hex[:12]}"

        package, target, preview = await self._prepare_inputs(request)
        context_stats = _build_context_stats(preview)

        if request.generation_mode == GenerationMode.LONG_FORM_REPORT:
            global_map, _, _ = self._compiler.build_citation_registry(preview.context_package)
            initial = GenerationResult(
                generation_id=gen_id,
                research_id=package.research_id,
                status=GenerationStatus.STREAMING,
                mode=request.generation_mode,
                template_type=request.template_type,
                output_format=GenerationFormat.MARKDOWN,
                connection_id=target.connection_id,
                provider_id=target.provider_id,
                server_endpoint=target.server_endpoint,
                model_id=target.model_id,
                source_map=global_map,
                context_stats=context_stats,
                created_at=now,
            )
            await self._repository.save_generation_result(initial)
            cancel_event = self._executor.job_registry.register(gen_id)
            yield {
                "event": "start",
                "generation_id": gen_id,
                "research_id": package.research_id,
                "mode": request.generation_mode.value,
                "model_id": target.model_id,
                "connection_id": target.connection_id,
                "server_endpoint": target.server_endpoint,
                "source_map": {k: v.model_dump(mode="json") for k, v in global_map.items()},
                "context_stats": context_stats.model_dump(mode="json"),
            }
            accumulated = ""
            try:
                async for ev in self._long_form.stream_generate(
                    generation_id=gen_id,
                    request=request,
                    package=package,
                    target=target,
                    capabilities=preview.context_package.capabilities,
                    global_citation_map=global_map,
                    cancel_event=cancel_event,
                    is_disconnected=is_disconnected,
                ):
                    if ev["event"] == "delta":
                        accumulated += ev.get("delta", "")
                        yield ev
                    elif ev["event"] in {"outline", "section_start", "section_complete"}:
                        yield ev
                    elif ev["event"] == "result":
                        res: GenerationResult = ev["result"].model_copy(
                            update={"context_stats": context_stats}
                        )
                        saved = await self._repository.save_generation_result(res)
                        event_name = (
                            "cancelled"
                            if saved.status == GenerationStatus.CANCELLED
                            else "complete"
                        )
                        yield {
                            "event": "citation",
                            "generation_id": gen_id,
                            "citations": [c.model_dump(mode="json") for c in saved.citations],
                            "citation_validation": (
                                saved.citation_validation.model_dump(mode="json")
                                if saved.citation_validation
                                else None
                            ),
                        }
                        yield {
                            "event": event_name,
                            "generation_id": gen_id,
                            "generation": saved.model_dump(mode="json"),
                            "result_obj": saved,
                        }
            except GenerationError as exc:
                failed = GenerationResult(
                    generation_id=gen_id,
                    research_id=package.research_id,
                    status=GenerationStatus.FAILED,
                    partial=bool(accumulated),
                    mode=request.generation_mode,
                    template_type=request.template_type,
                    output_format=GenerationFormat.MARKDOWN,
                    connection_id=target.connection_id,
                    provider_id=target.provider_id,
                    server_endpoint=target.server_endpoint,
                    model_id=target.model_id,
                    content=accumulated,
                    source_map=global_map,
                    context_stats=context_stats,
                    error=str(exc),
                    error_code=exc.error_code,
                    duration_ms=max(0, int((time.monotonic() - started) * 1000)),
                    created_at=now,
                    completed_at=datetime.now(timezone.utc),
                )
                saved_fail = await self._repository.save_generation_result(failed)
                yield {
                    "event": "error",
                    "generation_id": gen_id,
                    "detail": str(exc),
                    "error_code": exc.error_code,
                    "status_code": exc.status_code,
                    "partial_content": accumulated,
                    "generation": saved_fail.model_dump(mode="json"),
                }
            finally:
                self._executor.job_registry.unregister(gen_id)
            return

        # Standard single-pass streaming execution
        estimator = resolve_token_estimator(target.adapter)
        compiled = self._compiler.compile(
            preview.context_package,
            template_type=request.template_type,
            output_format=request.output_format,
            answer_length=request.answer_length,
            require_citations=request.require_citations,
            reserved_output_tokens=preview.context_package.output_reservation_tokens,
            json_schema_description=request.json_schema_description,
            custom_instruction=request.custom_instruction,
            token_estimator=estimator,
        )

        initial = GenerationResult(
            generation_id=gen_id,
            research_id=package.research_id,
            status=GenerationStatus.STREAMING,
            mode=request.generation_mode,
            template_type=request.template_type,
            output_format=request.output_format,
            connection_id=target.connection_id,
            provider_id=target.provider_id,
            server_endpoint=target.server_endpoint,
            model_id=target.model_id,
            source_map=compiled.citation_map,
            estimated_prompt_tokens=compiled.estimated_prompt_tokens,
            context_stats=context_stats,
            created_at=now,
        )
        await self._repository.save_generation_result(initial)
        cancel_event = self._executor.job_registry.register(gen_id)

        yield {
            "event": "start",
            "generation_id": gen_id,
            "research_id": package.research_id,
            "mode": request.generation_mode.value,
            "model_id": target.model_id,
            "connection_id": target.connection_id,
            "server_endpoint": target.server_endpoint,
            "source_map": {k: v.model_dump(mode="json") for k, v in compiled.citation_map.items()},
            "context_stats": context_stats.model_dump(mode="json"),
        }

        accumulated = ""
        finish_reason: str | None = None
        usage = None
        exec_warnings: list[str] = []

        try:
            async for ev in self._executor.execute_stream(
                target=target,
                context_package=preview.context_package,
                compiled_prompt=compiled,
                context_validation=preview.validation,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                stop_sequences=request.stop_sequences,
                timeout_seconds=request.timeout_seconds,
                cancel_event=cancel_event,
                is_disconnected=is_disconnected,
            ):
                if ev.kind == "warning" and ev.warning:
                    exec_warnings.append(ev.warning)
                    yield {"event": "warning", "generation_id": gen_id, "warning": ev.warning}
                elif ev.kind == "delta":
                    accumulated = ev.accumulated
                    yield {
                        "event": "delta",
                        "generation_id": gen_id,
                        "delta": ev.delta,
                        "model_id": ev.model_id,
                    }
                elif ev.kind == "done":
                    accumulated = ev.accumulated
                    finish_reason = ev.finish_reason
                    usage = ev.usage

            result = self._finalize_standard_result(
                generation_id=gen_id,
                created_at=now,
                started_monotonic=started,
                request=request,
                target=target,
                preview=preview,
                compiled=compiled,
                content=accumulated,
                status=GenerationStatus.COMPLETED,
                partial=False,
                finish_reason=finish_reason,
                usage=usage,
                warnings=exec_warnings,
            )
            saved = await self._repository.save_generation_result(result)
            yield {
                "event": "citation",
                "generation_id": gen_id,
                "citations": [c.model_dump(mode="json") for c in saved.citations],
                "citation_validation": (
                    saved.citation_validation.model_dump(mode="json")
                    if saved.citation_validation
                    else None
                ),
            }
            yield {
                "event": "complete",
                "generation_id": gen_id,
                "generation": saved.model_dump(mode="json"),
                "result_obj": saved,
            }
        except InferenceCancelledError as exc:
            partial_text = exc.partial_content or accumulated
            cancelled = self._finalize_standard_result(
                generation_id=gen_id,
                created_at=now,
                started_monotonic=started,
                request=request,
                target=target,
                preview=preview,
                compiled=compiled,
                content=partial_text,
                status=GenerationStatus.CANCELLED,
                partial=True,
                finish_reason="cancelled",
                usage=exc.usage or usage,
                warnings=[*exec_warnings, str(exc)],
                error=str(exc),
                error_code=exc.error_code,
            )
            saved_cancel = await self._repository.save_generation_result(cancelled)
            yield {
                "event": "citation",
                "generation_id": gen_id,
                "citations": [c.model_dump(mode="json") for c in saved_cancel.citations],
                "citation_validation": (
                    saved_cancel.citation_validation.model_dump(mode="json")
                    if saved_cancel.citation_validation
                    else None
                ),
            }
            yield {
                "event": "cancelled",
                "generation_id": gen_id,
                "generation": saved_cancel.model_dump(mode="json"),
                "result_obj": saved_cancel,
            }
        except GenerationError as exc:
            partial_text = exc.partial_content or accumulated
            failed = self._finalize_standard_result(
                generation_id=gen_id,
                created_at=now,
                started_monotonic=started,
                request=request,
                target=target,
                preview=preview,
                compiled=compiled,
                content=partial_text,
                status=GenerationStatus.FAILED,
                partial=bool(partial_text),
                finish_reason="error",
                usage=exc.usage or usage,
                warnings=exec_warnings,
                error=str(exc),
                error_code=exc.error_code,
            )
            saved_fail = await self._repository.save_generation_result(failed)
            yield {
                "event": "error",
                "generation_id": gen_id,
                "detail": str(exc),
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "partial_content": partial_text,
                "generation": saved_fail.model_dump(mode="json"),
            }
        finally:
            self._executor.job_registry.unregister(gen_id)
