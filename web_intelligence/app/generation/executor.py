"""Centralized local LLM InferenceExecutor with context gating, cooperative cancellation, and normalized errors."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from web_intelligence.app.context.context_validator import ContextValidator
from web_intelligence.app.context.models import ContextPackage, ContextValidationReport
from web_intelligence.app.core.logging import get_logger
from web_intelligence.app.llm.base import (
    LLMAuthError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    UniversalLocalModelAdapter,
)
from web_intelligence.app.llm.manager import (
    ConnectionNotFoundError,
    LLMConnectionManager,
    NoActiveConnectionError,
)
from web_intelligence.app.llm.model_registry import (
    ModelRegistry,
    sanitize_endpoint_for_registry,
)
from web_intelligence.app.llm.security import LLMConnectionSecurityError, redact_llm_error
from web_intelligence.app.schemas.generation import (
    CompiledPrompt,
    GenerationFormat,
    ModelRegistryEntry,
    ModelTaskRole,
)
from web_intelligence.app.schemas.llm import (
    CapabilitySource,
    LLMConnectionConfig,
    LLMInferenceRequest,
    LLMInferenceResponse,
    ModelCapabilities,
    TokenUsage,
)

logger = get_logger("generation.executor")


class GenerationError(RuntimeError):
    """Normalized execution error exposing HTTP status, error code, and partial output."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "generation_error",
        status_code: int = 500,
        partial_content: str = "",
        usage: TokenUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code
        self.partial_content = partial_content
        self.usage = usage


class ModelNotConfiguredError(GenerationError):
    """No active or requested local LLM server/model is configured."""


class ContextBudgetBlockedError(GenerationError):
    """Pre-inference Stage 5 or compiled-prompt context limit validation failed."""


class ProviderUnavailableError(GenerationError):
    """Local inference server is unreachable or failed trust-boundary checks."""


class ProviderAuthFailedError(GenerationError):
    """Inference server rejected credentials (401/403)."""


class ProviderTimeoutError(GenerationError):
    """Inference request exceeded configured timeout."""


class ProviderResponseInvalidError(GenerationError):
    """Inference server returned malformed output or rejected request parameters."""


class InferenceCancelledError(GenerationError):
    """Inference run was interrupted cleanly by user cancellation or client disconnect."""

    def __init__(
        self,
        message: str = "Generation cancelled by user.",
        *,
        partial_content: str = "",
        usage: TokenUsage | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code="generation_cancelled",
            status_code=200,
            partial_content=partial_content,
            usage=usage,
        )


@dataclass
class ResolvedInferenceTarget:
    """Resolved model, server adapter, endpoint metadata, and capabilities for execution."""

    connection_id: str
    provider_id: str
    server_endpoint: str
    model_id: str
    config: LLMConnectionConfig
    adapter: UniversalLocalModelAdapter
    capabilities: ModelCapabilities
    registry_entry: ModelRegistryEntry | None = None


@dataclass
class ExecutorStreamEvent:
    """Internal event yielded during streaming inference execution."""

    kind: str  # "delta" | "warning" | "done"
    delta: str = ""
    accumulated: str = ""
    finish_reason: str | None = None
    model_id: str = ""
    usage: TokenUsage | None = None
    warning: str | None = None
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0


class GenerationJobRegistry:
    """Tracks in-flight generation cancellation signals cleanly by generation_id."""

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}

    def register(self, generation_id: str) -> asyncio.Event:
        """Create and register a fresh cancellation Event for `generation_id`."""
        event = asyncio.Event()
        self._events[generation_id] = event
        return event

    def cancel(self, generation_id: str) -> bool:
        """Trigger cancellation on an active generation run if currently executing."""
        event = self._events.get(generation_id)
        if event is None:
            return False
        event.set()
        return True

    def is_active(self, generation_id: str) -> bool:
        """Return True if `generation_id` is currently registered in flight."""
        return generation_id in self._events

    def is_cancelled(self, generation_id: str) -> bool:
        """Return True if `generation_id` exists and cancellation was requested."""
        event = self._events.get(generation_id)
        return bool(event is not None and event.is_set())

    def unregister(self, generation_id: str) -> None:
        """Remove `generation_id` after completion, cancellation, or error."""
        self._events.pop(generation_id, None)


class InferenceExecutor:
    """Provider-independent execution layer enforcing validation, capability gating, and cancellation."""

    def __init__(
        self,
        llm_manager: LLMConnectionManager,
        model_registry: ModelRegistry | None = None,
        *,
        context_validator: ContextValidator | None = None,
        job_registry: GenerationJobRegistry | None = None,
    ) -> None:
        self._llm_manager = llm_manager
        self._model_registry = model_registry
        self._context_validator = context_validator or ContextValidator()
        self.job_registry = job_registry or GenerationJobRegistry()

    async def resolve_target(
        self,
        *,
        connection_id: str | None = None,
        model_id: str | None = None,
        task_role: ModelTaskRole | None = None,
    ) -> ResolvedInferenceTarget:
        """Resolve server connection, SSRF-validated adapter, model identifier, and capabilities."""
        registry_entry: ModelRegistryEntry | None = None
        if self._model_registry is not None:
            registry_entry = await self._model_registry.resolve_target(
                connection_id=connection_id,
                model_id=model_id,
                task_role=task_role,
            )

        effective_conn_id = connection_id or (registry_entry.connection_id if registry_entry else None)

        try:
            cfg, adapter = await self._llm_manager.get_adapter(effective_conn_id)
        except NoActiveConnectionError as exc:
            raise ModelNotConfiguredError(
                "No local LLM model is configured or active. Connect a local inference server first.",
                error_code="no_active_model",
                status_code=400,
            ) from exc
        except ConnectionNotFoundError as exc:
            raise ModelNotConfiguredError(
                str(exc),
                error_code="model_connection_not_found",
                status_code=404,
            ) from exc
        except LLMConnectionSecurityError as exc:
            raise ProviderUnavailableError(
                str(exc),
                error_code="endpoint_security_blocked",
                status_code=400,
            ) from exc

        resolved_model = (
            (model_id or "").strip()
            or (registry_entry.model_id if registry_entry and registry_entry.connection_id == cfg.id else "")
            or cfg.model.strip()
        )
        if not resolved_model:
            raise ModelNotConfiguredError(
                f"Connection '{cfg.name}' has no model selected. Select or enter a model identifier.",
                error_code="no_model_selected",
                status_code=400,
            )

        caps = adapter.capabilities()
        if (
            registry_entry is not None
            and registry_entry.connection_id == cfg.id
            and registry_entry.model_id == resolved_model
        ):
            caps = ModelCapabilities(
                model_name=resolved_model,
                backend=cfg.provider,
                max_context_tokens=registry_entry.context_limit,
                recommended_output_tokens=registry_entry.max_output_limit,
                supports_streaming=registry_entry.supports_streaming,
                supports_tool_calling=registry_entry.supports_tool_calling,
                supports_json_mode=registry_entry.supports_json_mode,
                sources=registry_entry.capability_provenance,
                available_metadata=registry_entry.model_metadata,
            )

        return ResolvedInferenceTarget(
            connection_id=cfg.id,
            provider_id=cfg.provider,
            server_endpoint=sanitize_endpoint_for_registry(cfg.base_url, cfg.api_endpoint),
            model_id=resolved_model,
            config=cfg,
            adapter=adapter,
            capabilities=caps,
            registry_entry=registry_entry,
        )

    def _validate_preconditions(
        self,
        context_package: ContextPackage,
        compiled_prompt: CompiledPrompt,
        *,
        context_validation: ContextValidationReport | None = None,
    ) -> None:
        """Strictly reject invalid context packages or oversized compiled prompts."""
        report = context_validation or self._context_validator.validate(context_package)
        if not report.is_valid:
            err_msgs = [
                f"[{issue.code}] {issue.message}"
                for issue in report.issues
                if issue.severity == "error"
            ]
            summary = "; ".join(err_msgs) or "Context validation failed."
            raise ContextBudgetBlockedError(
                f"Context validation blocked inference: {summary}",
                error_code="context_validation_failed",
                status_code=422,
            )

        if not compiled_prompt.fits_context_limit:
            raise ContextBudgetBlockedError(
                (
                    f"Compiled prompt ({compiled_prompt.estimated_prompt_tokens} tok) + "
                    f"output reservation ({compiled_prompt.reserved_output_tokens} tok) = "
                    f"{compiled_prompt.total_required_tokens} tok exceeds limit "
                    f"({compiled_prompt.context_limit} tok). Reduce context budget or evidence depth."
                ),
                error_code="context_limit_exceeded",
                status_code=422,
            )

    def _build_inference_request(
        self,
        target: ResolvedInferenceTarget,
        compiled_prompt: CompiledPrompt,
        *,
        temperature: float,
        max_output_tokens: int | None,
        stop_sequences: list[str] | None,
        timeout_seconds: float | None,
        stream: bool,
    ) -> tuple[LLMInferenceRequest, list[str]]:
        """Assemble normalized LLMInferenceRequest gated conservatively by known capabilities."""
        warnings: list[str] = []
        caps = target.capabilities
        headroom = max(
            16,
            compiled_prompt.context_limit - compiled_prompt.estimated_prompt_tokens,
        )
        requested_out = max_output_tokens or compiled_prompt.reserved_output_tokens
        effective_out = max(16, min(requested_out, headroom))
        if max_output_tokens and effective_out < max_output_tokens:
            warnings.append(
                f"Requested max_output_tokens ({max_output_tokens}) clamped to {effective_out} "
                f"to fit context window ({compiled_prompt.context_limit})."
            )

        wants_json = compiled_prompt.output_format == GenerationFormat.JSON
        response_format = "text"
        if wants_json:
            if caps.supports_json_mode:
                response_format = "json_object"
            else:
                warnings.append(
                    "Model does not advertise native json_object mode; enforcing JSON via prompt instructions."
                )

        req = LLMInferenceRequest(
            messages=compiled_prompt.messages,
            model=target.model_id,
            temperature=temperature,
            max_tokens=effective_out,
            stream=stream,
            response_format=response_format,
            stop=stop_sequences,
            timeout_seconds=timeout_seconds,
        )
        return req, warnings

    @staticmethod
    def _map_adapter_exception(
        exc: Exception,
        target: ResolvedInferenceTarget,
        *,
        partial_content: str = "",
        usage: TokenUsage | None = None,
    ) -> GenerationError:
        """Translate low-level HTTP/adapter errors into normalized safe GenerationErrors."""
        raw_secret = target.config.api_key.get_secret_value() if target.config.api_key else None
        safe_msg = redact_llm_error(exc, secrets=[raw_secret])
        if isinstance(exc, LLMAuthError):
            return ProviderAuthFailedError(
                safe_msg,
                error_code="authentication_failed",
                status_code=401,
                partial_content=partial_content,
                usage=usage,
            )
        if isinstance(exc, LLMTimeoutError):
            return ProviderTimeoutError(
                safe_msg,
                error_code="inference_timeout",
                status_code=504,
                partial_content=partial_content,
                usage=usage,
            )
        if isinstance(exc, LLMConnectionSecurityError):
            return ProviderUnavailableError(
                safe_msg,
                error_code="endpoint_security_blocked",
                status_code=400,
                partial_content=partial_content,
                usage=usage,
            )
        if isinstance(exc, (LLMResponseError, LLMError)):
            lowered = safe_msg.lower()
            if any(
                marker in lowered
                for marker in ("connection failed", "connect", "refused", "unreachable", "dns")
            ):
                return ProviderUnavailableError(
                    safe_msg,
                    error_code="endpoint_unreachable",
                    status_code=502,
                    partial_content=partial_content,
                    usage=usage,
                )
            return ProviderResponseInvalidError(
                safe_msg,
                error_code="invalid_provider_response",
                status_code=502,
                partial_content=partial_content,
                usage=usage,
            )
        return GenerationError(
            safe_msg or "Unexpected inference execution failure.",
            error_code="inference_failed",
            status_code=500,
            partial_content=partial_content,
            usage=usage,
        )

    async def _record_health(
        self,
        target: ResolvedInferenceTarget,
        *,
        ok: bool,
        error_msg: str | None = None,
    ) -> None:
        if self._model_registry is None:
            return
        try:
            await self._model_registry.sync_connection(
                target.config,
                ok=ok,
                last_connected_at=datetime.now(timezone.utc) if ok else None,
                last_error=error_msg,
            )
        except Exception:
            logger.warning("Non-fatal failure updating ModelRegistry status after inference")

    async def execute(
        self,
        *,
        target: ResolvedInferenceTarget,
        context_package: ContextPackage,
        compiled_prompt: CompiledPrompt,
        context_validation: ContextValidationReport | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        timeout_seconds: float | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> tuple[LLMInferenceResponse, list[str]]:
        """Execute a single non-streaming completion with capability fallback and cancellation checks."""
        self._validate_preconditions(
            context_package, compiled_prompt, context_validation=context_validation
        )
        if cancel_event is not None and cancel_event.is_set():
            raise InferenceCancelledError()

        req, warnings = self._build_inference_request(
            target,
            compiled_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences,
            timeout_seconds=timeout_seconds,
            stream=False,
        )

        response: LLMInferenceResponse
        try:
            chat_task = asyncio.create_task(target.adapter.chat(req))
            if cancel_event is None:
                response = await chat_task
            else:
                cancel_task = asyncio.create_task(cancel_event.wait())
                try:
                    done, pending = await asyncio.wait(
                        {chat_task, cancel_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                    if cancel_task in done and cancel_event.is_set():
                        raise InferenceCancelledError()
                    response = await chat_task
                except BaseException:
                    chat_task.cancel()
                    cancel_task.cancel()
                    raise
        except InferenceCancelledError:
            raise
        except LLMResponseError as exc:
            # Graceful fallback when endpoint rejects optional response_format or stop parameter
            err_low = str(exc).lower()
            retried = False
            if req.response_format == "json_object" and "response_format" in err_low:
                warnings.append(
                    "Endpoint rejected response_format=json_object; retried with plain text JSON prompt."
                )
                req = req.model_copy(update={"response_format": "text"})
                retried = True
            elif req.stop and "stop" in err_low:
                warnings.append(
                    "Endpoint rejected stop parameter; retried without stop sequences."
                )
                req = req.model_copy(update={"stop": None})
                retried = True

            if not retried:
                mapped = self._map_adapter_exception(exc, target)
                await self._record_health(target, ok=False, error_msg=str(mapped))
                raise mapped from exc

            try:
                response = await target.adapter.chat(req)
            except Exception as retry_exc:
                mapped = self._map_adapter_exception(retry_exc, target)
                await self._record_health(target, ok=False, error_msg=str(mapped))
                raise mapped from retry_exc
        except Exception as exc:
            mapped = self._map_adapter_exception(exc, target)
            await self._record_health(target, ok=False, error_msg=str(mapped))
            raise mapped from exc

        await self._record_health(target, ok=True)
        return response, warnings

    async def execute_stream(
        self,
        *,
        target: ResolvedInferenceTarget,
        context_package: ContextPackage,
        compiled_prompt: CompiledPrompt,
        context_validation: ContextValidationReport | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        timeout_seconds: float | None = None,
        cancel_event: asyncio.Event | None = None,
        is_disconnected: Callable[[], Any] | None = None,
    ) -> AsyncIterator[ExecutorStreamEvent]:
        """Stream true incremental tokens with immediate cancellation and partial recovery."""
        started = time.monotonic()
        self._validate_preconditions(
            context_package, compiled_prompt, context_validation=context_validation
        )
        if cancel_event is not None and cancel_event.is_set():
            raise InferenceCancelledError()

        req, warnings = self._build_inference_request(
            target,
            compiled_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences,
            timeout_seconds=timeout_seconds,
            stream=True,
        )

        # If server explicitly prohibits streaming, fall back honestly without faking word chunks
        streaming_explicitly_off = (
            not target.capabilities.supports_streaming
            and target.capabilities.sources.streaming == CapabilitySource.VERIFIED
        ) or not target.config.streaming
        if streaming_explicitly_off:
            warnings.append(
                "Configured server/model has streaming disabled; executing single-response completion."
            )
            resp, exec_warnings = await self.execute(
                target=target,
                context_package=context_package,
                compiled_prompt=compiled_prompt,
                context_validation=context_validation,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stop_sequences=stop_sequences,
                timeout_seconds=timeout_seconds,
                cancel_event=cancel_event,
            )
            all_warnings = [*warnings, *exec_warnings]
            for warn in all_warnings:
                yield ExecutorStreamEvent(kind="warning", warning=warn)
            if resp.content:
                yield ExecutorStreamEvent(
                    kind="delta",
                    delta=resp.content,
                    accumulated=resp.content,
                    model_id=resp.model or target.model_id,
                    finish_reason=resp.finish_reason,
                    usage=resp.usage,
                )
            yield ExecutorStreamEvent(
                kind="done",
                accumulated=resp.content,
                finish_reason=resp.finish_reason,
                model_id=resp.model or target.model_id,
                usage=resp.usage,
                warnings=all_warnings,
                duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            )
            return

        for warn in warnings:
            yield ExecutorStreamEvent(kind="warning", warning=warn)

        accumulated_parts: list[str] = []
        last_usage: TokenUsage | None = None
        last_finish_reason: str | None = None
        last_model = target.model_id

        aiter = target.adapter.stream_chat(req)
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise InferenceCancelledError(
                        partial_content="".join(accumulated_parts),
                        usage=last_usage,
                    )
                if is_disconnected is not None:
                    disconn = is_disconnected()
                    if asyncio.iscoroutine(disconn):
                        disconn = await disconn
                    if disconn:
                        raise InferenceCancelledError(
                            "Client disconnected during generation.",
                            partial_content="".join(accumulated_parts),
                            usage=last_usage,
                        )

                next_task = asyncio.create_task(aiter.__anext__())
                if cancel_event is not None:
                    cancel_task = asyncio.create_task(cancel_event.wait())
                    done, pending = await asyncio.wait(
                        {next_task, cancel_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                    if cancel_task in done and cancel_event.is_set():
                        raise InferenceCancelledError(
                            partial_content="".join(accumulated_parts),
                            usage=last_usage,
                        )

                try:
                    chunk = await next_task
                except StopAsyncIteration:
                    break

                if chunk.model:
                    last_model = chunk.model
                if chunk.usage is not None:
                    last_usage = chunk.usage
                if chunk.finish_reason is not None:
                    last_finish_reason = chunk.finish_reason
                if chunk.delta:
                    accumulated_parts.append(chunk.delta)
                    yield ExecutorStreamEvent(
                        kind="delta",
                        delta=chunk.delta,
                        accumulated="".join(accumulated_parts),
                        finish_reason=chunk.finish_reason,
                        model_id=last_model,
                        usage=last_usage,
                    )

                if chunk.done:
                    break
        except InferenceCancelledError:
            raise
        except asyncio.CancelledError as exc:
            raise InferenceCancelledError(
                "Generation task cancelled.",
                partial_content="".join(accumulated_parts),
                usage=last_usage,
            ) from exc
        except Exception as exc:
            partial = "".join(accumulated_parts)
            mapped = self._map_adapter_exception(
                exc,
                target,
                partial_content=partial,
                usage=last_usage,
            )
            await self._record_health(target, ok=False, error_msg=str(mapped))
            raise mapped from exc
        finally:
            aclose = getattr(aiter, "aclose", None)
            if callable(aclose):
                with contextlib.suppress(Exception):
                    await aclose()

        await self._record_health(target, ok=True)
        yield ExecutorStreamEvent(
            kind="done",
            accumulated="".join(accumulated_parts),
            finish_reason=last_finish_reason,
            model_id=last_model,
            usage=last_usage,
            warnings=warnings,
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        )
