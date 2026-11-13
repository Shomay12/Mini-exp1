"""Adapter for OpenAI-compatible HTTP inference servers (vLLM, LM Studio, llama.cpp, LocalAI)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from web_intelligence.app.core.logging import get_logger
from web_intelligence.app.llm.base import (
    LLMAuthError,
    LLMResponseError,
    LLMTimeoutError,
    UniversalLocalModelAdapter,
    extract_json_payload,
)
from web_intelligence.app.llm.security import LLMConnectionSecurityError, normalize_base_url
from web_intelligence.app.schemas.llm import (
    CapabilityProvenance,
    CapabilitySource,
    DiscoveredModel,
    LLMInferenceRequest,
    LLMInferenceResponse,
    LLMProviderType,
    LLMStreamChunk,
    ModelCapabilities,
    TokenUsage,
    ToolCall,
)

logger = get_logger("llm.openai_compatible")


def _extract_positive_int(mapping: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        val = mapping.get(key)
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return val
    return None


def _coerce_message_content(content: Any) -> str:
    """Normalize string or content-part lists returned by OpenAI-compatible backends."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_val = item.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "".join(parts)
    return ""


class OpenAICompatibleAdapter(UniversalLocalModelAdapter):
    """Adapter speaking OpenAI-compatible REST endpoints without assuming full OpenAI parity."""

    provider_id = LLMProviderType.OPENAI_COMPATIBLE.value

    def _origin_url(self) -> str:
        clean = normalize_base_url(self.config.base_url)
        parsed = urlparse(clean)
        return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))

    def _v1_url(self, suffix: str) -> str:
        clean = normalize_base_url(self.config.base_url)
        if clean.endswith("/v1"):
            return f"{clean}{suffix}"
        return f"{clean}/v1{suffix}"

    def _chat_url(self) -> str:
        if self.config.api_endpoint:
            override = self.config.api_endpoint
            clean = normalize_base_url(self.config.base_url)
            if override.startswith("/v1/") and clean.endswith("/v1"):
                return f"{self._origin_url()}{override}"
            return f"{clean}{override}"
        return self._v1_url("/chat/completions")

    def _build_capabilities_from_model_item(
        self,
        model_name: str,
        item: dict[str, Any] | None = None,
    ) -> ModelCapabilities:
        """Construct capabilities distinguishing verified metadata from configured assumptions."""
        provenance = CapabilityProvenance(
            context_length=CapabilitySource.CONFIGURED,
            max_output_tokens=CapabilitySource.CONFIGURED,
            streaming=CapabilitySource.CONFIGURED,
            tool_calling=CapabilitySource.UNKNOWN,
            json_mode=CapabilitySource.UNKNOWN,
            vision=CapabilitySource.UNKNOWN,
        )
        max_context = self.config.context_length
        max_output = self.config.max_output_tokens
        supports_streaming = self.config.streaming
        supports_tools = False
        supports_json = False
        supports_vision = False
        metadata: dict[str, Any] = {}

        if isinstance(item, dict):
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            detected_ctx = _extract_positive_int(
                item, "max_model_len", "context_length", "max_context_length"
            ) or _extract_positive_int(meta, "n_ctx_train", "context_length")
            if detected_ctx is not None and detected_ctx >= 512:
                max_context = detected_ctx
                provenance.context_length = CapabilitySource.VERIFIED
                metadata["reported_context_length"] = detected_ctx

            detected_out = _extract_positive_int(item, "max_output_tokens", "max_tokens")
            if detected_out is not None and detected_out >= 64:
                max_output = detected_out
                provenance.max_output_tokens = CapabilitySource.VERIFIED
                metadata["reported_max_output_tokens"] = detected_out

            if isinstance(item.get("owned_by"), str):
                metadata["owned_by"] = item["owned_by"]

            # Only set capability flags as VERIFIED if the server explicitly reports them
            raw_caps = item.get("capabilities")
            if isinstance(raw_caps, dict):
                if isinstance(raw_caps.get("tools"), bool):
                    supports_tools = raw_caps["tools"]
                    provenance.tool_calling = CapabilitySource.VERIFIED
                if isinstance(raw_caps.get("vision"), bool):
                    supports_vision = raw_caps["vision"]
                    provenance.vision = CapabilitySource.VERIFIED
                if isinstance(raw_caps.get("json_mode"), bool):
                    supports_json = raw_caps["json_mode"]
                    provenance.json_mode = CapabilitySource.VERIFIED
                if isinstance(raw_caps.get("streaming"), bool):
                    supports_streaming = raw_caps["streaming"]
                    provenance.streaming = CapabilitySource.VERIFIED

            supported_params = item.get("supported_parameters")
            if isinstance(supported_params, list):
                params_set = {str(p) for p in supported_params}
                supports_tools = "tools" in params_set
                provenance.tool_calling = CapabilitySource.VERIFIED
                supports_json = "response_format" in params_set or "structured_outputs" in params_set
                provenance.json_mode = CapabilitySource.VERIFIED
                metadata["supported_parameters"] = sorted(params_set)

            if isinstance(item.get("supports_tool_calling"), bool):
                supports_tools = item["supports_tool_calling"]
                provenance.tool_calling = CapabilitySource.VERIFIED
            if isinstance(item.get("supports_vision"), bool):
                supports_vision = item["supports_vision"]
                provenance.vision = CapabilitySource.VERIFIED

        return ModelCapabilities(
            model_name=model_name or self.config.model or "unselected",
            backend=self.provider_id,
            max_context_tokens=max_context,
            recommended_output_tokens=min(max_output, max_context - 1),
            supports_streaming=supports_streaming,
            supports_tool_calling=supports_tools,
            supports_json_mode=supports_json,
            supports_vision=supports_vision,
            sources=provenance,
            available_metadata=metadata,
        )

    async def _fetch_models_payload(self, client: httpx.AsyncClient) -> tuple[list[dict[str, Any]], bool, str]:
        models_url = self._v1_url("/models")
        response = await client.get(models_url)
        if response.status_code in {404, 405, 501}:
            return (
                [],
                False,
                "Server does not expose GET /v1/models; enter the model identifier manually.",
            )
        self._check_response_status(response, action="Model discovery")
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMResponseError("Model discovery returned invalid JSON.") from exc

        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            items = [item for item in payload["data"] if isinstance(item, dict)]
        elif isinstance(payload, list):
            items = [item for item in payload if isinstance(item, dict)]
        else:
            raise LLMResponseError("Model discovery payload missing 'data' model list.")
        return items, True, f"Discovered {len(items)} model(s) from /v1/models."

    async def list_models(self) -> tuple[list[DiscoveredModel], bool, str]:
        try:
            async with await self._open_client() as client:
                items, supported, message = await self._fetch_models_payload(client)
        except (LLMAuthError, LLMResponseError, LLMConnectionSecurityError):
            raise
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Timed out querying /v1/models.") from exc
        except httpx.RequestError as exc:
            raise LLMResponseError(
                f"Connection failed while querying /v1/models: {self._safe_error(exc)}"
            ) from exc

        models: list[DiscoveredModel] = []
        for item in items:
            model_id = str(item.get("id") or item.get("model") or "").strip()
            if not model_id:
                continue
            caps = self._build_capabilities_from_model_item(model_id, item)
            safe_raw = {
                k: v
                for k, v in item.items()
                if k in {"id", "object", "owned_by", "max_model_len", "context_length", "capabilities", "supported_parameters"}
            }
            models.append(
                DiscoveredModel(
                    id=model_id,
                    name=model_id,
                    provider=self.provider_id,
                    owned_by=item.get("owned_by") if isinstance(item.get("owned_by"), str) else None,
                    context_length=(
                        caps.max_context_tokens
                        if caps.sources.context_length == CapabilitySource.VERIFIED
                        else None
                    ),
                    capabilities=caps,
                    raw_metadata=safe_raw,
                )
            )
        return models, supported, message

    async def inspect_model_capabilities(
        self, model_name: str | None = None
    ) -> ModelCapabilities:
        target = (model_name or self.config.model).strip()
        try:
            models, supported, _ = await self.list_models()
        except Exception as exc:
            logger.warning(
                "Model capability inspection fell back to configured assumptions",
                extra={"extra_data": {"error": self._safe_error(exc)}},
            )
            return self._build_capabilities_from_model_item(target, None)

        if not supported or not models:
            return self._build_capabilities_from_model_item(target, None)

        matched = next((m for m in models if m.id == target), None)
        if matched is None and not target:
            matched = models[0]
        if matched and matched.capabilities:
            return matched.capabilities
        return self._build_capabilities_from_model_item(target, None)

    async def test_connection(
        self,
    ) -> tuple[bool, str, list[DiscoveredModel], ModelCapabilities]:
        started = time.monotonic()
        try:
            models, supported, discovery_msg = await self.list_models()
        except (LLMAuthError, LLMConnectionSecurityError, LLMTimeoutError, LLMResponseError) as exc:
            caps = self._build_capabilities_from_model_item(self.config.model, None)
            return False, self._safe_error(exc), [], caps

        if not supported:
            # Verify base endpoint/health reachability when /v1/models is unsupported
            try:
                async with await self._open_client() as client:
                    probe = await client.get(f"{self._origin_url()}/health")
                    if probe.status_code >= 400 and probe.status_code != 404:
                        self._check_response_status(probe, action="Health check")
            except (LLMAuthError, LLMConnectionSecurityError, LLMResponseError) as exc:
                return (
                    False,
                    self._safe_error(exc),
                    [],
                    self._build_capabilities_from_model_item(self.config.model, None),
                )
            except httpx.TimeoutException:
                return (
                    False,
                    "Timed out checking server health endpoint.",
                    [],
                    self._build_capabilities_from_model_item(self.config.model, None),
                )
            except httpx.RequestError as exc:
                return (
                    False,
                    f"Connection failed: {self._safe_error(exc)}",
                    [],
                    self._build_capabilities_from_model_item(self.config.model, None),
                )

            return (
                True,
                discovery_msg,
                [],
                self._build_capabilities_from_model_item(self.config.model, None),
            )

        selected_name = self.config.model or (models[0].id if models else "")
        matched = next((m for m in models if m.id == selected_name), None)
        caps = (
            matched.capabilities
            if matched and matched.capabilities is not None
            else self._build_capabilities_from_model_item(selected_name, None)
        )
        elapsed = self._elapsed_ms(started)
        return (
            True,
            f"Connected ({elapsed} ms). {discovery_msg}",
            models,
            caps,
        )

    def _build_chat_payload(
        self, request: LLMInferenceRequest, *, stream: bool
    ) -> dict[str, Any]:
        model = (request.model or self.config.model).strip()
        if not model:
            raise LLMResponseError("No model identifier is configured for this connection.")

        serialized_messages: list[dict[str, Any]] = []
        for msg in request.messages:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.name:
                entry["name"] = msg.name
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            serialized_messages.append(entry)

        payload: dict[str, Any] = {
            "model": model,
            "messages": serialized_messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or self.config.max_output_tokens,
            "stream": stream,
        }
        if request.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
        return payload

    async def chat(self, request: LLMInferenceRequest) -> LLMInferenceResponse:
        started = time.monotonic()
        payload = self._build_chat_payload(request, stream=False)
        chat_url = self._chat_url()

        try:
            async with await self._open_client() as client:
                response = await client.post(chat_url, json=payload)
                self._check_response_status(response, action="Chat completion")
                data = response.json()
        except (LLMAuthError, LLMConnectionSecurityError, LLMResponseError):
            raise
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Inference request timed out.") from exc
        except httpx.RequestError as exc:
            raise LLMResponseError(
                f"Inference request failed: {self._safe_error(exc)}"
            ) from exc
        except ValueError as exc:
            raise LLMResponseError("Inference endpoint returned malformed JSON.") from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("Inference response missing non-empty 'choices' list.")

        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = _coerce_message_content(message.get("content"))

        parsed_tool_calls: list[ToolCall] = []
        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, list):
            for call in raw_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = str(fn.get("name") or call.get("name") or "").strip()
                if not name:
                    continue
                raw_args = fn.get("arguments", {})
                args_val: dict[str, Any] | str = raw_args if isinstance(raw_args, dict) else str(raw_args or "{}")
                if isinstance(raw_args, str):
                    try:
                        decoded = json.loads(raw_args)
                        if isinstance(decoded, dict):
                            args_val = decoded
                    except json.JSONDecodeError:
                        args_val = raw_args
                parsed_tool_calls.append(
                    ToolCall(
                        id=str(call["id"]) if call.get("id") is not None else None,
                        name=name,
                        arguments=args_val,
                    )
                )

        usage_obj: TokenUsage | None = None
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            usage_obj = TokenUsage(
                prompt_tokens=raw_usage.get("prompt_tokens"),
                completion_tokens=raw_usage.get("completion_tokens"),
                total_tokens=raw_usage.get("total_tokens"),
            )

        parsed_json = (
            extract_json_payload(content)
            if request.response_format == "json_object"
            else None
        )
        return LLMInferenceResponse(
            content=content,
            model=str(data.get("model") or payload["model"]),
            finish_reason=(
                str(first["finish_reason"])
                if first.get("finish_reason") is not None
                else None
            ),
            tool_calls=parsed_tool_calls,
            parsed_json=parsed_json,
            usage=usage_obj,
            latency_ms=self._elapsed_ms(started),
        )

    async def stream_chat(
        self, request: LLMInferenceRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        payload = self._build_chat_payload(request, stream=True)
        chat_url = self._chat_url()

        try:
            async with await self._open_client() as client:
                async with client.stream("POST", chat_url, json=payload) as response:
                    self._check_response_status(response, action="Streaming chat completion")
                    emitted_done = False
                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            emitted_done = True
                            yield LLMStreamChunk(done=True, model=payload["model"])
                            break
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Skipping malformed SSE JSON line from OpenAI-compatible server"
                            )
                            continue
                        if not isinstance(event, dict):
                            continue

                        choices = event.get("choices")
                        delta_text = ""
                        finish_reason: str | None = None
                        if isinstance(choices, list) and choices:
                            first = choices[0] if isinstance(choices[0], dict) else {}
                            delta_obj = (
                                first.get("delta") if isinstance(first.get("delta"), dict) else {}
                            )
                            delta_text = _coerce_message_content(delta_obj.get("content"))
                            if first.get("finish_reason") is not None:
                                finish_reason = str(first["finish_reason"])

                        usage_obj: TokenUsage | None = None
                        raw_usage = event.get("usage")
                        if isinstance(raw_usage, dict):
                            usage_obj = TokenUsage(
                                prompt_tokens=raw_usage.get("prompt_tokens"),
                                completion_tokens=raw_usage.get("completion_tokens"),
                                total_tokens=raw_usage.get("total_tokens"),
                            )

                        is_done = finish_reason is not None
                        if is_done:
                            emitted_done = True
                        if delta_text or is_done or usage_obj is not None:
                            yield LLMStreamChunk(
                                delta=delta_text,
                                done=is_done,
                                finish_reason=finish_reason,
                                model=str(event.get("model") or payload["model"]),
                                usage=usage_obj,
                            )
                    if not emitted_done:
                        yield LLMStreamChunk(done=True, model=payload["model"])
        except (LLMAuthError, LLMConnectionSecurityError, LLMResponseError):
            raise
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Streaming inference request timed out.") from exc
        except httpx.RequestError as exc:
            raise LLMResponseError(
                f"Streaming inference request failed: {self._safe_error(exc)}"
            ) from exc
