"""Adapter for native Ollama HTTP inference endpoints (/api/tags, /api/show, /api/chat)."""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

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

logger = get_logger("llm.ollama")

_NUM_CTX_RE = re.compile(r"^\s*num_ctx\s+(\d+)\s*$", re.MULTILINE)


class OllamaAdapter(UniversalLocalModelAdapter):
    """Native Ollama REST adapter with verified capability extraction via /api/show."""

    provider_id = LLMProviderType.OLLAMA.value

    def _native_root(self) -> str:
        clean = normalize_base_url(self.config.base_url)
        for suffix in ("/v1", "/api"):
            if clean.endswith(suffix):
                clean = clean[: -len(suffix)]
        return clean

    def _api_url(self, suffix: str) -> str:
        return f"{self._native_root()}{suffix}"

    def _chat_url(self) -> str:
        if self.config.api_endpoint:
            return f"{self._native_root()}{self.config.api_endpoint}"
        return self._api_url("/api/chat")

    def _capabilities_from_show_payload(
        self,
        model_name: str,
        show_data: dict[str, Any] | None = None,
        tag_item: dict[str, Any] | None = None,
    ) -> ModelCapabilities:
        provenance = CapabilityProvenance(
            context_length=CapabilitySource.CONFIGURED,
            max_output_tokens=CapabilitySource.CONFIGURED,
            streaming=CapabilitySource.VERIFIED,
            tool_calling=CapabilitySource.UNKNOWN,
            json_mode=CapabilitySource.VERIFIED,  # Ollama /api/chat natively implements format="json"
            vision=CapabilitySource.UNKNOWN,
        )
        max_context = self.config.context_length
        max_output = self.config.max_output_tokens
        supports_tools = False
        supports_vision = False
        metadata: dict[str, Any] = {}

        if isinstance(tag_item, dict) and isinstance(tag_item.get("details"), dict):
            details = tag_item["details"]
            for key in ("family", "parameter_size", "quantization_level", "format"):
                if details.get(key):
                    metadata[key] = details[key]
            if isinstance(details.get("families"), list):
                metadata["families"] = details["families"]
                if any("clip" in str(f).lower() or "vision" in str(f).lower() for f in details["families"]):
                    supports_vision = True
                    provenance.vision = CapabilitySource.VERIFIED

        if isinstance(show_data, dict):
            details = show_data.get("details")
            if isinstance(details, dict):
                for key in ("family", "parameter_size", "quantization_level", "format"):
                    if details.get(key):
                        metadata[key] = details[key]
                if isinstance(details.get("families"), list):
                    metadata["families"] = details["families"]
                    if any("clip" in str(f).lower() or "vision" in str(f).lower() for f in details["families"]):
                        supports_vision = True
                        provenance.vision = CapabilitySource.VERIFIED

            model_info = show_data.get("model_info")
            verified_ctx: int | None = None
            if isinstance(model_info, dict):
                for key, value in model_info.items():
                    if key.endswith(".context_length") and isinstance(value, int) and value >= 512:
                        verified_ctx = value
                    if ".vision." in key:
                        supports_vision = True
                        provenance.vision = CapabilitySource.VERIFIED
                if "general.architecture" in model_info:
                    metadata["architecture"] = model_info["general.architecture"]

            # Explicit num_ctx in Modelfile parameters overrides architectural max
            params_text = show_data.get("parameters")
            if isinstance(params_text, str):
                match = _NUM_CTX_RE.search(params_text)
                if match:
                    modelfile_ctx = int(match.group(1))
                    if modelfile_ctx >= 512:
                        verified_ctx = modelfile_ctx
                        metadata["modelfile_num_ctx"] = modelfile_ctx

            if verified_ctx is not None:
                max_context = verified_ctx
                provenance.context_length = CapabilitySource.VERIFIED
                metadata["reported_context_length"] = verified_ctx

            cap_list = show_data.get("capabilities")
            if isinstance(cap_list, list):
                normalized_caps = [str(c).lower() for c in cap_list]
                metadata["capabilities"] = normalized_caps
                supports_tools = "tools" in normalized_caps
                provenance.tool_calling = CapabilitySource.VERIFIED
                supports_vision = "vision" in normalized_caps or supports_vision
                provenance.vision = CapabilitySource.VERIFIED
            else:
                template = show_data.get("template")
                if isinstance(template, str) and ".Tools" in template:
                    supports_tools = True
                    provenance.tool_calling = CapabilitySource.VERIFIED

        return ModelCapabilities(
            model_name=model_name or self.config.model or "unselected",
            backend=self.provider_id,
            max_context_tokens=max_context,
            recommended_output_tokens=min(max_output, max_context - 1),
            supports_streaming=True,
            supports_tool_calling=supports_tools,
            supports_json_mode=True,
            supports_vision=supports_vision,
            sources=provenance,
            available_metadata=metadata,
        )

    async def list_models(self) -> tuple[list[DiscoveredModel], bool, str]:
        tags_url = self._api_url("/api/tags")
        try:
            async with await self._open_client() as client:
                response = await client.get(tags_url)
                if response.status_code in {404, 405, 501}:
                    return (
                        [],
                        False,
                        "Ollama /api/tags endpoint unavailable; enter a model name manually.",
                    )
                self._check_response_status(response, action="Ollama model discovery")
                data = response.json()
        except (LLMAuthError, LLMConnectionSecurityError, LLMResponseError):
            raise
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Timed out querying Ollama /api/tags.") from exc
        except httpx.RequestError as exc:
            raise LLMResponseError(
                f"Connection failed querying Ollama /api/tags: {self._safe_error(exc)}"
            ) from exc
        except ValueError as exc:
            raise LLMResponseError("Ollama /api/tags returned invalid JSON.") from exc

        raw_models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(raw_models, list):
            raise LLMResponseError("Ollama /api/tags response missing 'models' list.")

        models: list[DiscoveredModel] = []
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if not name:
                continue
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            caps = self._capabilities_from_show_payload(name, None, item)
            safe_raw = {
                k: v
                for k, v in item.items()
                if k in {"name", "model", "size", "digest", "modified_at", "details"}
            }
            models.append(
                DiscoveredModel(
                    id=name,
                    name=name,
                    provider=self.provider_id,
                    family=details.get("family"),
                    parameter_size=details.get("parameter_size"),
                    quantization_level=details.get("quantization_level"),
                    capabilities=caps,
                    raw_metadata=safe_raw,
                )
            )
        return models, True, f"Discovered {len(models)} Ollama model(s) from /api/tags."

    async def inspect_model_capabilities(
        self, model_name: str | None = None
    ) -> ModelCapabilities:
        target = (model_name or self.config.model).strip()
        if not target:
            return self._capabilities_from_show_payload("unselected", None, None)

        show_url = self._api_url("/api/show")
        try:
            async with await self._open_client() as client:
                response = await client.post(show_url, json={"model": target})
                if response.status_code >= 400:
                    return self._capabilities_from_show_payload(target, None, None)
                show_data = response.json()
        except Exception as exc:
            logger.warning(
                "Ollama /api/show failed; using configured defaults",
                extra={"extra_data": {"error": self._safe_error(exc), "model": target}},
            )
            return self._capabilities_from_show_payload(target, None, None)

        if not isinstance(show_data, dict):
            return self._capabilities_from_show_payload(target, None, None)
        return self._capabilities_from_show_payload(target, show_data, None)

    async def test_connection(
        self,
    ) -> tuple[bool, str, list[DiscoveredModel], ModelCapabilities]:
        started = time.monotonic()
        try:
            models, supported, discovery_msg = await self.list_models()
        except (LLMAuthError, LLMConnectionSecurityError, LLMTimeoutError, LLMResponseError) as exc:
            return (
                False,
                self._safe_error(exc),
                [],
                self._capabilities_from_show_payload(self.config.model, None, None),
            )

        selected_name = self.config.model or (models[0].id if models else "")
        caps = (
            await self.inspect_model_capabilities(selected_name)
            if selected_name and supported
            else self._capabilities_from_show_payload(selected_name, None, None)
        )
        elapsed = self._elapsed_ms(started)
        return (
            True,
            f"Connected to Ollama ({elapsed} ms). {discovery_msg}",
            models,
            caps,
        )

    def _build_chat_payload(
        self, request: LLMInferenceRequest, *, stream: bool
    ) -> dict[str, Any]:
        model = (request.model or self.config.model).strip()
        if not model:
            raise LLMResponseError("No Ollama model identifier is configured.")

        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            messages.append(entry)

        caps = self.capabilities()
        effective_ctx = (
            min(self.config.context_length, caps.max_context_tokens)
            if caps.sources.context_length == CapabilitySource.VERIFIED
            else self.config.context_length
        )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens or self.config.max_output_tokens,
                "num_ctx": effective_ctx,
            },
        }
        if request.response_format == "json_object":
            payload["format"] = "json"
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
                self._check_response_status(response, action="Ollama chat completion")
                data = response.json()
        except (LLMAuthError, LLMConnectionSecurityError, LLMResponseError):
            raise
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Ollama inference request timed out.") from exc
        except httpx.RequestError as exc:
            raise LLMResponseError(
                f"Ollama inference request failed: {self._safe_error(exc)}"
            ) from exc
        except ValueError as exc:
            raise LLMResponseError("Ollama returned invalid JSON.") from exc

        if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
            raise LLMResponseError("Ollama response missing 'message' object.")

        msg = data["message"]
        content = str(msg.get("content") or "")

        parsed_tool_calls: list[ToolCall] = []
        raw_calls = msg.get("tool_calls")
        if isinstance(raw_calls, list):
            for idx, call in enumerate(raw_calls):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = str(fn.get("name") or "").strip()
                if not name:
                    continue
                args = fn.get("arguments", {})
                parsed_tool_calls.append(
                    ToolCall(
                        id=str(call.get("id") or f"call_{idx}"),
                        name=name,
                        arguments=args if isinstance(args, (dict, str)) else {},
                    )
                )

        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        usage_obj: TokenUsage | None = None
        if prompt_tokens is not None or completion_tokens is not None:
            p_count = int(prompt_tokens or 0)
            c_count = int(completion_tokens or 0)
            usage_obj = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=p_count + c_count,
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
                str(data["done_reason"])
                if data.get("done_reason") is not None
                else ("stop" if data.get("done") else None)
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
                    self._check_response_status(
                        response, action="Ollama streaming chat completion"
                    )
                    emitted_done = False
                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("Skipping malformed NDJSON line from Ollama")
                            continue
                        if not isinstance(event, dict):
                            continue

                        msg = (
                            event.get("message")
                            if isinstance(event.get("message"), dict)
                            else {}
                        )
                        delta_text = str(msg.get("content") or "")
                        is_done = bool(event.get("done"))
                        finish_reason = (
                            str(event["done_reason"])
                            if event.get("done_reason") is not None
                            else ("stop" if is_done else None)
                        )
                        usage_obj: TokenUsage | None = None
                        if is_done:
                            emitted_done = True
                            p_tokens = event.get("prompt_eval_count")
                            c_tokens = event.get("eval_count")
                            if p_tokens is not None or c_tokens is not None:
                                usage_obj = TokenUsage(
                                    prompt_tokens=p_tokens,
                                    completion_tokens=c_tokens,
                                    total_tokens=int(p_tokens or 0) + int(c_tokens or 0),
                                )
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
            raise LLMTimeoutError("Ollama streaming request timed out.") from exc
        except httpx.RequestError as exc:
            raise LLMResponseError(
                f"Ollama streaming request failed: {self._safe_error(exc)}"
            ) from exc
