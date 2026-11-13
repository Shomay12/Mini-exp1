"""Provider-independent base interface and shared HTTP/JSON helpers for local LLMs."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from web_intelligence.app.core.logging import get_logger, register_secret
from web_intelligence.app.core.security import AsyncResolver
from web_intelligence.app.llm.security import (
    LLMConnectionSecurityError,
    redact_llm_error,
    validate_llm_endpoint_async,
)
from web_intelligence.app.research.future_interfaces import LocalModelAdapter
from web_intelligence.app.schemas.llm import (
    ChatMessage,
    DiscoveredModel,
    LLMConnectionConfig,
    LLMInferenceRequest,
    LLMInferenceResponse,
    LLMStreamChunk,
    ModelCapabilities,
)

logger = get_logger("llm.adapter")


class LLMError(RuntimeError):
    """Base error raised by Universal Local LLM adapters."""


class LLMAuthError(LLMError):
    """Raised when an inference endpoint rejects credentials (401/403)."""


class LLMTimeoutError(LLMError):
    """Raised when an inference endpoint exceeds configured request timeout."""


class LLMResponseError(LLMError):
    """Raised when an inference endpoint returns a 4xx/5xx or malformed payload."""


class LLMUnsupportedFeatureError(LLMError):
    """Raised when a requested capability is not supported by the target adapter/model."""


class LLMCancelledError(LLMError):
    """Raised when an active inference request is cancelled before completion."""


def extract_json_payload(raw_text: str) -> dict[str, Any] | list[Any] | None:
    """Best-effort parser extracting a JSON object or array from model output."""
    text = raw_text.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass

    # Attempt bounded slice between outermost braces/brackets when prose precedes JSON
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if 0 <= start < end:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, (dict, list)):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


class UniversalLocalModelAdapter(LocalModelAdapter, ABC):
    """Extended provider-independent interface for user-hosted HTTP inference servers."""

    provider_id: str = "base"

    def __init__(
        self,
        config: LLMConnectionConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: AsyncResolver | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._resolver = resolver
        raw_key = self._raw_api_key()
        if raw_key:
            register_secret(raw_key)

    def _raw_api_key(self) -> str | None:
        if self.config.api_key is None:
            return None
        val = self.config.api_key.get_secret_value().strip()
        return val or None

    def _safe_error(self, exc: Exception | str) -> str:
        return redact_llm_error(exc, secrets=[self._raw_api_key()])

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "LocalWebIntelligence/0.1",
        }
        api_key = self._raw_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _open_client(self, *, timeout_override: float | None = None) -> httpx.AsyncClient:
        """Validate endpoint trust boundary and build an isolated AsyncClient."""
        await validate_llm_endpoint_async(
            self.config.base_url,
            allow_remote_endpoint=self.config.allow_remote_endpoint,
            approved_by_user=self.config.approved_by_user,
            api_endpoint=self.config.api_endpoint,
            resolver=self._resolver,
        )
        timeout = httpx.Timeout(timeout_override or self.config.request_timeout_seconds)
        return httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            transport=self._transport,
            headers=self._build_headers(),
        )

    def _check_response_status(self, response: httpx.Response, *, action: str) -> None:
        """Map HTTP redirects and errors into redacted domain exceptions."""
        if 300 <= response.status_code < 400:
            raise LLMConnectionSecurityError(
                f"{action} blocked: HTTP redirects ({response.status_code}) are prohibited on LLM endpoints."
            )
        if response.status_code in {401, 403}:
            raise LLMAuthError(
                f"{action} unauthorized (HTTP {response.status_code}). Check optional API key."
            )
        if response.status_code >= 400:
            detail = self._extract_error_detail(response)
            raise LLMResponseError(
                f"{action} failed with HTTP {response.status_code}: {detail}"
            )

    def _extract_error_detail(self, response: httpx.Response) -> str:
        raw_body = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                err = payload.get("error")
                if isinstance(err, dict):
                    raw_body = str(err.get("message") or err.get("code") or err)
                elif err is not None:
                    raw_body = str(err)
                elif "detail" in payload:
                    raw_body = str(payload["detail"])
        except Exception:
            pass
        if not raw_body:
            try:
                raw_body = response.text[:240]
            except Exception:
                raw_body = "empty error body"
        return self._safe_error(raw_body)

    def capabilities(self) -> ModelCapabilities:
        """Return currently known capabilities without inventing unverified flags."""
        return self.config.effective_capabilities()

    @abstractmethod
    async def test_connection(
        self,
    ) -> tuple[bool, str, list[DiscoveredModel], ModelCapabilities]:
        """Verify connectivity and return (ok, message, models, capabilities)."""

    @abstractmethod
    async def list_models(self) -> tuple[list[DiscoveredModel], bool, str]:
        """Discover models from the server, returning (models, discovery_supported, message)."""

    @abstractmethod
    async def inspect_model_capabilities(
        self, model_name: str | None = None
    ) -> ModelCapabilities:
        """Fetch verified metadata when supported, preserving configured fallback assumptions."""

    @abstractmethod
    async def chat(self, request: LLMInferenceRequest) -> LLMInferenceResponse:
        """Execute a non-streaming chat completion."""

    @abstractmethod
    async def stream_chat(
        self, request: LLMInferenceRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream chat completion deltas."""

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Phase-2 LocalModelAdapter single-prompt entrypoint."""
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))
        req = LLMInferenceRequest(
            messages=messages,
            model=self.config.model or None,
            max_tokens=max_tokens or self.config.max_output_tokens,
            stream=False,
        )
        res = await self.chat(req)
        return res.content

    async def generate_json(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.1,
    ) -> dict[str, Any] | list[Any] | None:
        """Execute structured JSON generation, falling back cleanly if server lacks json_object."""
        caps = self.capabilities()
        req = LLMInferenceRequest(
            messages=messages,
            model=self.config.model or None,
            temperature=temperature,
            max_tokens=max_tokens or self.config.max_output_tokens,
            stream=False,
            response_format="json_object" if caps.supports_json_mode else "text",
        )
        try:
            response = await self.chat(req)
        except LLMResponseError:
            if req.response_format != "json_object":
                raise
            # Graceful downgrade if endpoint rejected response_format parameter
            fallback_req = req.model_copy(update={"response_format": "text"})
            response = await self.chat(fallback_req)
        return response.parsed_json or extract_json_payload(response.content)

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, int((time.monotonic() - started_at) * 1000))

    @staticmethod
    def _normalize_stop_sequences(stop: list[str] | None) -> list[str]:
        if not stop:
            return []
        normalized: list[str] = []
        for item in stop:
            if isinstance(item, str) and item != "" and item not in normalized:
                normalized.append(item)
        return normalized
