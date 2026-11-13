"""Extensible adapter registry and presets for user-hosted local LLM servers."""

from __future__ import annotations

import httpx

from web_intelligence.app.core.security import AsyncResolver
from web_intelligence.app.llm.base import UniversalLocalModelAdapter
from web_intelligence.app.llm.ollama import OllamaAdapter
from web_intelligence.app.llm.openai_compatible import OpenAICompatibleAdapter
from web_intelligence.app.schemas.llm import (
    LLMConnectionConfig,
    LLMProviderType,
    ProviderPreset,
)

PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        id="ollama",
        label="Ollama",
        provider=LLMProviderType.OLLAMA.value,
        default_base_url="http://localhost:11434",
        notes="Native Ollama REST API (/api/tags, /api/show, /api/chat).",
    ),
    ProviderPreset(
        id="lm-studio",
        label="LM Studio",
        provider=LLMProviderType.OPENAI_COMPATIBLE.value,
        default_base_url="http://localhost:1234/v1",
        notes="LM Studio local OpenAI-compatible server.",
    ),
    ProviderPreset(
        id="llamacpp",
        label="llama.cpp",
        provider=LLMProviderType.OPENAI_COMPATIBLE.value,
        default_base_url="http://localhost:8080/v1",
        notes="llama-server OpenAI-compatible HTTP endpoint.",
    ),
    ProviderPreset(
        id="vllm",
        label="vLLM",
        provider=LLMProviderType.OPENAI_COMPATIBLE.value,
        default_base_url="http://localhost:8000/v1",
        notes="vLLM OpenAI-compatible server exposing max_model_len.",
    ),
    ProviderPreset(
        id="localai",
        label="LocalAI",
        provider=LLMProviderType.OPENAI_COMPATIBLE.value,
        default_base_url="http://localhost:8080/v1",
        notes="LocalAI OpenAI-compatible HTTP server.",
    ),
    ProviderPreset(
        id="custom",
        label="Custom OpenAI-Compatible",
        provider=LLMProviderType.OPENAI_COMPATIBLE.value,
        default_base_url="http://localhost:8000/v1",
        notes="Any local or explicitly approved OpenAI-compatible API server.",
    ),
)


class UnsupportedProviderError(ValueError):
    """Raised when requesting an unregistered LLM adapter provider type."""


class AdapterRegistry:
    """Maps provider identifiers to UniversalLocalModelAdapter implementations."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[UniversalLocalModelAdapter]] = {}
        self.register(LLMProviderType.OPENAI_COMPATIBLE.value, OpenAICompatibleAdapter)
        self.register(LLMProviderType.OLLAMA.value, OllamaAdapter)

    def register(
        self, provider: str, adapter_cls: type[UniversalLocalModelAdapter]
    ) -> None:
        """Register or replace an adapter implementation for a provider identifier."""
        key = provider.strip().lower()
        if not key:
            raise ValueError("Provider identifier must be non-empty.")
        self._adapters[key] = adapter_cls

    def supports(self, provider: str) -> bool:
        """Return True if an adapter class is registered for `provider`."""
        return provider.strip().lower() in self._adapters

    def supported_providers(self) -> list[str]:
        """Return registered provider identifiers in deterministic order."""
        return sorted(self._adapters.keys())

    @staticmethod
    def presets() -> list[ProviderPreset]:
        """Return built-in server presets for UI configuration assistance."""
        return list(PROVIDER_PRESETS)

    def create(
        self,
        config: LLMConnectionConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: AsyncResolver | None = None,
    ) -> UniversalLocalModelAdapter:
        """Instantiate an adapter bound to `config`."""
        key = config.provider.strip().lower()
        adapter_cls = self._adapters.get(key)
        if adapter_cls is None:
            raise UnsupportedProviderError(
                f"Unsupported LLM provider '{config.provider}'. "
                f"Registered providers: {', '.join(self.supported_providers())}"
            )
        return adapter_cls(config, transport=transport, resolver=resolver)
