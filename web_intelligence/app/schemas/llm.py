"""Schemas for universal local LLM connections, discovery, capabilities, and inference."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator


class LLMProviderType(str, Enum):
    """Supported local/self-hosted inference server adapter types."""

    OPENAI_COMPATIBLE = "openai-compatible"
    OLLAMA = "ollama"


class ConnectionStatus(str, Enum):
    """Health/connectivity state of a configured LLM endpoint."""

    DISCONNECTED = "disconnected"
    CONFIGURED = "configured"
    CONNECTED = "connected"
    ERROR = "error"


class CapabilitySource(str, Enum):
    """Provenance distinguishing observed metadata from user-supplied assumptions."""

    VERIFIED = "verified"
    CONFIGURED = "configured"
    UNKNOWN = "unknown"


class CapabilityProvenance(BaseModel):
    """Tracks whether each model capability is verified, configured, or unknown."""

    context_length: CapabilitySource = CapabilitySource.UNKNOWN
    max_output_tokens: CapabilitySource = CapabilitySource.UNKNOWN
    streaming: CapabilitySource = CapabilitySource.UNKNOWN
    tool_calling: CapabilitySource = CapabilitySource.UNKNOWN
    json_mode: CapabilitySource = CapabilitySource.UNKNOWN
    vision: CapabilitySource = CapabilitySource.UNKNOWN


class ModelCapabilities(BaseModel):
    """Hardware, protocol, and context characteristics of a connected language model."""

    model_name: str = ""
    backend: str = LLMProviderType.OPENAI_COMPATIBLE.value
    max_context_tokens: int = Field(default=8192, ge=512)
    recommended_output_tokens: int = Field(default=1024, ge=64)
    supports_streaming: bool = False
    supports_tool_calling: bool = False
    supports_json_mode: bool = False
    supports_vision: bool = False
    sources: CapabilityProvenance = Field(default_factory=CapabilityProvenance)
    available_metadata: dict[str, Any] = Field(default_factory=dict)


class DiscoveredModel(BaseModel):
    """Model entry returned by an inference server discovery endpoint."""

    id: str
    name: str
    provider: str
    owned_by: str | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None
    context_length: int | None = None
    capabilities: ModelCapabilities | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderPreset(BaseModel):
    """Recommended endpoint defaults for common self-hosted runtimes."""

    id: str
    label: str
    provider: str
    default_base_url: str
    default_api_endpoint: str | None = None
    notes: str = ""


class LLMConnectionSummary(BaseModel):
    """Lightweight active-model summary for health checks and toolbar status."""

    id: str
    name: str
    provider: str
    model: str
    status: ConnectionStatus
    max_context_tokens: int
    recommended_output_tokens: int
    supports_streaming: bool
    allow_remote_endpoint: bool


class LLMConnectionPublic(BaseModel):
    """Frontend-safe LLM connection view (never includes the raw API key)."""

    id: str
    name: str
    provider: str
    preset: str | None = None
    base_url: str
    api_endpoint: str | None = None
    has_api_key: bool = False
    model: str = ""
    context_length: int
    max_output_tokens: int
    request_timeout_seconds: float
    streaming: bool
    allow_remote_endpoint: bool
    approved_by_user: bool
    is_active: bool
    status: ConnectionStatus
    last_error: str | None = None
    last_checked_at: datetime | None = None
    capabilities: ModelCapabilities | None = None
    created_at: datetime
    updated_at: datetime


class LLMConnectionConfig(BaseModel):
    """Internal persisted configuration for a user-hosted LLM endpoint."""

    id: str
    name: str = Field(..., min_length=1, max_length=120)
    provider: str = LLMProviderType.OPENAI_COMPATIBLE.value
    preset: str | None = None
    base_url: str
    api_endpoint: str | None = None
    api_key: SecretStr | None = None
    model: str = ""
    context_length: int = Field(default=8192, ge=512, le=2097152)
    max_output_tokens: int = Field(default=1024, ge=64, le=131072)
    request_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    streaming: bool = True
    allow_remote_endpoint: bool = False
    approved_by_user: bool = True
    is_active: bool = False
    status: ConnectionStatus = ConnectionStatus.CONFIGURED
    last_error: str | None = None
    last_checked_at: datetime | None = None
    capabilities: ModelCapabilities | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_token_budget(self) -> "LLMConnectionConfig":
        if self.max_output_tokens >= self.context_length:
            raise ValueError("max_output_tokens must be strictly smaller than context_length")
        return self

    def effective_capabilities(self) -> ModelCapabilities:
        """Return explicit capabilities distinguishing configured defaults when unverified."""
        if self.capabilities is not None:
            return self.capabilities
        return ModelCapabilities(
            model_name=self.model or "unselected",
            backend=self.provider,
            max_context_tokens=self.context_length,
            recommended_output_tokens=self.max_output_tokens,
            supports_streaming=self.streaming,
            supports_tool_calling=False,
            supports_json_mode=False,
            supports_vision=False,
            sources=CapabilityProvenance(
                context_length=CapabilitySource.CONFIGURED,
                max_output_tokens=CapabilitySource.CONFIGURED,
                streaming=CapabilitySource.CONFIGURED,
                tool_calling=CapabilitySource.UNKNOWN,
                json_mode=CapabilitySource.UNKNOWN,
                vision=CapabilitySource.UNKNOWN,
            ),
        )

    def to_public(self) -> LLMConnectionPublic:
        """Convert to a frontend-safe DTO with API key stripped."""
        raw_key = self.api_key.get_secret_value().strip() if self.api_key is not None else ""
        return LLMConnectionPublic(
            id=self.id,
            name=self.name,
            provider=self.provider,
            preset=self.preset,
            base_url=self.base_url,
            api_endpoint=self.api_endpoint,
            has_api_key=bool(raw_key),
            model=self.model,
            context_length=self.context_length,
            max_output_tokens=self.max_output_tokens,
            request_timeout_seconds=self.request_timeout_seconds,
            streaming=self.streaming,
            allow_remote_endpoint=self.allow_remote_endpoint,
            approved_by_user=self.approved_by_user,
            is_active=self.is_active,
            status=self.status,
            last_error=self.last_error,
            last_checked_at=self.last_checked_at,
            capabilities=self.capabilities,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def to_summary(self) -> LLMConnectionSummary:
        """Convert to a compact summary for health and toolbar UI."""
        caps = self.effective_capabilities()
        return LLMConnectionSummary(
            id=self.id,
            name=self.name,
            provider=self.provider,
            model=self.model,
            status=self.status,
            max_context_tokens=caps.max_context_tokens,
            recommended_output_tokens=caps.recommended_output_tokens,
            supports_streaming=caps.supports_streaming,
            allow_remote_endpoint=self.allow_remote_endpoint,
        )


class LLMConnectionCreate(BaseModel):
    """Request payload for adding a local or approved self-hosted LLM connection."""

    id: str | None = Field(default=None, min_length=1, max_length=80)
    name: str = Field(..., min_length=1, max_length=120)
    provider: str = Field(default=LLMProviderType.OPENAI_COMPATIBLE.value, min_length=1, max_length=60)
    preset: str | None = Field(default=None, max_length=60)
    base_url: str = Field(..., min_length=7, max_length=500)
    api_endpoint: str | None = Field(default=None, max_length=250)
    api_key: SecretStr | None = None
    model: str = Field(default="", max_length=200)
    context_length: int = Field(default=8192, ge=512, le=2097152)
    max_output_tokens: int = Field(default=1024, ge=64, le=131072)
    request_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    streaming: bool = True
    allow_remote_endpoint: bool = False
    approved_by_user: bool = True
    activate: bool = False

    @model_validator(mode="after")
    def _validate_token_budget(self) -> "LLMConnectionCreate":
        if self.max_output_tokens >= self.context_length:
            raise ValueError("max_output_tokens must be strictly smaller than context_length")
        return self


class LLMConnectionUpdate(BaseModel):
    """Partial update payload for modifying an existing LLM connection."""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    provider: str | None = Field(default=None, min_length=1, max_length=60)
    preset: str | None = Field(default=None, max_length=60)
    base_url: str | None = Field(default=None, min_length=7, max_length=500)
    api_endpoint: str | None = Field(default=None, max_length=250)
    api_key: SecretStr | None = None
    clear_api_key: bool = False
    model: str | None = Field(default=None, max_length=200)
    context_length: int | None = Field(default=None, ge=512, le=2097152)
    max_output_tokens: int | None = Field(default=None, ge=64, le=131072)
    request_timeout_seconds: float | None = Field(default=None, ge=1.0, le=600.0)
    streaming: bool | None = None
    allow_remote_endpoint: bool | None = None
    approved_by_user: bool | None = None


class ChatMessage(BaseModel):
    """Single message in a chat completion conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None


class ToolDefinition(BaseModel):
    """OpenAI-style function/tool schema definition."""

    name: str = Field(..., min_length=1, max_length=120)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """Parsed tool invocation emitted by an inference model."""

    id: str | None = None
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class TokenUsage(BaseModel):
    """Token consumption reported by the inference endpoint when available."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMInferenceRequest(BaseModel):
    """Normalized request object passed to a LocalModelAdapter."""

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    stream: bool = False
    response_format: Literal["text", "json_object"] = "text"
    tools: list[ToolDefinition] = Field(default_factory=list)


class LLMInferenceResponse(BaseModel):
    """Normalized non-streaming response from a LocalModelAdapter."""

    content: str = ""
    model: str = ""
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    parsed_json: dict[str, Any] | list[Any] | None = None
    usage: TokenUsage | None = None
    latency_ms: int = 0


class LLMStreamChunk(BaseModel):
    """Single delta emitted while streaming completion tokens."""

    delta: str = ""
    done: bool = False
    finish_reason: str | None = None
    model: str | None = None
    usage: TokenUsage | None = None


class ConnectionTestResponse(BaseModel):
    """Result returned by testing connectivity and inspecting capabilities."""

    connection: LLMConnectionPublic
    ok: bool
    message: str
    latency_ms: int = 0
    discovered_models: list[DiscoveredModel] = Field(default_factory=list)
    capabilities: ModelCapabilities | None = None


class ModelDiscoveryResponse(BaseModel):
    """Result returned by querying available models from a connected server."""

    connection_id: str
    provider: str
    models: list[DiscoveredModel] = Field(default_factory=list)
    discovery_supported: bool = True
    message: str = ""


class TestPromptRequest(BaseModel):
    """Request body for running a verification prompt against a connection."""

    __test__: ClassVar[bool] = False

    prompt: str = Field(..., min_length=1, max_length=8000)
    system_prompt: str | None = Field(default=None, max_length=2000)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=16, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class TestPromptResponse(BaseModel):
    """Result returned for non-streaming prompt verification."""

    __test__: ClassVar[bool] = False

    connection_id: str
    model: str
    content: str
    latency_ms: int
    usage: TokenUsage | None = None
