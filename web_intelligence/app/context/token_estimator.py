"""Adapter-aware token estimation and conservative fallback strategies."""

from __future__ import annotations

import inspect
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from web_intelligence.app.context.models import (
    ContextLimitExceededError,
    ModelContextCapabilities,
    TokenEstimate,
)
from web_intelligence.app.schemas.llm import (
    CapabilitySource,
    ChatMessage,
    LLMConnectionConfig,
    ModelCapabilities,
)

_SYMBOL_RE = re.compile(r"[^\w\s]", re.UNICODE)
_DIGIT_GROUP_RE = re.compile(r"\d+")
_MESSAGE_FRAMING_OVERHEAD = 6
_CONVERSATION_PRIMING_OVERHEAD = 4


class BaseContextTokenEstimator(ABC):
    """Protocol-compatible token estimator exposing both count and precision metadata."""

    @property
    @abstractmethod
    def tokenizer_available(self) -> bool:
        """Whether counts come from a real model tokenizer rather than heuristic fallback."""

    @property
    @abstractmethod
    def tokenizer_id(self) -> str | None:
        """Identifier of the active tokenizer, when available."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable identifier of the estimation method."""

    def estimate_tokens(self, text: str) -> int:
        """Return integer token count compatible with Stage 2 TokenEstimator protocol."""
        return self.estimate(text).tokens

    @abstractmethod
    def estimate(self, text: str) -> TokenEstimate:
        """Return detailed TokenEstimate marking exact vs approximate status."""

    def estimate_messages(self, messages: list[ChatMessage]) -> TokenEstimate:
        """Estimate total tokens for a structured chat-message sequence with framing overhead."""
        if not messages:
            return self.estimate("")

        total_tokens = _CONVERSATION_PRIMING_OVERHEAD
        total_chars = 0
        total_words = 0
        for message in messages:
            role_est = self.estimate(message.role)
            content_est = self.estimate(message.content)
            total_tokens += _MESSAGE_FRAMING_OVERHEAD + role_est.tokens + content_est.tokens
            total_chars += len(message.role) + len(message.content)
            total_words += role_est.word_count + content_est.word_count

        return TokenEstimate(
            tokens=total_tokens,
            is_approximate=not self.tokenizer_available,
            strategy=f"{self.strategy_name}_chat",
            tokenizer_available=self.tokenizer_available,
            tokenizer_id=self.tokenizer_id,
            char_count=total_chars,
            word_count=total_words,
            note=(
                f"Includes {_MESSAGE_FRAMING_OVERHEAD} tok/message chat framing "
                f"and {_CONVERSATION_PRIMING_OVERHEAD} priming tokens."
                + (
                    ""
                    if self.tokenizer_available
                    else " Underlying text tokens are conservatively approximated."
                )
            ),
        )


class ConservativeHeuristicEstimator(BaseContextTokenEstimator):
    """Conservative tokenizer-free estimator intended never to undercount English, JSON, or code."""

    @property
    def tokenizer_available(self) -> bool:
        return False

    @property
    def tokenizer_id(self) -> str | None:
        return None

    @property
    def strategy_name(self) -> str:
        return "conservative_heuristic"

    def estimate(self, text: str) -> TokenEstimate:
        if not text:
            return TokenEstimate(
                tokens=0,
                is_approximate=True,
                strategy=self.strategy_name,
                tokenizer_available=False,
                tokenizer_id=None,
                char_count=0,
                word_count=0,
                note="Empty input; 0 tokens.",
            )

        char_count = len(text)
        words = text.split()
        word_count = len(words)
        symbol_count = len(_SYMBOL_RE.findall(text))
        digit_groups = len(_DIGIT_GROUP_RE.findall(text))
        utf8_extra_bytes = max(0, len(text.encode("utf-8")) - char_count)

        # Conservative BPE upper-bound estimates across prose (~3.2-4 chars/tok),
        # symbols/JSON syntax, numbers, and non-ASCII multibyte text.
        char_bound = math.ceil(char_count / 3.2)
        lexical_bound = math.ceil(
            (word_count * 1.40)
            + (symbol_count * 0.40)
            + (digit_groups * 0.65)
            + (utf8_extra_bytes * 0.60)
        )
        tokens = max(1, char_bound, lexical_bound)

        return TokenEstimate(
            tokens=tokens,
            is_approximate=True,
            strategy=self.strategy_name,
            tokenizer_available=False,
            tokenizer_id=None,
            char_count=char_count,
            word_count=word_count,
            note=(
                "Model tokenizer metadata unavailable; using conservative upper-bound heuristic "
                "(approx. tokens only)."
            ),
        )


class TokenizerBackedEstimator(BaseContextTokenEstimator):
    """Exact token counter wrapping an adapter-supplied or injected local tokenizer."""

    def __init__(
        self,
        counter: Callable[[str], int],
        *,
        tokenizer_id: str = "adapter_tokenizer",
        strategy_name: str = "adapter_tokenizer",
    ) -> None:
        self._counter = counter
        self._tokenizer_id = tokenizer_id
        self._strategy_name = strategy_name

    @property
    def tokenizer_available(self) -> bool:
        return True

    @property
    def tokenizer_id(self) -> str | None:
        return self._tokenizer_id

    @property
    def strategy_name(self) -> str:
        return self._strategy_name

    def estimate(self, text: str) -> TokenEstimate:
        if not text:
            return TokenEstimate(
                tokens=0,
                is_approximate=False,
                strategy=self._strategy_name,
                tokenizer_available=True,
                tokenizer_id=self._tokenizer_id,
                char_count=0,
                word_count=0,
                note="Exact tokenizer count (0 tokens).",
            )

        count = max(0, int(self._counter(text)))
        return TokenEstimate(
            tokens=count,
            is_approximate=False,
            strategy=self._strategy_name,
            tokenizer_available=True,
            tokenizer_id=self._tokenizer_id,
            char_count=len(text),
            word_count=len(text.split()),
            note=f"Counted with active tokenizer '{self._tokenizer_id}'.",
        )


def _extract_tokenizer_from_object(
    obj: Any,
    capabilities: ModelCapabilities | None = None,
) -> TokenizerBackedEstimator | None:
    """Detect a real offline tokenizer callable on an adapter or capabilities metadata."""
    if obj is not None:
        count_tokens_fn = getattr(obj, "count_tokens", None)
        if callable(count_tokens_fn):
            meta_id = (
                capabilities.available_metadata.get("tokenizer_id")
                if capabilities and isinstance(capabilities.available_metadata, dict)
                else None
            )
            return TokenizerBackedEstimator(
                count_tokens_fn,
                tokenizer_id=str(meta_id or f"{getattr(obj, 'provider', 'adapter')}:count_tokens"),
                strategy_name="adapter_count_tokens",
            )

        tokenizer_attr = getattr(obj, "tokenizer", None)
        if tokenizer_attr is not None:
            encode_fn = getattr(tokenizer_attr, "encode", None)
            if callable(encode_fn):
                tok_id = str(getattr(tokenizer_attr, "name", "adapter_tokenizer"))
                return TokenizerBackedEstimator(
                    lambda s: len(encode_fn(s)),
                    tokenizer_id=tok_id,
                    strategy_name="adapter_tokenizer",
                )
            if callable(tokenizer_attr):
                return TokenizerBackedEstimator(
                    lambda s: int(tokenizer_attr(s)),
                    tokenizer_id="adapter_tokenizer_callable",
                    strategy_name="adapter_tokenizer",
                )

    if capabilities and isinstance(capabilities.available_metadata, dict):
        meta = capabilities.available_metadata
        meta_counter = meta.get("tokenizer_callable")
        if callable(meta_counter):
            return TokenizerBackedEstimator(
                meta_counter,
                tokenizer_id=str(meta.get("tokenizer_id") or "metadata_tokenizer"),
                strategy_name="metadata_tokenizer",
            )
        meta_encode = getattr(meta.get("tokenizer"), "encode", None)
        if callable(meta_encode):
            return TokenizerBackedEstimator(
                lambda s: len(meta_encode(s)),
                tokenizer_id=str(meta.get("tokenizer_id") or "metadata_tokenizer"),
                strategy_name="metadata_tokenizer",
            )

    return None


def resolve_token_estimator(
    adapter: Any = None,
    capabilities: ModelCapabilities | None = None,
    *,
    custom_tokenizer: Callable[[str], int] | None = None,
    custom_tokenizer_id: str | None = None,
) -> BaseContextTokenEstimator:
    """Resolve exact tokenizer when available, otherwise fall back conservatively."""
    if custom_tokenizer is not None:
        return TokenizerBackedEstimator(
            custom_tokenizer,
            tokenizer_id=custom_tokenizer_id or "custom_tokenizer",
            strategy_name="custom_tokenizer",
        )

    detected = _extract_tokenizer_from_object(adapter, capabilities)
    if detected is not None:
        return detected

    return ConservativeHeuristicEstimator()


def normalize_model_context_capabilities(
    capabilities: ModelCapabilities | None,
    *,
    config: LLMConnectionConfig | None = None,
    user_context_limit: int | None = None,
    adapter: Any = None,
    default_context_limit: int = 8192,
    default_output_tokens: int = 1024,
) -> ModelContextCapabilities:
    """Build ModelContextCapabilities strictly from verified or explicitly configured limits."""
    warnings: list[str] = []

    if capabilities is None:
        base_limit = config.context_length if config else default_context_limit
        base_source = (
            CapabilitySource.CONFIGURED if config else CapabilitySource.UNKNOWN
        )
        out_limit = config.max_output_tokens if config else default_output_tokens
        out_source = CapabilitySource.CONFIGURED if config else CapabilitySource.UNKNOWN
        model_name = config.model if config else ""
        provider = config.provider if config else "unconfigured"
        conn_id = config.id if config else None
        raw_meta: dict[str, Any] = {}
        supports_json = False
        supports_tools = False
        supports_stream = False
        has_active = config is not None
        if config is None:
            warnings.append(
                f"No active local model connected; using fallback context limit {base_limit} "
                "and approximate token estimation."
            )
    else:
        base_limit = capabilities.max_context_tokens
        base_source = capabilities.sources.context_length
        out_limit = capabilities.recommended_output_tokens
        out_source = capabilities.sources.max_output_tokens
        model_name = capabilities.model_name or (config.model if config else "")
        provider = capabilities.backend or (config.provider if config else "local")
        conn_id = config.id if config else None
        raw_meta = dict(capabilities.available_metadata)
        supports_json = capabilities.supports_json_mode
        supports_tools = capabilities.supports_tool_calling
        supports_stream = capabilities.supports_streaming
        has_active = True

        # Respect explicit user connection context_length ceiling if lower than adapter default.
        if config is not None:
            if (
                base_source == CapabilitySource.UNKNOWN
                or config.context_length < base_limit
            ):
                if config.context_length > base_limit and base_source == CapabilitySource.VERIFIED:
                    warnings.append(
                        f"Configured connection limit ({config.context_length}) exceeds verified model "
                        f"limit ({base_limit}); capped at {base_limit}."
                    )
                else:
                    base_limit = config.context_length
                    base_source = CapabilitySource.CONFIGURED
            if config.max_output_tokens < out_limit or out_source == CapabilitySource.UNKNOWN:
                out_limit = min(config.max_output_tokens, base_limit - 256) if base_limit > 512 else config.max_output_tokens
                out_limit = max(64, out_limit)
                out_source = CapabilitySource.CONFIGURED

        if base_source == CapabilitySource.UNKNOWN:
            warnings.append(
                f"Model endpoint did not report a verified context length for '{model_name or provider}'; "
                f"using configured/safe limit {base_limit} tokens without inferring from model name."
            )

    if user_context_limit is not None:
        if user_context_limit > base_limit:
            raise ContextLimitExceededError(
                f"Requested user_context_limit ({user_context_limit}) exceeds "
                f"{base_source.value} model context limit ({base_limit}) for '{model_name or provider}'."
            )
        effective_limit = user_context_limit
    else:
        effective_limit = base_limit

    estimator = resolve_token_estimator(adapter, capabilities)
    if not estimator.tokenizer_available:
        warnings.append(
            "Tokenizer unavailable for active model; token counts are conservative approximations."
        )

    return ModelContextCapabilities(
        model_name=model_name,
        provider=provider,
        connection_id=conn_id,
        has_active_model=has_active,
        max_context_tokens=base_limit,
        configured_context_limit=user_context_limit or (config.context_length if config else None),
        effective_context_limit=effective_limit,
        max_output_tokens=min(out_limit, max(64, effective_limit - 128)),
        context_limit_source=base_source,
        output_limit_source=out_source,
        tokenizer_available=estimator.tokenizer_available,
        tokenizer_id=estimator.tokenizer_id,
        estimation_strategy=estimator.strategy_name,
        estimation_is_approximate=not estimator.tokenizer_available,
        supports_structured_output=supports_json,
        supports_tool_calling=supports_tools,
        supports_streaming=supports_stream,
        warnings=warnings,
        raw_metadata=raw_meta,
    )


async def inspect_adapter_context_capabilities(
    adapter: Any = None,
    *,
    config: LLMConnectionConfig | None = None,
    user_context_limit: int | None = None,
    default_context_limit: int = 8192,
    default_output_tokens: int = 1024,
) -> ModelContextCapabilities:
    """Inspect capabilities from a live LocalModelAdapter (sync or async) safely."""
    caps: ModelCapabilities | None = None
    if adapter is not None:
        inspect_fn = getattr(adapter, "inspect_model_capabilities", None)
        caps_fn = getattr(adapter, "capabilities", None)
        if callable(caps_fn):
            maybe = caps_fn()
            caps = await maybe if inspect.isawaitable(maybe) else maybe
        elif callable(inspect_fn):
            model_name = config.model if config else None
            maybe = inspect_fn(model_name)
            caps = await maybe if inspect.isawaitable(maybe) else maybe
    elif config is not None and config.capabilities is not None:
        caps = config.capabilities

    return normalize_model_context_capabilities(
        caps,
        config=config,
        user_context_limit=user_context_limit,
        adapter=adapter,
        default_context_limit=default_context_limit,
        default_output_tokens=default_output_tokens,
    )
