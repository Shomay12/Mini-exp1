"""Dynamic token budget manager coordinating model capabilities, user config, and policies."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from web_intelligence.app.context.allocation_policy import AllocationPolicy
from web_intelligence.app.context.models import (
    AllocationProfile,
    BudgetAllocation,
    ContextBudgetConfig,
    ContextBudgetUpdateRequest,
    ExpectedOutputLength,
    ModelContextCapabilities,
)
from web_intelligence.app.context.token_estimator import (
    BaseContextTokenEstimator,
    inspect_adapter_context_capabilities,
    normalize_model_context_capabilities,
    resolve_token_estimator,
)
from web_intelligence.app.schemas.llm import LLMConnectionConfig, ModelCapabilities


class DynamicContextBudgetManager:
    """Computes and validates context token budgets against model and user limits."""

    def __init__(
        self,
        policy: AllocationPolicy | None = None,
        *,
        default_context_limit: int = 8192,
        default_output_tokens: int = 1024,
    ) -> None:
        self._policy = policy or AllocationPolicy()
        self._default_context_limit = default_context_limit
        self._default_output_tokens = default_output_tokens

    async def inspect_capabilities(
        self,
        adapter: Any = None,
        *,
        config: LLMConnectionConfig | None = None,
        user_context_limit: int | None = None,
    ) -> ModelContextCapabilities:
        """Inspect normalized capabilities from a connected LocalModelAdapter or fallback config."""
        return await inspect_adapter_context_capabilities(
            adapter,
            config=config,
            user_context_limit=user_context_limit,
            default_context_limit=self._default_context_limit,
            default_output_tokens=self._default_output_tokens,
        )

    def normalize_capabilities(
        self,
        capabilities: ModelCapabilities | None,
        *,
        config: LLMConnectionConfig | None = None,
        user_context_limit: int | None = None,
        adapter: Any = None,
    ) -> ModelContextCapabilities:
        """Synchronously normalize pre-fetched ModelCapabilities into ModelContextCapabilities."""
        return normalize_model_context_capabilities(
            capabilities,
            config=config,
            user_context_limit=user_context_limit,
            adapter=adapter,
            default_context_limit=self._default_context_limit,
            default_output_tokens=self._default_output_tokens,
        )

    def plan_budget(
        self,
        capabilities: ModelContextCapabilities,
        budget_config: ContextBudgetConfig,
        *,
        system_text: str | None = None,
        query_text: str | None = None,
        available_evidence_tokens: int | None = None,
        available_history_tokens: int | None = None,
        available_metadata_tokens: int | None = None,
        expected_output_length: ExpectedOutputLength | None = None,
        estimator: BaseContextTokenEstimator | None = None,
    ) -> BudgetAllocation:
        """Compute bounded 7-component BudgetAllocation respecting model and user limits."""
        token_counter = estimator or resolve_token_estimator()

        actual_sys = (
            token_counter.estimate(system_text).tokens if system_text is not None else None
        )
        actual_qry = (
            token_counter.estimate(query_text).tokens if query_text is not None else None
        )

        return self._policy.compute_allocation(
            capabilities,
            budget_config,
            actual_system_tokens=actual_sys,
            actual_query_tokens=actual_qry,
            available_evidence_tokens=available_evidence_tokens,
            available_history_tokens=available_history_tokens,
            available_metadata_tokens=available_metadata_tokens,
            expected_output_length=expected_output_length,
            query_text=query_text,
        )

    def validate_and_apply_update(
        self,
        current: ContextBudgetConfig,
        update: ContextBudgetUpdateRequest,
        capabilities: ModelContextCapabilities,
    ) -> tuple[ContextBudgetConfig, BudgetAllocation]:
        """Apply validated updates to ContextBudgetConfig and verify allocation feasibility."""
        data = current.model_dump()

        if update.profile is not None:
            data["profile"] = update.profile
        if update.clear_user_context_limit:
            data["user_context_limit"] = None
        elif update.user_context_limit is not None:
            data["user_context_limit"] = update.user_context_limit

        if update.clear_output_reservation:
            data["output_reservation_tokens"] = None
        elif update.output_reservation_tokens is not None:
            data["output_reservation_tokens"] = update.output_reservation_tokens

        if update.clear_safety_margin:
            data["safety_margin_tokens"] = None
        elif update.safety_margin_tokens is not None:
            data["safety_margin_tokens"] = update.safety_margin_tokens

        if update.custom_ratios is not None:
            data["custom_ratios"] = update.custom_ratios
            if update.profile is None:
                data["profile"] = AllocationProfile.CUSTOM

        if update.expected_output_length is not None:
            data["expected_output_length"] = update.expected_output_length
        if update.min_evidence_tokens is not None:
            data["min_evidence_tokens"] = update.min_evidence_tokens
        if update.max_history_turns is not None:
            data["max_history_turns"] = update.max_history_turns
        if update.compression_strategy is not None:
            data["compression_strategy"] = update.compression_strategy
        if update.allow_model_assisted_compression is not None:
            data["allow_model_assisted_compression"] = update.allow_model_assisted_compression

        data["updated_at"] = datetime.now(timezone.utc)
        candidate = ContextBudgetConfig.model_validate(data)
        allocation = self.plan_budget(capabilities, candidate)
        return candidate, allocation
