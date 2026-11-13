"""Task-aware allocation policies and profile rules for the Adaptive Context Engine."""

from __future__ import annotations

from dataclasses import dataclass

from web_intelligence.app.context.models import (
    BUDGET_BUCKET_KEYS,
    AllocationProfile,
    BudgetAllocation,
    ContextBudgetConfig,
    ContextConfigurationError,
    ContextLimitExceededError,
    ExpectedOutputLength,
    ModelContextCapabilities,
)
from web_intelligence.app.schemas.llm import CapabilitySource


@dataclass(frozen=True)
class ProfilePolicySpec:
    """Baseline dynamic policy parameters for an AllocationProfile."""

    profile: AllocationProfile
    output_ratio: float
    min_output_tokens: int
    max_output_tokens: int
    safety_ratio: float
    min_safety_tokens: int
    bucket_weights: dict[str, float]
    description: str


PROFILE_SPECS: dict[AllocationProfile, ProfilePolicySpec] = {
    AllocationProfile.QUICK_ANSWER: ProfilePolicySpec(
        profile=AllocationProfile.QUICK_ANSWER,
        output_ratio=0.12,
        min_output_tokens=128,
        max_output_tokens=768,
        safety_ratio=0.04,
        min_safety_tokens=32,
        bucket_weights={
            "system": 0.14,
            "query": 0.12,
            "history": 0.22,
            "evidence": 0.42,
            "metadata": 0.10,
        },
        description="Fast synthesis favoring concise answers and focused high-relevance evidence.",
    ),
    AllocationProfile.STANDARD_RESEARCH: ProfilePolicySpec(
        profile=AllocationProfile.STANDARD_RESEARCH,
        output_ratio=0.18,
        min_output_tokens=256,
        max_output_tokens=1536,
        safety_ratio=0.05,
        min_safety_tokens=48,
        bucket_weights={
            "system": 0.10,
            "query": 0.08,
            "history": 0.14,
            "evidence": 0.54,
            "metadata": 0.14,
        },
        description="Balanced multi-source research with claim/conflict metadata and recent history.",
    ),
    AllocationProfile.DEEP_RESEARCH: ProfilePolicySpec(
        profile=AllocationProfile.DEEP_RESEARCH,
        output_ratio=0.16,
        min_output_tokens=320,
        max_output_tokens=2048,
        safety_ratio=0.06,
        min_safety_tokens=64,
        bucket_weights={
            "system": 0.06,
            "query": 0.04,
            "history": 0.06,
            "evidence": 0.68,
            "metadata": 0.16,
        },
        description="Maximizes independent evidence breadth and conflict/coverage metadata.",
    ),
    AllocationProfile.LONG_FORM: ProfilePolicySpec(
        profile=AllocationProfile.LONG_FORM,
        output_ratio=0.34,
        min_output_tokens=512,
        max_output_tokens=4096,
        safety_ratio=0.06,
        min_safety_tokens=64,
        bucket_weights={
            "system": 0.10,
            "query": 0.08,
            "history": 0.12,
            "evidence": 0.54,
            "metadata": 0.16,
        },
        description="Reserves large output headroom for structured long-form grounded reports.",
    ),
    AllocationProfile.CUSTOM: ProfilePolicySpec(
        profile=AllocationProfile.CUSTOM,
        output_ratio=0.18,
        min_output_tokens=256,
        max_output_tokens=2048,
        safety_ratio=0.05,
        min_safety_tokens=48,
        bucket_weights={
            "system": 0.10,
            "query": 0.08,
            "history": 0.14,
            "evidence": 0.54,
            "metadata": 0.14,
        },
        description="User-defined ratios and bounds validated within active model limits.",
    ),
}

_OUTPUT_LENGTH_MULTIPLIER: dict[ExpectedOutputLength, float] = {
    ExpectedOutputLength.SHORT: 0.65,
    ExpectedOutputLength.MEDIUM: 1.0,
    ExpectedOutputLength.LONG: 1.45,
}


class AllocationPolicy:
    """Computes task-, evidence-, and output-aware token budgets strictly inside model limits."""

    def compute_allocation(
        self,
        capabilities: ModelContextCapabilities,
        config: ContextBudgetConfig,
        *,
        actual_system_tokens: int | None = None,
        actual_query_tokens: int | None = None,
        available_evidence_tokens: int | None = None,
        available_history_tokens: int | None = None,
        available_metadata_tokens: int | None = None,
        expected_output_length: ExpectedOutputLength | None = None,
        query_text: str | None = None,
    ) -> BudgetAllocation:
        """Compute bounded 7-bucket BudgetAllocation without exceeding active model limits."""
        spec = PROFILE_SPECS[config.profile]
        warnings: list[str] = list(capabilities.warnings)

        effective_limit = self._resolve_effective_limit(capabilities, config)
        safety_margin = self._resolve_safety_margin(effective_limit, capabilities, config, spec)
        output_tokens = self._resolve_output_reservation(
            effective_limit,
            capabilities,
            config,
            spec,
            safety_margin,
            expected_output_length or config.expected_output_length,
            query_text=query_text,
        )

        input_budget = effective_limit - safety_margin - output_tokens
        if input_budget < 128:
            raise ContextConfigurationError(
                f"Configured output reservation ({output_tokens}) and safety margin ({safety_margin}) "
                f"leave only {input_budget} tokens inside context limit {effective_limit} "
                "(minimum 128 input tokens required)."
            )

        weights = self._resolve_weights(config, spec, query_text=query_text)
        buckets = {
            key: int(input_budget * weights[key]) for key in BUDGET_BUCKET_KEYS
        }

        # Rebalance dynamically when actual prompt/evidence/history demand is known.
        rationale_notes: list[str] = [
            f"Profile={config.profile.value}",
            f"output_length={(expected_output_length or config.expected_output_length).value}",
        ]

        if actual_system_tokens is not None or actual_query_tokens is not None:
            sys_need = actual_system_tokens if actual_system_tokens is not None else buckets["system"]
            qry_need = actual_query_tokens if actual_query_tokens is not None else buckets["query"]
            # Cap fixed system+query reservation so other buckets stay non-negative;
            # ContextValidator raises overflow if actual prompt exceeds budget.
            min_non_prompt = min(192, input_budget // 3)
            max_prompt_pool = max(64, input_budget - min_non_prompt)
            if sys_need + qry_need <= max_prompt_pool:
                buckets["system"] = max(32, sys_need)
                buckets["query"] = max(24, qry_need)
            else:
                warnings.append(
                    f"System+query prompt ({sys_need + qry_need} tok) presses against input budget ({input_budget} tok)."
                )
                buckets["system"] = min(sys_need, max_prompt_pool // 2)
                buckets["query"] = min(qry_need, max_prompt_pool - buckets["system"])

            remaining_pool = max(0, input_budget - buckets["system"] - buckets["query"])
            flex_keys = ("evidence", "metadata", "history")
            flex_weight_sum = sum(weights[k] for k in flex_keys) or 1.0
            for k in flex_keys:
                buckets[k] = int(remaining_pool * (weights[k] / flex_weight_sum))

        # Dynamic shift from unused history when history supply is explicitly lower than budget
        if available_history_tokens is not None and available_history_tokens < buckets["history"]:
            unused_hist = buckets["history"] - max(0, available_history_tokens)
            buckets["history"] = max(0, available_history_tokens)
            ev_share = int(unused_hist * 0.78)
            buckets["evidence"] += ev_share
            buckets["metadata"] += unused_hist - ev_share
            rationale_notes.append(f"reallocated {unused_hist} unused history tok to evidence/metadata")

        # When conflict/structured metadata is richer than expected, gently rebalance from history if slack exists
        if available_metadata_tokens is not None and available_metadata_tokens > buckets["metadata"]:
            needed_meta = min(
                available_metadata_tokens - buckets["metadata"],
                max(0, buckets["history"] // 3),
            )
            if needed_meta > 0:
                buckets["metadata"] += needed_meta
                buckets["history"] -= needed_meta
                rationale_notes.append(f"boosted metadata by {needed_meta} tok")

        # Distribute integer rounding remainder into evidence bucket (never exceeding input_budget)
        allocated_input_sum = sum(buckets.values())
        remainder = input_budget - allocated_input_sum
        if remainder > 0:
            buckets["evidence"] += remainder
        elif remainder < 0:
            # Guard against rounding overshoot
            buckets["evidence"] = max(0, buckets["evidence"] + remainder)

        if available_evidence_tokens is not None and available_evidence_tokens < buckets["evidence"]:
            rationale_notes.append(
                f"evidence supply ({available_evidence_tokens} tok) fits inside budget ({buckets['evidence']} tok)"
            )

        total_planned = sum(buckets.values()) + output_tokens + safety_margin

        return BudgetAllocation(
            profile=config.profile,
            model_name=capabilities.model_name or capabilities.provider,
            effective_context_limit=effective_limit,
            model_max_context_tokens=capabilities.max_context_tokens,
            system_instructions_tokens=buckets["system"],
            user_query_tokens=buckets["query"],
            conversation_history_tokens=buckets["history"],
            research_evidence_tokens=buckets["evidence"],
            structured_metadata_tokens=buckets["metadata"],
            output_reservation_tokens=output_tokens,
            safety_margin_tokens=safety_margin,
            input_budget_tokens=sum(buckets.values()),
            total_planned_tokens=total_planned,
            policy_rationale="; ".join(rationale_notes),
            tokenizer_available=capabilities.tokenizer_available,
            estimation_is_approximate=capabilities.estimation_is_approximate,
            warnings=warnings,
        )

    def _resolve_effective_limit(
        self,
        capabilities: ModelContextCapabilities,
        config: ContextBudgetConfig,
    ) -> int:
        if config.user_context_limit is not None:
            if config.user_context_limit > capabilities.max_context_tokens:
                raise ContextLimitExceededError(
                    f"Configured user_context_limit ({config.user_context_limit}) exceeds "
                    f"{capabilities.context_limit_source.value} model context limit "
                    f"({capabilities.max_context_tokens})."
                )
            return config.user_context_limit
        return capabilities.effective_context_limit

    def _resolve_safety_margin(
        self,
        effective_limit: int,
        capabilities: ModelContextCapabilities,
        config: ContextBudgetConfig,
        spec: ProfilePolicySpec,
    ) -> int:
        if config.safety_margin_tokens is not None:
            if config.safety_margin_tokens >= effective_limit - 192:
                raise ContextConfigurationError(
                    f"safety_margin_tokens ({config.safety_margin_tokens}) leaves insufficient room "
                    f"within effective context limit ({effective_limit})."
                )
            return config.safety_margin_tokens

        approx_bonus = 0.02 if capabilities.estimation_is_approximate else 0.0
        raw_margin = round(effective_limit * (spec.safety_ratio + approx_bonus))
        bounded = max(spec.min_safety_tokens, raw_margin)
        return min(bounded, max(32, effective_limit // 6))

    def _resolve_output_reservation(
        self,
        effective_limit: int,
        capabilities: ModelContextCapabilities,
        config: ContextBudgetConfig,
        spec: ProfilePolicySpec,
        safety_margin: int,
        expected_output_length: ExpectedOutputLength,
        *,
        query_text: str | None = None,
    ) -> int:
        if config.output_reservation_tokens is not None:
            if config.output_reservation_tokens >= effective_limit - safety_margin:
                raise ContextLimitExceededError(
                    f"output_reservation_tokens ({config.output_reservation_tokens}) plus safety margin "
                    f"({safety_margin}) exceeds effective context limit ({effective_limit})."
                )
            if (
                capabilities.has_active_model
                and capabilities.output_limit_source == CapabilitySource.VERIFIED
                and config.output_reservation_tokens > capabilities.max_output_tokens
            ):
                raise ContextLimitExceededError(
                    f"output_reservation_tokens ({config.output_reservation_tokens}) exceeds verified "
                    f"model max_output_tokens ({capabilities.max_output_tokens})."
                )
            return config.output_reservation_tokens

        mult = _OUTPUT_LENGTH_MULTIPLIER[expected_output_length]
        if query_text:
            lowered = query_text.lower()
            if any(tok in lowered for tok in ("comprehensive", "report", "detailed analysis", "compare in depth")):
                mult = max(mult, 1.25)
            elif any(tok in lowered for tok in ("briefly", "one sentence", "yes or no", "quick summary")):
                mult = min(mult, 0.75)

        target = round(effective_limit * spec.output_ratio * mult)
        ceiling = min(spec.max_output_tokens, capabilities.max_output_tokens, effective_limit - safety_margin - 192)
        floor = min(spec.min_output_tokens, ceiling) if ceiling >= 64 else 64
        return max(64, min(max(floor, target), max(64, ceiling)))

    def _resolve_weights(
        self,
        config: ContextBudgetConfig,
        spec: ProfilePolicySpec,
        *,
        query_text: str | None = None,
    ) -> dict[str, float]:
        if config.profile == AllocationProfile.CUSTOM and config.custom_ratios:
            raw = {key: float(config.custom_ratios.get(key, 0.0)) for key in BUDGET_BUCKET_KEYS}
        else:
            raw = dict(spec.bucket_weights)

        # Subtle task-type weight adaptation based on query structure when not Custom
        if query_text and config.profile != AllocationProfile.CUSTOM:
            lowered = query_text.lower()
            if any(word in lowered for word in ("conflict", "contradict", "disagree", "compare", "versus", "vs")):
                raw["metadata"] = raw["metadata"] + 0.04
                raw["evidence"] = max(0.30, raw["evidence"] - 0.02)
                raw["history"] = max(0.04, raw["history"] - 0.02)

        total = sum(raw.values())
        if total <= 0:
            raise ContextConfigurationError("Allocation bucket weights must sum to > 0.")
        return {k: v / total for k, v in raw.items()}
