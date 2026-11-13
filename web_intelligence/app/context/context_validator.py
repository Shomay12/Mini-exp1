"""Pre-inference context validation ensuring strict budget, citation, and structural integrity."""

from __future__ import annotations

import re

from web_intelligence.app.context.models import (
    ContextPackage,
    ContextValidationIssue,
    ContextValidationReport,
)
from web_intelligence.app.context.token_estimator import (
    BaseContextTokenEstimator,
    ConservativeHeuristicEstimator,
)
from web_intelligence.app.schemas.llm import ChatMessage

_CITATION_RE = re.compile(r"\[E\d+\]")


class ContextValidator:
    """Performs 7-stage pre-inference validation without ever silently truncating evidence."""

    def __init__(self, estimator: BaseContextTokenEstimator | None = None) -> None:
        self._estimator = estimator or ConservativeHeuristicEstimator()

    def validate(
        self,
        package: ContextPackage,
        *,
        adapter_messages: list[ChatMessage] | None = None,
        estimator: BaseContextTokenEstimator | None = None,
        strict_section_caps: bool = True,
    ) -> ContextValidationReport:
        """Execute all pre-inference context checks and return actionable diagnostics."""
        counter = estimator or self._estimator
        issues: list[ContextValidationIssue] = []
        checked_rules = [
            "token_usage_estimation",
            "active_context_limit_compliance",
            "output_token_reservation",
            "source_reference_integrity",
            "context_section_well_formedness",
            "configured_budget_compliance",
            "non_truncation_evidence_guard",
        ]

        # 1. Re-calculate actual token usage cleanly from rendered messages or sections.
        if adapter_messages:
            measured_from_messages = counter.estimate_messages(adapter_messages).tokens
        else:
            measured_from_messages = (
                counter.estimate(package.system_instructions).tokens
                + counter.estimate(package.rendered_user_prompt).tokens
                + 16
            )
        measured_input = max(package.total_input_tokens, measured_from_messages)

        active_limit = min(
            package.capabilities.effective_context_limit,
            package.budget_allocation.effective_context_limit,
            package.capabilities.max_context_tokens,
        )
        output_tokens = package.output_reservation_tokens
        total_required = measured_input + output_tokens
        remaining = active_limit - total_required
        approx = not counter.tokenizer_available or package.estimation_is_approximate

        # 2. Validate active context limit
        if total_required > active_limit:
            overage = total_required - active_limit
            issues.append(
                ContextValidationIssue(
                    code="CONTEXT_LIMIT_EXCEEDED",
                    severity="error",
                    section="total",
                    message=(
                        f"Combined input ({measured_input} tok) + reserved output ({output_tokens} tok) = "
                        f"{total_required} tok exceeds active model context limit ({active_limit} tok) "
                        f"by {overage} tok."
                    ),
                    remediation=(
                        "Switch compression strategy to 'structured' or 'extractive', "
                        "compact or reduce conversation history/notes, lower output_reservation_tokens, "
                        "or select a model/configuration with a larger context window."
                    ),
                )
            )

        # 3. Validate output-token reservation
        if output_tokens < 64:
            issues.append(
                ContextValidationIssue(
                    code="OUTPUT_RESERVATION_TOO_SMALL",
                    severity="error",
                    section="output_reservation",
                    message=f"Reserved output tokens ({output_tokens}) is below minimum viable floor (64).",
                    remediation="Increase output_reservation_tokens to at least 64 (recommended >= 256).",
                )
            )
        elif output_tokens >= active_limit:
            issues.append(
                ContextValidationIssue(
                    code="OUTPUT_RESERVATION_EXHAUSTS_WINDOW",
                    severity="error",
                    section="output_reservation",
                    message=(
                        f"Reserved output tokens ({output_tokens}) leaves no room inside context limit ({active_limit})."
                    ),
                    remediation="Reduce output_reservation_tokens below effective_context_limit - input_tokens.",
                )
            )
        elif output_tokens > package.capabilities.max_output_tokens:
            issues.append(
                ContextValidationIssue(
                    code="OUTPUT_RESERVATION_EXCEEDS_MODEL_MAX",
                    severity="error",
                    section="output_reservation",
                    message=(
                        f"Reserved output tokens ({output_tokens}) exceeds model maximum output tokens "
                        f"({package.capabilities.max_output_tokens})."
                    ),
                    remediation=(
                        f"Reduce output_reservation_tokens to <= {package.capabilities.max_output_tokens}."
                    ),
                )
            )

        # 4. Check missing or dangling source references
        refs_by_label = {ref.citation_label: ref for ref in package.source_references}
        if len(refs_by_label) != len(package.source_references):
            issues.append(
                ContextValidationIssue(
                    code="DUPLICATE_CITATION_LABELS",
                    severity="error",
                    section="source_references",
                    message="Duplicate [E#] citation labels detected in source_references table.",
                    remediation="Rebuild context package via ContextBuilder so citation labels stay unique.",
                )
            )

        for ref in package.source_references:
            if not ref.source_id or not ref.url or not ref.citation_label:
                issues.append(
                    ContextValidationIssue(
                        code="INCOMPLETE_SOURCE_REFERENCE",
                        severity="error",
                        section="source_references",
                        message=f"Source reference '{ref.citation_label or ref.evidence_id}' is missing source_id or url.",
                        remediation="Ensure every evidence passage carries valid source_id and HTTP(S) source URL.",
                    )
                )

        for sel in package.selected_evidence:
            if not sel.source_id or not sel.source_url:
                issues.append(
                    ContextValidationIssue(
                        code="MISSING_EVIDENCE_PROVENANCE",
                        severity="error",
                        section="selected_evidence",
                        message=f"Selected evidence '{sel.evidence_id}' lacks source_id or source_url.",
                        remediation="Attach complete source provenance before sending evidence to a model.",
                    )
                )
            if sel.citation_label not in refs_by_label:
                issues.append(
                    ContextValidationIssue(
                        code="MISSING_SOURCE_REFERENCE",
                        severity="error",
                        section="source_references",
                        message=(
                            f"Selected evidence '{sel.evidence_id}' ({sel.citation_label}) "
                            "is missing from source_references table."
                        ),
                        remediation="Include all selected evidence citations in source_references.",
                    )
                )

        for unit in package.compression.units:
            if not unit.provenance_verified:
                issues.append(
                    ContextValidationIssue(
                        code="COMPRESSION_PROVENANCE_UNVERIFIED",
                        severity="error",
                        section="compression",
                        message=f"Compressed unit '{unit.unit_id}' failed provenance verification.",
                        remediation="Use deterministic compression ('extractive', 'deduplication', or 'structured').",
                    )
                )
            if not unit.supporting_source_ids or not unit.supporting_source_urls:
                issues.append(
                    ContextValidationIssue(
                        code="COMPRESSED_UNIT_MISSING_PROVENANCE",
                        severity="error",
                        section="compression",
                        message=f"Compressed unit '{unit.unit_id}' lost supporting source IDs/URLs.",
                        remediation="Regenerate compression output preserving supporting_source_ids and URLs.",
                    )
                )
            for tag in unit.citation_labels:
                if tag not in refs_by_label:
                    issues.append(
                        ContextValidationIssue(
                            code="DANGLING_CITATION_TAG",
                            severity="error",
                            section="compression",
                            message=f"Compressed unit '{unit.unit_id}' references unknown citation {tag}.",
                            remediation="Ensure compressed text citations map cleanly to source_references.",
                        )
                    )

        # Verify no orphan [E#] tags appear inside rendered text
        for tag in set(_CITATION_RE.findall(package.rendered_user_prompt)):
            if tag not in refs_by_label:
                issues.append(
                    ContextValidationIssue(
                        code="UNRESOLVED_PROMPT_CITATION",
                        severity="error",
                        section="rendered_user_prompt",
                        message=f"Rendered prompt cites {tag}, which is absent from source_references.",
                        remediation="Re-run ContextBuilder or repair missing citation reference.",
                    )
                )

        # Check conflict pairs were not silently broken
        selected_ev_ids = {sel.evidence_id for sel in package.selected_evidence}
        for conflict in package.conflicting_claims:
            a_in = conflict.claim_a.evidence_id in selected_ev_ids
            b_in = conflict.claim_b.evidence_id in selected_ev_ids
            if a_in != b_in:
                issues.append(
                    ContextValidationIssue(
                        code="PARTIAL_CONFLICT_PAIR",
                        severity="error",
                        section="conflicting_claims",
                        message=(
                            f"Conflict '{conflict.conflict_id}' includes only one opposing side; "
                            "partial conflict framing risks biasing synthesis."
                        ),
                        remediation=(
                            "Allocate enough evidence budget (or use 'structured'/'extractive' compression) "
                            "to include both conflicting claims together."
                        ),
                    )
                )

        # 5. Check malformed context sections
        if not package.system_instructions.strip():
            issues.append(
                ContextValidationIssue(
                    code="MALFORMED_SECTION",
                    severity="error",
                    section="system_instructions",
                    message="system_instructions section is empty.",
                    remediation="Provide non-empty grounding system instructions.",
                )
            )
        if not package.original_user_question.strip():
            issues.append(
                ContextValidationIssue(
                    code="MALFORMED_SECTION",
                    severity="error",
                    section="original_user_question",
                    message="original_user_question section is empty.",
                    remediation="Supply a non-empty user research question.",
                )
            )
        if not package.research_objective.strip():
            issues.append(
                ContextValidationIssue(
                    code="MALFORMED_SECTION",
                    severity="error",
                    section="research_objective",
                    message="research_objective section is empty.",
                    remediation="Specify the research objective for this context package.",
                )
            )
        if not package.rendered_user_prompt.strip():
            issues.append(
                ContextValidationIssue(
                    code="MALFORMED_SECTION",
                    severity="error",
                    section="rendered_user_prompt",
                    message="rendered_user_prompt section is empty.",
                    remediation="Assemble rendered_user_prompt via ContextBuilder.",
                )
            )
        if package.selected_evidence and not package.rendered_evidence_block.strip():
            issues.append(
                ContextValidationIssue(
                    code="MALFORMED_SECTION",
                    severity="error",
                    section="rendered_evidence_block",
                    message="Selected evidence records exist but rendered_evidence_block is empty.",
                    remediation="Re-render compressed evidence units before inference.",
                )
            )

        # Warn cleanly if evidence existed in run/input yet none could fit (e.g. overly massive single doc with NONE compression)
        if not package.selected_evidence and package.excluded_evidence:
            issues.append(
                ContextValidationIssue(
                    code="NO_EVIDENCE_SELECTED_FOR_BUDGET",
                    severity="warning",
                    section="selected_evidence",
                    message=(
                        f"All {len(package.excluded_evidence)} available evidence passage(s) were excluded "
                        f"for budget '{package.budget_allocation.research_evidence_tokens}' tok."
                    ),
                    remediation=(
                        "Enable 'extractive' or 'structured' compression, or increase research evidence budget / context limit."
                    ),
                )
            )

        # 6. Verify configured budget compliance (never exceed configured input budget / safety margin)
        if measured_input > package.budget_allocation.input_budget_tokens:
            issues.append(
                ContextValidationIssue(
                    code="CONFIGURED_BUDGET_EXCEEDED",
                    severity="error",
                    section="budget_allocation",
                    message=(
                        f"Actual input tokens ({measured_input}) exceed allocated input budget "
                        f"({package.budget_allocation.input_budget_tokens}, after {package.budget_allocation.safety_margin_tokens} safety margin "
                        f"and {output_tokens} output reservation)."
                    ),
                    remediation=(
                        "Apply tighter compression ('structured' or 'extractive'), compact history/disable notes, "
                        "or adjust budget allocation/profile."
                    ),
                )
            )

        if strict_section_caps:
            for usage in package.section_token_usage:
                if usage.allocated_tokens > 0 and usage.estimated_tokens > usage.allocated_tokens:
                    # Only emit warning if total input budget is still healthy; error if significant overrun
                    over_pct = (usage.estimated_tokens - usage.allocated_tokens) / max(1, usage.allocated_tokens)
                    issues.append(
                        ContextValidationIssue(
                            code="SECTION_BUDGET_OVERRUN",
                            severity=(
                                "error"
                                if measured_input > package.budget_allocation.input_budget_tokens
                                and over_pct > 0.25
                                else "warning"
                            ),
                            section=usage.section,
                            message=(
                                f"Section '{usage.section}' uses {usage.estimated_tokens} tok vs "
                                f"allocated {usage.allocated_tokens} tok."
                            ),
                            remediation=(
                                f"Rebalance profile ratios or compress section '{usage.section}' to stay within {usage.allocated_tokens} tok."
                            ),
                        )
                    )

        has_errors = any(issue.severity == "error" for issue in issues)
        return ContextValidationReport(
            is_valid=not has_errors,
            estimated_input_tokens=measured_input,
            output_reservation_tokens=output_tokens,
            total_required_tokens=total_required,
            active_context_limit=active_limit,
            remaining_tokens=remaining,
            estimation_is_approximate=approx,
            issues=issues,
            checked_rules=checked_rules,
        )
