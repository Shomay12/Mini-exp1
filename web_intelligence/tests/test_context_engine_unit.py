"""Unit tests for Stage 5 Adaptive Context Management Engine core components."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from web_intelligence.app.context import (
    AllocationProfile,
    CompressionStrategy,
    ConservativeHeuristicEstimator,
    ContextAdapterConverter,
    ContextBudgetConfig,
    ContextCompressor,
    ContextConfigurationError,
    ContextLimitExceededError,
    ContextValidator,
    ConversationHistoryManager,
    DynamicContextBudgetManager,
    EvidenceSelector,
    ExpectedOutputLength,
    StructuredContextBuilder,
    TokenizerBackedEstimator,
    WorkspaceNote,
    normalize_model_context_capabilities,
    resolve_token_estimator,
)
from web_intelligence.app.schemas.documents import SourceClassification, SourceProvenance
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CorroborationSignals,
    CorroborationStatus,
    CoverageAnalysis,
    CoverageStatus,
    EvidenceItem,
    ExtractedClaim,
    RelevanceBreakdown,
    SubtopicCoverage,
    SubtopicStatus,
)
from web_intelligence.app.schemas.llm import (
    CapabilityProvenance,
    CapabilitySource,
    LLMInferenceResponse,
    ModelCapabilities,
    TokenUsage,
)


def _make_evidence(
    ev_id: str,
    *,
    source_id: str | None = None,
    domain: str = "example.org",
    url: str | None = None,
    text: str = "Sample evidence text about local-first web research.",
    relevance: float = 0.8,
    classification: SourceClassification = SourceClassification.PRIMARY,
    corroboration_status: CorroborationStatus = CorroborationStatus.SINGLE_SOURCE,
    independent_sources: int = 1,
    cluster: list[str] | None = None,
    conflict_ids: list[str] | None = None,
    published_at: datetime | None = None,
    section_heading: str | None = None,
) -> EvidenceItem:
    sid = source_id or f"src-{ev_id}"
    full_url = url or f"https://{domain}/articles/{ev_id}"
    return EvidenceItem(
        evidence_id=ev_id,
        source_id=sid,
        source_url=full_url,
        source_domain=domain,
        source_title=f"Title for {domain}",
        chunk_index=0,
        section_heading=section_heading,
        passage_text=text,
        char_start=0,
        char_end=len(text),
        token_estimate=max(1, len(text) // 4),
        relevance=RelevanceBreakdown(
            bm25_score=relevance,
            keyword_match_score=relevance,
            combined_score=relevance,
        ),
        provenance=SourceProvenance(
            domain=domain,
            source_classification=classification,
            published_at=published_at,
            original_url=full_url,
            canonical_url=full_url,
        ),
        corroboration=CorroborationSignals(
            corroboration_status=corroboration_status,
            independent_supporting_sources=independent_sources,
            supporting_source_ids=[sid],
            supporting_domains=[domain],
            conflicting_claim_ids=conflict_ids or [],
            same_underlying_source_cluster=cluster or [],
        ),
    )


def _make_caps(
    *,
    model_name: str = "local-test-model",
    max_ctx: int = 4096,
    max_out: int = 1024,
    ctx_source: CapabilitySource = CapabilitySource.VERIFIED,
    out_source: CapabilitySource = CapabilitySource.VERIFIED,
) -> ModelCapabilities:
    return ModelCapabilities(
        model_name=model_name,
        backend="ollama",
        max_context_tokens=max_ctx,
        recommended_output_tokens=max_out,
        supports_streaming=True,
        supports_json_mode=True,
        sources=CapabilityProvenance(
            context_length=ctx_source,
            max_output_tokens=out_source,
            json_mode=CapabilitySource.VERIFIED,
        ),
    )


class _AdapterWithTokenizer:
    provider = "mock-tok"

    def count_tokens(self, text: str) -> int:
        # Deterministic exact word-split test tokenizer
        return len(text.split()) if text.strip() else 0


class _AdapterWithoutTokenizer:
    provider = "mock-no-tok"


class _MockSummarizerAdapter:
    def __init__(self, response_text: str | Exception) -> None:
        self._response = response_text
        self.called = False

    async def chat(self, messages, *, temperature: float = 0.0, max_tokens: int | None = None):
        del messages, temperature, max_tokens
        self.called = True
        if isinstance(self._response, Exception):
            raise self._response
        return LLMInferenceResponse(
            content=self._response,
            model="mock-summarizer",
            usage=TokenUsage(total_tokens=40),
        )


def test_model_context_limits_never_inferred_from_name_and_tokenizer_detection():
    # Even with "128k" in the name, verified 2048 limit is honored verbatim.
    raw = _make_caps(model_name="llama-3.1-128k-q4", max_ctx=2048, max_out=512)
    caps_no_tok = normalize_model_context_capabilities(raw, adapter=_AdapterWithoutTokenizer())

    assert caps_no_tok.max_context_tokens == 2048
    assert caps_no_tok.effective_context_limit == 2048
    assert caps_no_tok.tokenizer_available is False
    assert caps_no_tok.estimation_is_approximate is True
    assert caps_no_tok.estimation_strategy == "conservative_heuristic"

    # Reject user ceiling above verified limit
    with pytest.raises(ContextLimitExceededError):
        normalize_model_context_capabilities(raw, user_context_limit=4096)

    # Exact tokenizer path
    est_exact = resolve_token_estimator(_AdapterWithTokenizer(), raw)
    assert isinstance(est_exact, TokenizerBackedEstimator)
    exact_result = est_exact.estimate("alpha beta gamma delta")
    assert exact_result.tokens == 4
    assert exact_result.is_approximate is False
    assert exact_result.tokenizer_available is True

    # Conservative heuristic path must explicitly flag approximate precision
    est_heur = resolve_token_estimator(_AdapterWithoutTokenizer(), raw)
    assert isinstance(est_heur, ConservativeHeuristicEstimator)
    approx_result = est_heur.estimate("alpha beta gamma delta")
    assert approx_result.tokens >= 4
    assert approx_result.is_approximate is True
    assert approx_result.tokenizer_available is False


def test_dynamic_allocation_profiles_and_invalid_configurations():
    mgr = DynamicContextBudgetManager()
    caps_8k = normalize_model_context_capabilities(_make_caps(max_ctx=8192, max_out=4096))
    caps_2k = normalize_model_context_capabilities(_make_caps(max_ctx=2048, max_out=512))

    quick = mgr.plan_budget(
        caps_8k,
        ContextBudgetConfig(profile=AllocationProfile.QUICK_ANSWER),
    )
    deep = mgr.plan_budget(
        caps_8k,
        ContextBudgetConfig(profile=AllocationProfile.DEEP_RESEARCH),
    )
    long_form = mgr.plan_budget(
        caps_8k,
        ContextBudgetConfig(
            profile=AllocationProfile.LONG_FORM,
            expected_output_length=ExpectedOutputLength.LONG,
        ),
    )
    small_window = mgr.plan_budget(
        caps_2k,
        ContextBudgetConfig(profile=AllocationProfile.STANDARD_RESEARCH),
    )

    assert deep.research_evidence_tokens > quick.research_evidence_tokens
    assert long_form.output_reservation_tokens > quick.output_reservation_tokens
    assert small_window.total_planned_tokens <= 2048
    assert deep.total_planned_tokens <= 8192

    # When available history is 0, budget dynamically frees slack toward evidence/metadata
    with_history = mgr.plan_budget(
        caps_8k,
        ContextBudgetConfig(profile=AllocationProfile.STANDARD_RESEARCH),
        available_history_tokens=900,
    )
    no_history = mgr.plan_budget(
        caps_8k,
        ContextBudgetConfig(profile=AllocationProfile.STANDARD_RESEARCH),
        available_history_tokens=0,
    )
    assert no_history.research_evidence_tokens > with_history.research_evidence_tokens

    # Invalid custom ratios & invalid budget configurations
    with pytest.raises(ValueError, match="negative"):
        ContextBudgetConfig(
            profile=AllocationProfile.CUSTOM,
            custom_ratios={"evidence": -0.5, "history": 1.0},
        )
    with pytest.raises(ContextLimitExceededError):
        mgr.plan_budget(
            caps_2k,
            ContextBudgetConfig(user_context_limit=4096),
        )
    with pytest.raises(ContextConfigurationError):
        mgr.plan_budget(
            caps_2k,
            ContextBudgetConfig(output_reservation_tokens=1900, safety_margin_tokens=300),
        )


def test_evidence_selector_preserves_conflicts_and_avoids_redundancy():
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    conflict = ClaimConflict(
        conflict_id="cf-1",
        topic="latency",
        entity="EngineX",
        attribute="latency_ms",
        conflict_type="numeric_mismatch",
        claim_a=ExtractedClaim(
            claim_id="cl-a",
            source_id="src-a",
            evidence_id="ev-a",
            source_url="https://alpha.example.org/bench",
            source_domain="alpha.example.org",
            entity="EngineX",
            attribute="latency_ms",
            statement_text="EngineX measured 42 ms median latency in 2025.",
            value_type="quantity",
            normalized_value="42",
            unit="ms",
        ),
        claim_b=ExtractedClaim(
            claim_id="cl-b",
            source_id="src-b",
            evidence_id="ev-b",
            source_url="https://beta.example.net/report",
            source_domain="beta.example.net",
            entity="EngineX",
            attribute="latency_ms",
            statement_text="EngineX measured 85 ms median latency under load in 2025.",
            value_type="quantity",
            normalized_value="85",
            unit="ms",
        ),
        summary="Alpha reports 42 ms vs Beta 85 ms.",
    )

    ev_a = _make_evidence(
        "ev-a",
        source_id="src-a",
        domain="alpha.example.org",
        text="EngineX measured 42 ms median latency in 2025 benchmarks across cold starts.",
        relevance=0.95,
        conflict_ids=["cf-1"],
        corroboration_status=CorroborationStatus.CONTESTED,
        published_at=now - timedelta(days=10),
    )
    # Lower relevance opposing claim must NOT be discarded when higher-scoring claims exist
    ev_b = _make_evidence(
        "ev-b",
        source_id="src-b",
        domain="beta.example.net",
        text="EngineX measured 85 ms median latency under production load in 2025.",
        relevance=0.35,
        conflict_ids=["cf-1"],
        corroboration_status=CorroborationStatus.CONTESTED,
        published_at=now - timedelta(days=12),
    )
    ev_dup = _make_evidence(
        "ev-dup",
        source_id="src-dup",
        domain="alpha.example.org",
        text="EngineX measured 42 ms median latency in 2025 benchmarks across cold starts cleanly.",
        relevance=0.92,
        cluster=["cluster-alpha"],
    )
    ev_high_non_conflict = _make_evidence(
        "ev-c",
        source_id="src-c",
        domain="gamma.edu",
        text="Gamma University verified deployment security isolation guarantees for local execution.",
        relevance=0.88,
        classification=SourceClassification.REFERENCE,
        published_at=now - timedelta(days=5),
    )

    # First ensure ev_a doesn't trigger redundancy against ev_dup; give cluster tag to ev_a too
    ev_a.corroboration.same_underlying_source_cluster = ["cluster-alpha"]
    coverage = CoverageAnalysis(
        subtopic_details=[
            SubtopicCoverage(subtopic="security isolation guarantees", status=SubtopicStatus.COVERED),
            SubtopicCoverage(subtopic="latency benchmarks", status=SubtopicStatus.PARTIAL),
        ],
        coverage_status=CoverageStatus.PARTIAL,
    )

    selector = EvidenceSelector()
    selected, excluded = selector.select(
        [ev_a, ev_dup, ev_high_non_conflict, ev_b],
        query="EngineX latency and security isolation",
        conflicts=[conflict],
        coverage=coverage,
        evidence_budget_tokens=300,
        reference_time=now,
    )

    selected_ids = {s.evidence_id for s in selected}
    assert "ev-a" in selected_ids
    assert "ev-b" in selected_ids  # contradictory source preserved despite 0.35 relevance
    assert all(s.source_url and s.citation_label.startswith("[E") for s in selected)
    assert any(ex.evidence_id == "ev-dup" and ex.exclusion_reason == "redundant_passage" for ex in excluded)


def test_dedup_and_structured_compression_preserve_independent_sources_and_provenance():
    compressor = ContextCompressor()
    selector = EvidenceSelector()

    ev1 = _make_evidence(
        "e1",
        source_id="s1",
        domain="nature.org",
        text="Global sea level rose 3.4 mm per year between 2010 and 2024 according to tide gauges.",
    )
    ev2 = _make_evidence(
        "e2",
        source_id="s2",
        domain="nasa.gov",
        text="Global sea level rose 3.4 mm per year between 2010 and 2024 in satellite altimetry records.",
    )
    ev3 = _make_evidence(
        "e3",
        source_id="s3",
        domain="arctic.io",
        text="Regional Baltic sea level dropped 1.2 mm per year due to post-glacial rebound.",
        conflict_ids=["cf-regional"],
    )

    selected = compressor.evidence_items_to_selected([ev1, ev2, ev3])
    assert len(selected) == 3

    dedup_res = compressor.compress(selected, strategy=CompressionStrategy.DEDUPLICATION)
    assert dedup_res.deduplicated_count == 1
    assert dedup_res.independent_sources_preserved == 3
    assert dedup_res.provenance_preserved is True
    # Merged unit must cite BOTH nature.org [E1] and nasa.gov [E2]
    merged_unit = next(u for u in dedup_res.units if "s1" in u.supporting_source_ids)
    assert "s2" in merged_unit.supporting_source_ids
    assert "nasa.gov" in merged_unit.supporting_domains
    assert "[E1]" in merged_unit.citation_labels and "[E2]" in merged_unit.citation_labels
    # Conflicted unit kept distinct
    assert any("e3" in u.original_evidence_ids for u in dedup_res.units)

    struct_res = compressor.compress(selected, strategy=CompressionStrategy.STRUCTURED)
    assert struct_res.strategy_applied == CompressionStrategy.STRUCTURED
    assert struct_res.provenance_preserved is True
    all_facts = [fact for u in struct_res.units for fact in u.structured_facts]
    assert any("2024" in f.dates or "2010" in f.dates for f in all_facts)
    assert any("3.4" in f.numbers for f in all_facts)
    assert all(f.source_refs and f.source_urls for f in all_facts)
    # Silence unused import if selector isn't called further
    assert selector is not None


async def test_model_assisted_compression_success_and_validation_fallback():
    compressor = ContextCompressor()
    ev = _make_evidence(
        "e100",
        domain="nist.gov",
        text="NIST SP 800-53 Revision 5 was published in September 2020 and defines 20 control families.",
    )
    selected = compressor.evidence_items_to_selected([ev])

    # 1. Valid local model summary preserving citation, date, and numbers
    good_adapter = _MockSummarizerAdapter(
        "[E1] NIST published SP 800-53 Revision 5 in 2020 with 20 control families."
    )
    ok_res = await compressor.compress_async(
        selected,
        strategy=CompressionStrategy.MODEL_ASSISTED,
        adapter=good_adapter,
        allow_model_assisted=True,
    )
    assert good_adapter.called is True
    assert ok_res.model_assisted_used is True
    assert ok_res.strategy_applied == CompressionStrategy.MODEL_ASSISTED
    assert ok_res.provenance_preserved is True

    # 2. Bad model summary alters number (20 -> 99) and drops citation -> rejected & falls back
    corrupt_adapter = _MockSummarizerAdapter(
        "NIST released an update defining 99 control families in some year."
    )
    reverted_res = await compressor.compress_async(
        selected,
        strategy=CompressionStrategy.MODEL_ASSISTED,
        adapter=corrupt_adapter,
        allow_model_assisted=True,
    )
    assert reverted_res.model_assisted_used is False
    assert reverted_res.strategy_applied == CompressionStrategy.STRUCTURED
    assert reverted_res.model_assisted_fallback_reason is not None
    assert "2020" in reverted_res.units[0].compressed_text
    assert "[E1]" in reverted_res.units[0].compressed_text

    # 3. Failing adapter exception -> falls back gracefully with redacted reason
    failing_adapter = _MockSummarizerAdapter(RuntimeError("local daemon timeout"))
    err_res = await compressor.compress_async(
        selected,
        strategy=CompressionStrategy.MODEL_ASSISTED,
        adapter=failing_adapter,
        allow_model_assisted=True,
    )
    assert err_res.model_assisted_used is False
    assert err_res.strategy_applied == CompressionStrategy.STRUCTURED
    assert "timeout" in (err_res.model_assisted_fallback_reason or "").lower()


def test_history_manager_compaction_preserves_decisions_questions_and_user_notes():
    hm = ConversationHistoryManager()
    turns = [
        hm.new_turn(
            role="user",
            content="Compare PostgreSQL vs SQLite for local offline storage.",
            scope="session",
        ),
        hm.new_turn(
            role="assistant",
            content="We decided to use SQLite WAL mode for single-user desktop persistence. Unresolved: What retention period should defaults use?",
            scope="session",
        ),
        hm.new_turn(
            role="user",
            content="Focus next on FTS5 indexing overhead.",
            scope="session",
        ),
        hm.new_turn(
            role="user",
            content="Verify exact pragma synchronous settings now.",
            scope="current_turn",
        ),
    ]
    enabled_note = WorkspaceNote(
        note_id="n1",
        title="Constraint",
        content="Zero cloud dependencies permitted.",
        enabled=True,
    )
    disabled_note = WorkspaceNote(
        note_id="n2",
        title="Archived",
        content="Ignore legacy v0 schema.",
        enabled=False,
    )

    pkg = hm.build_history_package(
        turns=turns,
        workspace_notes=[enabled_note, disabled_note],
        history_budget_tokens=90,
        max_history_turns=1,
    )

    assert pkg.compacted is True
    assert pkg.compacted_summary is not None
    assert len(pkg.current_turn_items) == 1
    assert any("SQLite WAL mode" in d for d in pkg.compacted_summary.preserved_decisions)
    assert any("retention period" in q for q in pkg.compacted_summary.preserved_unresolved_questions)
    assert [n.note_id for n in pkg.workspace_notes] == ["n1"]


async def test_builder_and_validator_empty_excessive_doc_overflow_and_provenance_checks():
    caps = normalize_model_context_capabilities(_make_caps(max_ctx=1024, max_out=256))
    builder = StructuredContextBuilder()
    validator = ContextValidator()

    # 1. Empty evidence package builds cleanly and passes validation
    empty_pkg = await builder.build_async(
        question="What is the status of project Helios?",
        capabilities=caps,
        budget_config=ContextBudgetConfig(profile=AllocationProfile.QUICK_ANSWER),
        evidence_items=[],
    )
    empty_report = validator.validate(empty_pkg)
    assert empty_report.is_valid is True
    assert empty_pkg.selected_evidence == []

    # 2. Excessively large document under CompressionStrategy.NONE is NOT silently truncated
    huge_text = " ".join(["quantum telemetry protocol calibration"] * 350)  # ~1750+ tokens
    huge_ev = _make_evidence("ev-huge", domain="bigdoc.org", text=huge_text)
    uncompressed_pkg = await builder.build_async(
        question="Summarize quantum calibration.",
        capabilities=caps,
        budget_config=ContextBudgetConfig(
            profile=AllocationProfile.STANDARD_RESEARCH,
            compression_strategy=CompressionStrategy.NONE,
        ),
        evidence_items=[huge_ev],
    )
    assert uncompressed_pkg.selected_evidence == []
    assert len(uncompressed_pkg.excluded_evidence) == 1
    assert uncompressed_pkg.excluded_evidence[0].exclusion_reason == "single_passage_exceeds_evidence_budget"
    uncompressed_report = validator.validate(uncompressed_pkg)
    assert any(i.code == "NO_EVIDENCE_SELECTED_FOR_BUDGET" for i in uncompressed_report.issues)

    # 3. Same oversized passage with Extractive compression fits budget cleanly preserving citation
    extracted_pkg = await builder.build_async(
        question="Summarize quantum calibration.",
        capabilities=caps,
        budget_config=ContextBudgetConfig(
            profile=AllocationProfile.STANDARD_RESEARCH,
            compression_strategy=CompressionStrategy.EXTRACTIVE,
        ),
        evidence_items=[huge_ev],
    )
    assert len(extracted_pkg.selected_evidence) == 1
    assert extracted_pkg.total_with_output_tokens <= 1024
    ext_report = validator.validate(extracted_pkg)
    assert ext_report.is_valid is True

    # 4. Adapter conversion sets max_tokens to output_reservation_tokens
    req = ContextAdapterConverter.to_inference_request(extracted_pkg)
    assert req.max_tokens == extracted_pkg.output_reservation_tokens
    assert len(req.messages) == 2

    # 5. Context overflow + missing source reference + malformed section caught deterministically
    broken = extracted_pkg.model_copy(
        update={
            "source_references": [],
            "system_instructions": "   ",
            "total_input_tokens": 5000,
        }
    )
    broken_report = validator.validate(broken)
    assert broken_report.is_valid is False
    codes = {i.code for i in broken_report.issues}
    assert "CONTEXT_LIMIT_EXCEEDED" in codes
    assert "MISSING_SOURCE_REFERENCE" in codes
    assert "MALFORMED_SECTION" in codes
    assert "CONFIGURED_BUDGET_EXCEEDED" in codes
