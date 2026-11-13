"""Comprehensive tests for Stage 6 InferenceExecutor, AnswerGenerator, and LongFormReportGenerator."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from web_intelligence.app.context.engine import ContextManagementEngine
from web_intelligence.app.generation.answer_generator import AnswerGenerator
from web_intelligence.app.generation.executor import (
    ContextBudgetBlockedError,
    InferenceExecutor,
    ModelNotConfiguredError,
    ProviderAuthFailedError,
    ProviderResponseInvalidError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from web_intelligence.app.llm.manager import LLMConnectionManager
from web_intelligence.app.llm.model_registry import ModelRegistry
from web_intelligence.app.schemas.documents import (
    ExtractionStatus,
    SourceProvenance,
    SourceRecord,
)
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
from web_intelligence.app.schemas.generation import (
    GenerationFormat,
    GenerationLength,
    GenerationMode,
    GenerationRequest,
    GenerationStatus,
    LongFormReportConfig,
    PromptTemplateType,
)
from web_intelligence.app.schemas.llm import LLMConnectionCreate, LLMConnectionUpdate
from web_intelligence.app.schemas.research import ResearchMetadata, ResearchPackage, ResearchStatus
from web_intelligence.app.storage.database import DatabaseManager
from web_intelligence.app.storage.repositories import ResearchRepository


async def _seed_research(repo: ResearchRepository) -> ResearchPackage:
    now = datetime.now(timezone.utc)
    src1 = SourceRecord(
        source_id="s-fts",
        original_url="https://docs.example.org/fts5",
        canonical_url="https://docs.example.org/fts5",
        final_url="https://docs.example.org/fts5",
        domain="docs.example.org",
        title="SQLite FTS5 Guide",
        main_content="SQLite FTS5 uses BM25 natively.",
        content_hash="hash-1",
        extraction_status=ExtractionStatus.SUCCESS,
        retrieved_at=now,
    )
    src2 = SourceRecord(
        source_id="s-llm",
        original_url="https://perf.example.net/local-bench",
        canonical_url="https://perf.example.net/local-bench",
        final_url="https://perf.example.net/local-bench",
        domain="perf.example.net",
        title="Local Small LLM Benchmarks",
        main_content="Qwen3-4B achieves 64 tok/s locally with Q4_K_M quantization.",
        content_hash="hash-2",
        extraction_status=ExtractionStatus.SUCCESS,
        retrieved_at=now,
    )
    ev1 = EvidenceItem(
        evidence_id="ev-1",
        source_id="s-fts",
        source_url=src1.original_url,
        source_domain=src1.domain,
        source_title=src1.title,
        chunk_index=0,
        section_heading="Indexing & Retrieval",
        passage_text="SQLite FTS5 builds inverted indexes and computes BM25 scores locally in milliseconds.",
        char_start=0,
        char_end=88,
        token_estimate=18,
        relevance=RelevanceBreakdown(bm25_score=0.92, combined_score=0.92),
        corroboration=CorroborationSignals(
            corroboration_status=CorroborationStatus.CORROBORATED,
            independent_supporting_sources=2,
            supporting_source_ids=["s-fts", "s-llm"],
        ),
        provenance=SourceProvenance(
            original_url=src1.original_url,
            canonical_url=src1.canonical_url,
            domain=src1.domain,
        ),
    )
    ev2 = EvidenceItem(
        evidence_id="ev-2",
        source_id="s-llm",
        source_url=src2.original_url,
        source_domain=src2.domain,
        source_title=src2.title,
        chunk_index=0,
        section_heading="Local Inference Speed",
        passage_text="Qwen3-4B quantized to Q4_K_M sustains 64 tokens per second on desktop unified memory.",
        char_start=0,
        char_end=85,
        token_estimate=18,
        relevance=RelevanceBreakdown(bm25_score=0.88, combined_score=0.88),
        corroboration=CorroborationSignals(
            corroboration_status=CorroborationStatus.SINGLE_SOURCE,
            independent_supporting_sources=1,
            supporting_source_ids=["s-llm"],
        ),
        provenance=SourceProvenance(
            original_url=src2.original_url,
            canonical_url=src2.canonical_url,
            domain=src2.domain,
        ),
    )
    c1 = ExtractedClaim(
        claim_id="cl-1",
        source_id="s-fts",
        evidence_id="ev-1",
        source_url=src1.original_url,
        source_domain=src1.domain,
        entity="Qwen3-4B",
        attribute="Memory Footprint",
        value_type="quantity",
        normalized_value="2.8 GB",
        statement_text="CPU runtime footprint measures 2.8 GB RAM.",
    )
    c2 = ExtractedClaim(
        claim_id="cl-2",
        source_id="s-llm",
        evidence_id="ev-2",
        source_url=src2.original_url,
        source_domain=src2.domain,
        entity="Qwen3-4B",
        attribute="Memory Footprint",
        value_type="quantity",
        normalized_value="3.4 GB",
        statement_text="Metal GPU offload footprint measures 3.4 GB unified memory.",
    )
    conflict = ClaimConflict(
        conflict_id="conf-1",
        topic="Memory Footprint",
        entity="Qwen3-4B",
        attribute="Memory Footprint",
        conflict_type="numeric_mismatch",
        claim_a=c1,
        claim_b=c2,
        summary="2.8 GB CPU vs 3.4 GB Metal footprint.",
    )
    pkg = ResearchPackage(
        research_id="res-gen-1",
        original_query="How fast and memory-efficient is local FTS5 + Qwen3-4B research?",
        research_status=ResearchStatus.COMPLETED,
        sources=[src1, src2],
        evidence=[ev1, ev2],
        claims=[c1, c2],
        conflicts=[conflict],
        coverage=CoverageAnalysis(
            covered_topics=["Indexing & Retrieval", "Local Inference Speed"],
            partial_topics=["Memory Footprint"],
            missing_topics=["Battery impact on laptops"],
            subtopic_details=[
                SubtopicCoverage(
                    subtopic="Indexing & Retrieval",
                    status=SubtopicStatus.COVERED,
                    supporting_evidence_ids=["ev-1"],
                    supporting_source_ids=["s-fts"],
                    independent_domain_count=1,
                ),
                SubtopicCoverage(
                    subtopic="Local Inference Speed",
                    status=SubtopicStatus.COVERED,
                    supporting_evidence_ids=["ev-2"],
                    supporting_source_ids=["s-llm"],
                    independent_domain_count=1,
                ),
            ],
            coverage_status=CoverageStatus.PARTIAL,
        ),
        metadata=ResearchMetadata(retrieved_at=now, retrieved_sources=2, evidence_items=2),
    )
    await repo.save_research_package(pkg)
    return pkg


async def _wire_stack(
    tmp_path: Path, handler
) -> tuple[ResearchRepository, LLMConnectionManager, ModelRegistry, AnswerGenerator]:
    db = DatabaseManager(tmp_path / "gen.db")
    await db.initialize()
    repo = ResearchRepository(db)
    registry = ModelRegistry(repo)
    manager = LLMConnectionManager(
        repo,
        model_registry=registry,
        transport=httpx.MockTransport(handler),
    )
    await manager.initialize()
    ctx_engine = ContextManagementEngine(repository=repo, llm_manager=manager)
    executor = InferenceExecutor(manager, registry)
    generator = AnswerGenerator(
        repository=repo,
        context_engine=ctx_engine,
        executor=executor,
    )
    return repo, manager, registry, generator


@pytest.mark.asyncio
async def test_answer_generator_standard_json_fallback_truncation_and_omitted_usage(
    tmp_path: Path,
) -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode()) if request.content else {}
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "Qwen3-4B",
                            "max_model_len": 8192,
                            "supported_parameters": ["response_format"],
                        }
                    ]
                },
            )
        seen_payloads.append(body)
        # First JSON request simulates endpoint rejecting response_format -> graceful fallback
        if body.get("response_format"):
            return httpx.Response(400, json={"error": {"message": "response_format not supported"}})
        if "VALID JSON ONLY" in body["messages"][0]["content"]:
            return httpx.Response(
                200,
                json={
                    "model": "Qwen3-4B",
                    "choices": [
                        {
                            "message": {
                                "content": '{"answer_summary": "Fast local pipeline [S1]", "findings": ["64 tok/s [S2]"]}'
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    # Intentionally omit usage to verify we NEVER invent usage counts!
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "Qwen3-4B",
                "choices": [
                    {
                        "message": {
                            "content": "SQLite FTS5 indexes evidence locally with BM25 ranking [S1], while Qwen3-4B streams at 64 tokens per second [S2]."
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 210, "completion_tokens": 32, "total_tokens": 242},
            },
        )

    repo, manager, _reg, generator = await _wire_stack(tmp_path, handler)
    pkg = await _seed_research(repo)

    # Without any connection -> ModelNotConfiguredError
    with pytest.raises(ModelNotConfiguredError):
        await generator.generate(GenerationRequest(research_id=pkg.research_id))

    conn = await manager.create_connection(
        LLMConnectionCreate(
            name="vLLM",
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            model="Qwen3-4B",
            context_length=8192,
            max_output_tokens=1024,
            activate=True,
        )
    )
    await manager.test_connection(conn.id)

    # Standard generation with finish_reason="length" -> truncated=True, real usage preserved
    res = await generator.generate(
        GenerationRequest(
            research_id=pkg.research_id,
            template_type=PromptTemplateType.QUICK_ANSWER,
            output_format=GenerationFormat.MARKDOWN,
            answer_length=GenerationLength.SHORT,
        )
    )
    assert res.status == GenerationStatus.COMPLETED
    assert res.truncated is True
    assert res.finish_reason == "length"
    assert res.usage is not None
    assert res.usage.total_tokens == 242
    assert res.citation_validation is not None
    assert res.citation_validation.passed is True
    assert [c.citation_id for c in res.citations] == ["S1", "S2"]
    assert res.context_stats is not None
    assert res.context_stats.evidence_items_included == 2

    loaded = await generator.get_generation(res.generation_id)
    assert loaded is not None
    assert loaded.content == res.content

    # JSON output format -> triggers response_format fallback & keeps usage=None honestly
    json_res = await generator.generate(
        GenerationRequest(
            research_id=pkg.research_id,
            output_format=GenerationFormat.JSON,
        )
    )
    assert json_res.parsed_json == {
        "answer_summary": "Fast local pipeline [S1]",
        "findings": ["64 tok/s [S2]"],
    }
    assert json_res.usage is None
    assert any("rejected response_format=json_object" in w for w in json_res.warnings)


@pytest.mark.asyncio
async def test_streaming_and_cancellation_preserves_partial_response(tmp_path: Path) -> None:
    gate = asyncio.Event()

    class SlowSSEStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"model":"qwen","choices":[{"index":0,"delta":{"content":"First clause [S1] "}}]}\n\n'
            gate.set()
            await asyncio.sleep(5)
            yield b'data: {"model":"qwen","choices":[{"index":0,"delta":{"content":"never reached"}}]}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                stream=SlowSSEStream(),
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(200, json={"data": [{"id": "qwen", "max_model_len": 8192}]})

    repo, manager, _reg, generator = await _wire_stack(tmp_path, handler)
    pkg = await _seed_research(repo)
    await manager.create_connection(
        LLMConnectionCreate(
            name="SlowStream",
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            model="qwen",
            context_length=8192,
            max_output_tokens=512,
            streaming=True,
            activate=True,
        )
    )

    events: list[dict] = []
    gen_id: str | None = None

    async def runner() -> None:
        nonlocal gen_id
        async for ev in generator.stream_generate(
            GenerationRequest(research_id=pkg.research_id, stream=True),
            generation_id="gen-cancel-test",
        ):
            events.append(ev)
            if ev["event"] == "start":
                gen_id = ev["generation_id"]

    task = asyncio.create_task(runner())
    await asyncio.wait_for(gate.wait(), timeout=2.0)
    await asyncio.sleep(0.05)
    cancelled_state = await generator.cancel_generation("gen-cancel-test")
    assert cancelled_state is not None
    await asyncio.wait_for(task, timeout=2.0)

    cancel_ev = next(e for e in events if e["event"] == "cancelled")
    assert cancel_ev["generation"]["status"] == "cancelled"
    assert cancel_ev["generation"]["partial"] is True
    assert cancel_ev["generation"]["content"] == "First clause [S1] "

    persisted = await generator.get_generation("gen-cancel-test")
    assert persisted is not None
    assert persisted.status == GenerationStatus.CANCELLED
    assert persisted.partial is True
    assert persisted.content == "First clause [S1] "
    assert [c.citation_id for c in persisted.citations] == ["S1"]


@pytest.mark.asyncio
async def test_error_normalization_and_non_streaming_honest_fallback(tmp_path: Path) -> None:
    mode = "ok"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen", "max_model_len": 8192}]})
        if mode == "auth":
            return httpx.Response(401, json={"error": "invalid api token sk-secret-123"})
        if mode == "connect_err":
            raise httpx.ConnectError("Connection refused")
        if mode == "timeout":
            raise httpx.ReadTimeout("Timed out")
        if mode == "malformed":
            return httpx.Response(200, content=b"not json at all")
        return httpx.Response(
            200,
            json={
                "model": "qwen",
                "choices": [{"message": {"content": "Non-streaming grounded answer [S1]."}, "finish_reason": "stop"}],
            },
        )

    repo, manager, _reg, generator = await _wire_stack(tmp_path, handler)
    pkg = await _seed_research(repo)
    conn = await manager.create_connection(
        LLMConnectionCreate(
            name="TestServer",
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            api_key=SecretStr("sk-secret-123"),
            model="qwen",
            context_length=8192,
            max_output_tokens=512,
            streaming=False,
            activate=True,
        )
    )

    # Streaming disabled -> honest single-response fallback without fake chunks
    stream_events = [
        ev
        async for ev in generator.stream_generate(
            GenerationRequest(research_id=pkg.research_id, stream=True)
        )
    ]
    assert len([e for e in stream_events if e["event"] == "delta"]) == 1
    assert any(
        e["event"] == "warning" and "streaming disabled" in e["warning"] for e in stream_events
    )

    mode = "auth"
    with pytest.raises(ProviderAuthFailedError) as auth_exc:
        await generator.generate(GenerationRequest(research_id=pkg.research_id))
    assert "sk-secret-123" not in str(auth_exc.value)

    mode = "connect_err"
    with pytest.raises(ProviderUnavailableError):
        await generator.generate(GenerationRequest(research_id=pkg.research_id))

    mode = "timeout"
    with pytest.raises(ProviderTimeoutError):
        await generator.generate(GenerationRequest(research_id=pkg.research_id))

    mode = "malformed"
    with pytest.raises(ProviderResponseInvalidError):
        await generator.generate(GenerationRequest(research_id=pkg.research_id))

    # Oversized requested max_output_tokens leaving < input prompt tokens fails cleanly without silent drops
    mode = "ok"
    await manager.update_connection(
        conn.id, LLMConnectionUpdate(context_length=512, max_output_tokens=450)
    )
    with pytest.raises(ContextBudgetBlockedError):
        await generator.generate(
            GenerationRequest(
                research_id=pkg.research_id,
                max_output_tokens=450,
            )
        )


@pytest.mark.asyncio
async def test_long_form_report_generator_multi_stage_repetition_and_conflicts(
    tmp_path: Path,
) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen-long", "max_model_len": 8192}]})
        req_body = json.loads(request.content.decode()) if request.content else {}
        call_count += 1
        # Emit deliberately identical sentence in sections 1 & 2 to exercise repetition detection!
        repeated_sentence = (
            "SQLite FTS5 computes BM25 relevance rankings across retrieved local passages [S1]."
        )
        if call_count == 1:
            body_text = f"{repeated_sentence} Executive synthesis confirms fast indexing [S1]."
            finish = "stop"
        elif call_count == 2:
            body_text = f"{repeated_sentence} CPU footprint is reported as 2.8 GB [S1]."
            finish = "stop"
        elif call_count == 3:
            body_text = "Quantized Metal throughput sustains 64 tokens per second with 3.4 GB footprint [S2]."
            finish = "length"
        else:
            body_text = "Sources report 2.8 GB on CPU [S1] versus 3.4 GB on Metal [S2], while battery impact remains unconfirmed."
            finish = "stop"
        if req_body.get("stream"):
            mid = len(body_text) // 2
            chunks = [
                f'data: {json.dumps({"model": "qwen-long", "choices": [{"index": 0, "delta": {"content": body_text[:mid]}}]})}\n\n',
                f'data: {json.dumps({"model": "qwen-long", "choices": [{"index": 0, "delta": {"content": body_text[mid:]}, "finish_reason": finish}], "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}})}\n\n',
                "data: [DONE]\n\n",
            ]
            return httpx.Response(
                200,
                content="".join(chunks).encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "model": "qwen-long",
                "choices": [{"message": {"content": body_text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
            },
        )

    repo, manager, _reg, generator = await _wire_stack(tmp_path, handler)
    pkg = await _seed_research(repo)
    await manager.create_connection(
        LLMConnectionCreate(
            name="LongFormvLLM",
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            model="qwen-long",
            context_length=8192,
            max_output_tokens=1024,
            streaming=True,
            activate=True,
        )
    )

    events = [
        ev
        async for ev in generator.stream_generate(
            GenerationRequest(
                research_id=pkg.research_id,
                generation_mode=GenerationMode.LONG_FORM_REPORT,
                template_type=PromptTemplateType.RESEARCH_REPORT,
                long_form_config=LongFormReportConfig(
                    max_sections=4,
                    section_max_output_tokens=256,
                    include_executive_summary=True,
                    include_contradiction_section=True,
                    detect_repetition=True,
                    detect_contradictions=True,
                ),
            )
        )
    ]

    complete_ev = next((e for e in events if e["event"] == "complete"), None)
    assert complete_ev is not None, f"Missing complete event; got events={[(e.get('event'), e.get('detail')) for e in events]}"
    report_res = complete_ev["result_obj"]
    assert report_res.mode == GenerationMode.LONG_FORM_REPORT
    assert report_res.long_form is not None
    assert len(report_res.long_form.outline) == 4
    assert len(report_res.long_form.sections) == 4
    assert report_res.truncated is True  # section 3 hit finish_reason="length"
    assert len(report_res.long_form.repetition_warnings) >= 1
    assert len(report_res.long_form.contradiction_warnings) >= 1
    assert report_res.usage is not None
    assert report_res.usage.prompt_tokens == 400
    assert report_res.usage.completion_tokens == 100
    assert report_res.usage.total_tokens == 500
    assert [c.citation_id for c in report_res.citations] == ["S1", "S2"]
