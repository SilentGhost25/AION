"""
AION Production Hardening Unit Test Suite
==========================================
Tests all 11 parts of the Production Hardening Specification.
"""

import pytest
from pathlib import Path
from core.sse.events import SSEEvent, SSEEventType, make_success_event, make_failure_event
from core.sse.stream_manager import SSEStreamManager
from core.artifacts.lifecycle import (
    ArtifactStatus, ArtifactStatusTransition, GenerationGuard, IllegalStatusTransitionError
)
from core.artifacts.manifest import DocumentManifest, SourceArtifact
from core.artifacts.store import ArtifactStore
from core.extraction.exceptions import ExtractionHardStop, HardStopCode
from core.evidence.budget import StratifiedValidationBudget
from core.extraction.module_segmenter import ModuleSegmenter, SegmentationStrategy
from core.generation.qwen_warmup import QwenWarmupPolicy
from core.validators.question_validation_pipeline import QuestionValidationPipeline
from core.config.dependency_versions import log_startup_versions
from core.artifacts.cache import build_cache_key, is_cache_valid
from core.evidence.deduplication import EvidenceDeduplicator
from core.contracts.final_paper import FinalPaperIR, FinalQuestion, ORPair, QuestionSegment


def test_sse_event_serialization():
    evt = SSEEvent(event=SSEEventType.CONNECTED, data={"msg": "hello"}, id="evt_1")
    serialized = evt.serialize()
    assert "id: evt_1" in serialized
    assert "event: connected" in serialized
    assert 'data: {"msg": "hello"}' in serialized

    succ = make_success_event(paper_id="p1", qa_score=0.95)
    assert succ.event == SSEEventType.DONE
    assert succ.data["status"] == "SUCCESS"

    fail = make_failure_event(code="ERR_1", stage="extraction", message="Failed", recoverable=False)
    assert fail.event == SSEEventType.PIPELINE_ERROR
    assert fail.data["status"] == "FAILED"


def test_sse_stream_manager():
    def dummy_gen():
        yield {"event": "progress", "percent": 50}
        return {"paper_id": "p_dummy", "qa_score": 0.9}

    stream = list(SSEStreamManager.run_generator("req_100", dummy_gen))
    assert len(stream) >= 3
    assert "event: connected" in stream[0]
    assert "event: progress" in stream[1]
    assert "event: done" in stream[-2] or "event: done" in stream[-1]


def test_artifact_lifecycle_transitions(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.5\nTest\n%EOF\n")

    manifest = store.store_from_temp(str(pdf_file), "test.pdf", document_id="doc_lh_1")
    assert manifest.status == ArtifactStatus.UPLOADED

    # Test valid transitions: UPLOADED -> VALIDATING -> EXTRACTING -> EVIDENCE_VALIDATED -> READY
    manifest = ArtifactStatusTransition.transition(manifest, ArtifactStatus.VALIDATING, store=store)
    assert manifest.status == ArtifactStatus.VALIDATING

    manifest = ArtifactStatusTransition.transition(manifest, ArtifactStatus.EXTRACTING, store=store)
    assert manifest.status == ArtifactStatus.EXTRACTING

    manifest = ArtifactStatusTransition.transition(manifest, ArtifactStatus.EVIDENCE_VALIDATED, store=store)
    assert manifest.status == ArtifactStatus.EVIDENCE_VALIDATED

    manifest = ArtifactStatusTransition.transition(manifest, ArtifactStatus.READY, store=store)
    assert manifest.status == ArtifactStatus.READY

    # Test GenerationGuard on READY vs non-READY
    guard_ready = GenerationGuard.check("doc_lh_1", store=store)
    assert guard_ready.allowed is True

    # Test illegal transition: READY -> UPLOADED
    with pytest.raises(IllegalStatusTransitionError):
        ArtifactStatusTransition.transition(manifest, ArtifactStatus.UPLOADED, store=store)


def test_generation_guard_not_ready(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    pdf_file = tmp_path / "test2.pdf"
    pdf_file.write_bytes(b"%PDF-1.5\nTest2\n%EOF\n")

    manifest = store.store_from_temp(str(pdf_file), "test2.pdf", document_id="doc_lh_2")
    # Manifest is UPLOADED (not READY)
    guard = GenerationGuard.check("doc_lh_2", store=store)
    assert guard.allowed is False
    assert guard.code == "ARTIFACT_NOT_READY"


def test_extraction_hard_stop_exception():
    hs = ExtractionHardStop(
        code=HardStopCode.TXT_AS_SOURCE_REJECTED,
        stage="extraction",
        message="TXT source rejected",
        recoverable=False
    )
    assert hs.code == "TXT_AS_SOURCE_REJECTED"
    assert hs.stage == "extraction"
    assert hs.recoverable is False


def test_stratified_validation_budget():
    class DummyChunk:
        def __init__(self, cid, mod, ct, conf=0.9):
            self.chunk_id = cid
            self.module_id = mod
            self.content_type = ct
            self.confidence = conf

    chunks = [DummyChunk(f"c_{i}", i % 5 + 1, "TEXT" if i % 2 == 0 else "EQUATION") for i in range(1000)]
    budget = StratifiedValidationBudget.compute(chunks, modules=[1, 2, 3, 4, 5], content_types=["TEXT", "EQUATION"])

    assert budget.total_chunks == 1000
    assert len(budget.selected_chunks) >= 300
    assert budget.stratified is True


def test_module_segmenter_hierarchy():
    class DummyBlock:
        def __init__(self, text, page):
            self.text = text
            self.page = page

    blocks = [
        DummyBlock("Module 1: Introduction to AI", page=1),
        DummyBlock("Module 2: Search Algorithms", page=10),
        DummyBlock("Module 3: Knowledge Representation", page=20),
    ]

    class DummyArtifact:
        text_blocks = blocks
        page_count = 30

    segments = ModuleSegmenter.segment(DummyArtifact())
    assert len(segments) == 3
    assert segments[0].strategy_used == SegmentationStrategy.EXPLICIT_MODULE_HEADING
    assert segments[0].confidence == 0.98


def test_qwen_warmup_policy():
    policy = QwenWarmupPolicy(qwen_loaded_permanently=True)
    assert policy.check_and_warmup(evidence_gate_passed=True) is True

    policy_ondemand = QwenWarmupPolicy(qwen_loaded_permanently=False)
    assert policy_ondemand.check_and_warmup(evidence_gate_passed=False) is False


def test_question_validation_pipeline():
    class DummyIntent:
        bloom = "REMEMBER"
        bloom_verb = "Define"
        marks = 10

    class DummyGen:
        question_text = "Define Artificial Intelligence and explain its foundational goals."
        marks = 10

    result = QuestionValidationPipeline.validate(DummyGen(), DummyIntent())
    assert result.status == "APPROVED"

    class BadGen:
        question_text = "Create between artificial intelligence..."
        marks = 10

    bad_result = QuestionValidationPipeline.validate(BadGen(), DummyIntent())
    assert bad_result.status == "REJECTED"
    assert bad_result.reason == "FORBIDDEN_PHRASE"


def test_cache_versioning_and_deduplication(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    pdf_file = tmp_path / "test3.pdf"
    pdf_file.write_bytes(b"%PDF-1.5\nTest3\n%EOF\n")

    manifest = store.store_from_temp(str(pdf_file), "test3.pdf", document_id="doc_cache_1")
    key = build_cache_key(manifest)
    assert len(key.compute_key()) == 64

    class DummyChunk:
        def __init__(self, text, page=1, bbox=None, conf=0.9):
            self.text = text
            self.page_start = page
            self.bbox = bbox
            self.confidence = conf

    chunks = [
        DummyChunk("Same duplicate text", page=1),
        DummyChunk("Same duplicate text", page=1),
        DummyChunk("Unique text block", page=1),
    ]

    deduped = EvidenceDeduplicator.deduplicate(chunks)
    assert len(deduped) == 2


def test_final_paper_ir():
    q1 = FinalQuestion(
        question_id="q1",
        question_no=1,
        sub_label="a",
        module_id=1,
        marks=10,
        bloom="REMEMBER",
        co="CO1",
        question_type="DESCRIPTIVE",
        status="APPROVED",
        segments=[QuestionSegment(segment_type="text", value="Define AI.")],
    )
    q2 = FinalQuestion(
        question_id="q2",
        question_no=1,
        sub_label="b",
        module_id=1,
        marks=10,
        bloom="UNDERSTAND",
        co="CO1",
        question_type="DESCRIPTIVE",
        status="APPROVED",
        segments=[QuestionSegment(segment_type="text", value="Explain Search.")],
    )
    or_pair = ORPair(module_id=1, alt_a=q1, alt_b=q2, mark_distribution=(10, 10))
    assert or_pair.parity_valid() is True

    paper = FinalPaperIR(
        paper_id="p_001",
        request_id="req_001",
        or_pairs=[or_pair],
        qa_status="PASS"
    )
    assert paper.is_exportable() is True
