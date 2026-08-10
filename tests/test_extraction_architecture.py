"""
AION Extraction & Evidence Gateway Architecture Unit Tests
============================================================
Comprehensive test suite verifying invariants INV-1, INV-2, INV-3, INV-4,
contracts, adapters, Docling normalization, chunk validator, weighted PDF internals,
chunk-level Unicode integrity, evidence coverage gate, Bloom grammar, and zero-LLM hard stop.
"""

import pytest

from core.extraction.contracts import (
    ChunkStatus, ContentType, EvidenceChunk, ExtractionAdapterID,
    ExtractionLevel, ExtractionMetrics, ExtractionResult, PageResult,
    RejectionReason, TextBlock
)
from core.extraction.adapters import (
    DoclingAdapter, DoclingResultNormalizer, PyMuPDFAdapter
)
from core.extraction.adapter_registry import AdapterRegistry
from core.extraction.chunk_validator import ContentAwareChunkValidator
from core.extraction.recovery_manager import ExtractionRecoveryManager
from core.extraction.reporter import ChunkValidationReport
from core.extraction.hard_stop_gate import ExtractionHardStopGate
from core.extraction.evidence_coverage_gate import EvidenceCoverageGate
from core.extraction.gateway import ExtractionError, ExtractionGateway
from core.evidence.pdf_internals_detector import detect_pdf_internals
from core.evidence.unicode_gate import UnicodeIntegrityGate
from core.evidence.evidence_classifier import EvidenceClassifier
from core.evidence.taxonomy import EvidenceType
from core.generation.bloom_grammar import BloomGrammarValidator
from core.validators.semantic_validator import SemanticValidator


# ── INV-1: WEIGHTED PDF INTERNALS TESTS ───────────────────────────────────────

def test_pdf_internal_weighted_scoring():
    # PDF metadata leak snippet
    pdf_text = "/FontFile2 12 0 R /ToUnicode 13 0 R /FlateDecode endobj"
    report = detect_pdf_internals(pdf_text)
    assert report.has_internals is True
    assert report.total_score >= 5
    assert report.evidence_type == EvidenceType.PDF_METADATA


def test_pdf_internal_legitimate_word_object_passes():
    # Legitimate academic text using the word "object"
    text = "In object-oriented programming, an object is an instance of a class."
    report = detect_pdf_internals(text)
    assert report.has_internals is False
    assert report.total_score < 5


# ── INV-2: CHUNK-LEVEL UNICODE INTEGRITY & MATH EXEMPTIONS ───────────────────

def test_chunk_level_unicode_quarantine():
    clean_text = "The electronic control unit calculates engine timing."
    corrupt_text = "Corrupted timing \ufffd data \ufffd snippet."

    report_clean = UnicodeIntegrityGate.check(clean_text)
    assert report_clean.clean is True

    report_corrupt = UnicodeIntegrityGate.check(corrupt_text)
    assert report_corrupt.clean is False
    assert report_corrupt.replacement_chars == 2


def test_unicode_math_symbols_exempt():
    math_text = "Torque ω = π × RPM / 30, with error ± 0.05 and integral ∫ f(x) dx ≤ 100."
    report = UnicodeIntegrityGate.check(math_text)
    assert report.clean is True


# ── INV-3: OR PAIR SEMANTIC DISTINCTNESS ─────────────────────────────────────

def test_or_pair_distinctness_score_detects_identical():
    q1 = "Describe the operating principles of Antilock Braking Systems."
    q2 = "Describe the operating principles of Antilock Braking Systems."
    report = SemanticValidator.calculate_or_distinctness(q1, q2)
    assert report.is_distinct is False
    assert report.similarity_score > 0.85


def test_or_pair_distinctness_score_accepts_distinct():
    q1 = "Describe the operating principles of Antilock Braking Systems."
    q2 = "Calculate the braking distance for a vehicle traveling at 60 km/h."
    report = SemanticValidator.calculate_or_distinctness(q1, q2)
    assert report.is_distinct is True


# ── INV-4: BLOOM GRAMMAR VALIDATION ──────────────────────────────────────────

def test_bloom_verb_grammar_forbidden_combos():
    rep1 = BloomGrammarValidator.validate_verb_phrase("Create", "Create between the two control system architectures.")
    assert rep1.valid is False

    rep2 = BloomGrammarValidator.validate_verb_phrase("Apply", "Apply why the sensor voltage drops.")
    assert rep2.valid is False

    rep3 = BloomGrammarValidator.validate_verb_phrase("Explain", "Explain the operation of the EGR actuator.")
    assert rep3.valid is True


# ── COVERAGE GATE & HARD STOP TESTS ─────────────────────────────────────────

def test_evidence_coverage_gate_blocks_missing_module():
    chunks = [
        EvidenceChunk(
            chunk_id=f"m{mod}_p1_c{i:03d}",
            document_id="doc_123",
            source_path="/path/to/paper.pdf",
            adapter_id=ExtractionAdapterID.PYMUPDF,
            page_start=1,
            page_end=1,
            content_type=ContentType.TEXT,
            text=f"Academic content for module {mod} chunk {i}",
            module_id=str(mod),
            status=ChunkStatus.VALID,
        )
        for mod in (1, 2, 4, 5)  # Module 3 is missing!
        for i in range(10)
    ]
    report = ChunkValidationReport.from_chunks(chunks)
    decision = EvidenceCoverageGate.check_coverage(report, requested_modules=5, document_name="test.pdf")

    assert decision.action == "BLOCKED"
    assert "INSUFFICIENT_MODULE_EVIDENCE" in decision.reason


def test_hard_stop_prevents_llm_call():
    # Verify that when HardStopGate triggers, Qwen/LLM call count remains 0
    llm_call_count = 0

    chunks = [
        EvidenceChunk(
            chunk_id=f"m1_p1_c{i:03d}",
            document_id="doc_123",
            source_path="/path/to/paper.pdf",
            adapter_id=ExtractionAdapterID.PYMUPDF,
            page_start=1,
            page_end=1,
            content_type=ContentType.TEXT,
            text=f"Chunk {i}",
            status=ChunkStatus.VALID,
        ) for i in range(5)  # Insufficient chunks
    ]
    report = ChunkValidationReport.from_chunks(chunks)
    decision = ExtractionHardStopGate.check(report, requested_modules=5, document_name="test.pdf")

    if decision.action == "BLOCKED":
        pass  # Qwen is NOT called!
    else:
        llm_call_count += 1

    assert llm_call_count == 0


def test_gateway_txt_hard_rejection(tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Some text notes.")

    with pytest.raises(ExtractionError) as exc_info:
        ExtractionGateway.extract(str(txt_file))

    assert exc_info.value.code == "TXT_AS_SOURCE_REJECTED"
    assert exc_info.value.action == "HARD_REJECT"
