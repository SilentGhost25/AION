# tests/integration/test_failure_injection.py

import json
import pytest
from unittest.mock import patch, MagicMock
from concurrent.futures import Future
from aion_api import app
from v0_1.main import run_pipeline
from core.contracts.question_slot import QuestionSlot
from core.contracts.question import GeneratedQuestion
from core.contracts.budgets import AnswerBudget, QuestionBudget
from core.contracts.task_signature import TaskSignature
from core.validation.common import CheckResult, RetryAction
from core.validation.linter import run_linter, check_contract_integrity
from core.validation.export_gate import ExportGate
from core.extraction.gateway import ExtractionGateway, ExtractionError, DocumentArtifact
from core.extraction.contracts import EvidenceChunk, ChunkStatus, ContentType, ExtractionAdapterID
from pydantic import ValidationError


class SyncExecutor:
    def __init__(self, max_workers=None):
        pass
    def submit(self, fn, *args, **kwargs):
        f = Future()
        try:
            res = fn(*args, **kwargs)
            f.set_result(res)
        except Exception as e:
            f.set_exception(e)
        return f
    def shutdown(self, wait=True):
        pass


# Helper to create a standard QuestionSlot and QuestionContract/Output
def create_test_slot_and_gq(slot_id="Q1", marks=6, bloom_level="L2", co="CO1", module_id=1, math_required=False):
    from core.generation.output_schema import QuestionOutput
    
    answer_budget = AnswerBudget.from_marks_and_bloom(marks, bloom_level)
    question_budget = QuestionBudget.from_bloom(bloom_level, marks)
    task_signature = TaskSignature.from_bloom_marks_type(bloom_level, marks, "descriptive")
    
    if bloom_level == "L2":
        bloom_verb = "Explain"
        bloom_op = "UNDERSTAND"
    elif bloom_level == "L3":
        bloom_verb = "Solve"
        bloom_op = "APPLY"
    elif bloom_level == "L4":
        bloom_verb = "Compare"
        bloom_op = "ANALYZE"
    elif bloom_level == "L5":
        bloom_verb = "Justify"
        bloom_op = "EVALUATE"
    else:
        bloom_verb = "Explain"
        bloom_op = "UNDERSTAND"
        
    slot = QuestionSlot(
        slot_id=slot_id,
        question_no=1,
        sub_label="",
        or_pair_id="mod1_OR_1",
        is_alternative=False,
        module_id=module_id,
        marks=marks,
        bloom_level=bloom_level,
        bloom_verb=bloom_verb,
        bloom_operation=bloom_op,
        co=co,
        difficulty="MIXED",
        question_type="descriptive",
        topic=slot_id,
        evidence_ids=("chk1",),
        answer_budget=answer_budget,
        question_budget=question_budget,
        task_signature=task_signature,
        math_required=math_required,
        visual_required=False,
        generation_seed=123
    )
    
    output = QuestionOutput(
        instruction=f"{bloom_verb} the key principles, describe their characteristics, and outline the details.",
        question_text=f"{bloom_verb} the key principles, describe their characteristics, and outline the details.",
        math_blocks=[]
    )
    
    gq = GeneratedQuestion(output, slot)
    return slot, gq, slot.to_contract()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# -------------------------------------------------------------
# Test 23 — Wrong CO
# -------------------------------------------------------------
def test_failure_injection_wrong_co():
    """
    Force a legacy adapter or bypass to construct a GeneratedQuestion with a mismatched CO.
    Expected: check_contract_integrity returns a CRITICAL CONTRACT_FAILURE.
    """
    slot, gq, contract = create_test_slot_and_gq(co="CO1")
    
    # Manually bypass/override CO to wrong CO
    gq.co = "CO2"
    
    res = check_contract_integrity(gq, slot)
    assert not res.passed
    assert res.code == "CONTRACT_FAILURE"
    assert "CO mismatch" in res.message


# -------------------------------------------------------------
# Test 24 — Wrong Bloom (Bloom mismatch)
# -------------------------------------------------------------
def test_failure_injection_wrong_bloom():
    """
    Force Qwen to output a question text with a verb mismatching the slot's Bloom level.
    Expected: DemandValidator/Linter fails with BLOOM_MISMATCH.
    """
    from core.generation.output_schema import QuestionOutput
    
    slot, gq, contract = create_test_slot_and_gq(bloom_level="L4") # ANALYZE -> requires Compare
    
    # Qwen outputs a remember level verb ("List") for an L4 slot
    output = QuestionOutput(
        instruction="List three routing protocols.",
        question_text="List three routing protocols.",
        math_blocks=[]
    )
    
    res = run_linter(output, gq, slot, contract)
    assert not res.passed
    assert "bloom_verb_start" in res.checks
    assert not res.checks["bloom_verb_start"].passed
    assert "expected Bloom verb 'Compare'" in res.checks["bloom_verb_start"].message


# -------------------------------------------------------------
# Test 25 — Answer leakage
# -------------------------------------------------------------
def test_failure_injection_answer_leakage():
    """
    Qwen generates question text containing answer leakage/solution.
    Expected: Pydantic schema validation raises ValidationError (Answer leakage).
    """
    from core.generation.output_schema import QuestionOutput
    
    with pytest.raises(ValidationError) as exc_info:
        QuestionOutput(
            instruction="Explain the concept of DNS.",
            question_text="Explain the concept of DNS. Answer: DNS translates domain names.",
            math_blocks=[]
        )
    assert "Answer leakage" in str(exc_info.value)


# -------------------------------------------------------------
# Test 26 — Broken math
# -------------------------------------------------------------
def test_failure_injection_broken_math():
    """
    Qwen outputs malformed LaTeX (unbalanced braces or invalid commands).
    Expected: Linter fails with MATH_UNCLOSED_BRACES / MATH_RENDER_FAILURE.
    """
    from core.generation.output_schema import QuestionOutput
    
    slot, gq, contract = create_test_slot_and_gq(bloom_level="L3", math_required=True)
    
    output = QuestionOutput(
        instruction="Solve the formula.",
        question_text="Solve the formula [MATH:eq1].",
        math_blocks=[{
            "block_id": "eq1",
            "latex": "\\frac{x{2}", # unbalanced brace
            "display_mode": False
        }]
    )
    
    res = run_linter(output, gq, slot, contract)
    assert not res.passed
    assert "math_render" in res.checks
    assert not res.checks["math_render"].passed
    assert "unclosed braces" in res.checks["math_render"].message.lower()


# -------------------------------------------------------------
# Test 27 — Unicode corruption
# -------------------------------------------------------------
def test_failure_injection_unicode_corruption():
    """
    Qwen outputs text with replacement/corrupt characters.
    Expected: Linter fails with UNICODE_CORRUPT.
    """
    from core.generation.output_schema import QuestionOutput
    
    slot, gq, contract = create_test_slot_and_gq(bloom_level="L2")
    
    output = QuestionOutput(
        instruction="Explain the \ufffd protocol.",
        question_text="Explain the \ufffd protocol in detail.",
        math_blocks=[]
    )
    
    res = run_linter(output, gq, slot, contract)
    assert not res.passed
    assert "unicode_text" in res.checks
    assert not res.checks["unicode_text"].passed
    assert "corrupt" in res.checks["unicode_text"].message.lower()


# -------------------------------------------------------------
# Test 28 — PDF internal leakage
# -------------------------------------------------------------
def test_failure_injection_pdf_leakage():
    """
    Qwen outputs text containing raw PDF metadata internals.
    Expected: Pydantic schema validation raises ValidationError (Meta-language).
    """
    from core.generation.output_schema import QuestionOutput
    
    with pytest.raises(ValidationError) as exc_info:
        QuestionOutput(
            instruction="Explain the /fontfile system.",
            question_text="Explain the flatdecode contents.",
            math_blocks=[]
        )
    assert "Meta-language" in str(exc_info.value)


# -------------------------------------------------------------
# Test 29 — Fused subquestions
# -------------------------------------------------------------
def test_failure_injection_fused_subquestions():
    """
    Qwen outputs a question text that contains subquestion list labels (a), (b).
    Expected: Linter fails with MULTI_SLOT_CONTAMINATION.
    """
    from core.generation.output_schema import QuestionOutput
    
    slot, gq, contract = create_test_slot_and_gq(bloom_level="L2")
    
    output = QuestionOutput(
        instruction="Explain TCP.",
        question_text="(a) Explain TCP headers and (b) UDP headers.",
        math_blocks=[]
    )
    
    res = run_linter(output, gq, slot, contract)
    assert not res.passed
    assert "multi_slot" in res.checks
    assert not res.checks["multi_slot"].passed
    assert "Multi-slot contamination" in res.checks["multi_slot"].message


# -------------------------------------------------------------
# Test 30 — OR duplicate
# -------------------------------------------------------------
def test_failure_injection_or_duplicate():
    """
    Test OR pair validation when alternative B is identical/duplicate to A.
    Expected: validate_and_repair regenerates alternative B.
    """
    from core.assembly.or_pair_validator import validate_and_repair
    from core.contracts.question import LegacyGeneratedQuestionAdapter
    
    gq_a = [LegacyGeneratedQuestionAdapter.adapt("Q1a", "Explain BFS algorithm.", 5, "L2", "CO1")]
    gq_b = [LegacyGeneratedQuestionAdapter.adapt("Q2a", "Explain BFS algorithm.", 5, "L2", "CO1")]
    slots_b = [gq_b[0].slot]
    
    mock_orch = MagicMock()
    mock_orch.generate.return_value = LegacyGeneratedQuestionAdapter.adapt("Q2a", "Explain DFS traversal in detail.", 5, "L2", "CO1")
    
    new_a, new_b = validate_and_repair(
        q_a=gq_a,
        q_b=gq_b,
        slots_b=slots_b,
        excluded=frozenset(),
        orchestrator=mock_orch
    )
    
    assert mock_orch.generate.called
    assert new_b[0].question_text == "Explain DFS traversal in detail."


# -------------------------------------------------------------
# Test 31 — Marks violation
# -------------------------------------------------------------
def test_failure_injection_marks_violation():
    """
    Test a slot with high marks (e.g. 10M) where Qwen produces a simple 1-part instruction.
    Expected: Linter fails with DEMAND_FAILURE (insufficient cognitive dimensions).
    """
    from core.generation.output_schema import QuestionOutput
    
    slot, gq, contract = create_test_slot_and_gq(marks=10, bloom_level="L3")
    
    # Instruction is too simple (1 part) for 10 marks (which requires at least 4 dimensions)
    output = QuestionOutput(
        instruction="Solve the equation.",
        question_text="Solve the equation in detail.",
        math_blocks=[]
    )
    
    res = run_linter(output, gq, slot, contract)
    assert not res.passed
    assert "answer_demand" in res.checks
    assert not res.checks["answer_demand"].passed
    assert "insufficient cognitive dimensions" in res.checks["answer_demand"].message.lower() or "requires at least" in res.checks["answer_demand"].message.lower()


# -------------------------------------------------------------
# Test 32 — Extraction failure
# -------------------------------------------------------------
def test_failure_injection_extraction_failure(client):
    """
    Submit a document with zero valid academic content.
    Expected: HTTP 200 stream response containing the extraction rejection event and NO Qwen calls.
    """
    payload = {
        "subject": "Unknown",
        "department": "AIML",
        "semester": 5,
        "exam_type": "IAT-1",
        "selected_modules": [1],
        "bloom_levels": ["L2"],
        "difficulty": "MIXED",
        "model": "qwen2.5:14b",
        "notes_text": "Not academic.",
    }
    
    def mock_extract_empty(source_path: str, document_id="doc_001", store=None):
        raise ExtractionError(
            code="ACADEMIC_QUALITY_REJECTED",
            message="No academic syllabus or lecture text detected.",
            action="STOP"
        )
        
    with patch("core.extraction.gateway.ExtractionGateway.extract", mock_extract_empty), \
         patch("v0_1.main.upload", lambda x: x):
        response = client.post(
            "/api/generate/stream",
            json=payload
        )
        
    assert response.status_code == 200
    events = response.data.decode("utf-8")
    assert "ACADEMIC_QUALITY_REJECTED" in events or "Extraction Hard Stop" in events


# Test 33 — Wrong CO contract validation (A. Wrong CO)
# -------------------------------------------------------------
def test_failure_injection_wrong_co():
    """
    Test wrong CO contract validation.
    Expected: Contract validation fails with critical mismatch.
    """
    slot, _, _ = create_test_slot_and_gq(co="CO3")
    output = MagicMock()
    output.question_text = "Explain the DFS algorithm."
    output.math_blocks = []
    
    gq = GeneratedQuestion(output, slot)
    # Manually violate the contract to simulate wrong CO override
    gq.co = "CO2"
    
    res = check_contract_integrity(gq, slot)
    assert not res.passed
    assert "CO mismatch" in res.message


# Test 34 — Wrong Bloom verb (B. Wrong Bloom)
# -------------------------------------------------------------
def test_failure_injection_wrong_bloom_verb():
    """
    Test wrong Bloom level verb validation.
    Expected: Bloom verb validation fails (BLOOM_VERB_FAILURE).
    """
    from core.validation.linter import check_bloom_verb_at_start
    slot, _, _ = create_test_slot_and_gq(bloom_level="L5") # Evaluate/Justify
    
    # instruction starts with "List" (L1 verb), violating L5 contract
    instruction = "List all components of DFS."
    res = check_bloom_verb_at_start(instruction, slot)
    assert not res.passed
    assert "must start with expected Bloom verb" in res.message


# Test 35 — Cross-module evidence (C. Cross-module evidence)
# -------------------------------------------------------------
def test_failure_injection_cross_module_evidence():
    """
    Test cross-module evidence validation.
    Expected: Module isolation check rejects chunk with cross-module reference.
    """
    from core.validation.linter import check_module_isolation
    # Slot is for module 3
    slot, _, _ = create_test_slot_and_gq(module_id=3)
    
    # GeneratedQuestion references an evidence chunk from module 4
    output = MagicMock()
    output.question_text = "Design a hash table."
    output.math_blocks = []
    
    gq = GeneratedQuestion(output, slot)
    gq.evidence_ids = ("chunk_m4_101",) # module 4 chunk
    
    res = check_module_isolation(gq, slot)
    assert not res.passed
    assert "Cross-module evidence detected" in res.message
