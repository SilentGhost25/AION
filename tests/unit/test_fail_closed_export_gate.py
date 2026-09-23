# tests/unit/test_fail_closed_export_gate.py

import os
import pytest
from unittest.mock import MagicMock, patch

from core.contracts.question_slot import QuestionSlot, SlotStatus, UnresolvedSlotException
from core.contracts.budgets import AnswerBudget, QuestionBudget
from core.contracts.task_signature import TaskSignature
from core.contracts.question import GeneratedQuestion
from core.generation.output_schema import QuestionOutput
from core.validation.export_gate import ExportGate, DEFAULT_FAITHFULNESS_THRESHOLD, DEFAULT_QA_THRESHOLD


def _make_slot(slot_id: str = "slot_1", status: str = SlotStatus.PENDING.value) -> QuestionSlot:
    return QuestionSlot(
        slot_id=slot_id,
        question_no=1,
        sub_label="a",
        or_pair_id="OR_1",
        is_alternative=False,
        module_id=1,
        marks=6,
        bloom_level="L2",
        bloom_verb="Explain",
        bloom_operation="UNDERSTAND",
        co="CO1",
        difficulty="MEDIUM",
        question_type="descriptive",
        topic="Cloud Architecture",
        evidence_ids=("chunk_1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L2"),
        question_budget=QuestionBudget.from_bloom("L2", 6),
        task_signature=TaskSignature.from_bloom_marks_type("L2", 6, "descriptive"),
        math_required=False,
        visual_required=False,
        status=status,
    )


def _make_valid_question(slot: QuestionSlot) -> GeneratedQuestion:
    output = QuestionOutput(
        instruction="Explain the core architectural components of cloud service models.",
        question_text="Explain the core architectural components of cloud service models.",
        math_blocks=[]
    )
    gq = GeneratedQuestion(output=output, slot=slot)
    gq.status = "PASS"
    return gq


def test_export_gate_passes_clean_question():
    """Verify that valid, validated questions pass ExportGate cleanly."""
    slot = _make_slot()
    gq = _make_valid_question(slot)

    decision = ExportGate.evaluate(generated_slots=[gq])
    assert decision.passed is True
    assert decision.status == "PASS"

    chk = ExportGate.validate([gq])
    assert chk.passed is True


def test_export_gate_blocks_unresolved_status():
    """Verify that any question with status='UNRESOLVED' blocks export fail-closed."""
    slot = _make_slot()
    gq = _make_valid_question(slot)
    gq.status = "UNRESOLVED"

    chk = ExportGate.validate([gq])
    assert chk.passed is False
    assert chk.code == "UNRESOLVED_SLOT_DETECTED"
    assert "unresolved slot" in chk.message

    decision = ExportGate.evaluate(generated_slots=[gq])
    assert decision.passed is False
    assert decision.status == "BLOCKED"
    assert any("UNRESOLVED_SLOT_DETECTED" in f for f in decision.failures)


def test_export_gate_blocks_unresolved_slot_status():
    """Verify that if the underlying slot has status='UNRESOLVED', export is blocked."""
    slot = _make_slot(status=SlotStatus.UNRESOLVED.value)
    gq = _make_valid_question(slot)
    gq.status = "PASS"  # Question claims pass, but underlying slot is unresolved

    chk = ExportGate.validate([gq])
    assert chk.passed is False
    assert chk.code == "UNRESOLVED_SLOT_DETECTED"


def test_export_gate_blocks_unresolved_stub_question():
    """Verify that GeneratedQuestion.create_unresolved stub is strictly blocked."""
    slot = _make_slot()
    stub = GeneratedQuestion.create_unresolved(
        slot=slot,
        failure_code="TEACHER_SUITABILITY_FAILURE",
        reason="Exhausted retries on pedagogical validity"
    )

    assert stub.status == "UNRESOLVED"
    chk = ExportGate.validate([stub])
    assert chk.passed is False
    assert chk.code == "UNRESOLVED_SLOT_DETECTED"


def test_export_gate_faithfulness_threshold():
    """Verify that faithfulness scores below threshold fail export when hard block is enabled."""
    slot = _make_slot()
    gq = _make_valid_question(slot)

    # Attach ragas metric with low faithfulness
    mock_ragas = MagicMock()
    mock_ragas.faithfulness = 0.60
    setattr(gq, "ragas_metrics", mock_ragas)

    with patch.dict(os.environ, {"AION_ENABLE_UNRESOLVED_HARD_BLOCK": "true"}):
        chk = ExportGate.validate([gq])
        assert chk.passed is False
        assert chk.code == "FAITHFULNESS_BELOW_THRESHOLD"

    with patch.dict(os.environ, {"AION_ENABLE_UNRESOLVED_HARD_BLOCK": "false"}):
        chk = ExportGate.validate([gq])
        assert chk.passed is True


def test_slot_status_enum_values():
    """Verify SlotStatus enum contains the exact 5 states without PASS_WITH_WARNING."""
    assert SlotStatus.PENDING == "PENDING"
    assert SlotStatus.GENERATED == "GENERATED"
    assert SlotStatus.PASS == "PASS"
    assert SlotStatus.UNRESOLVED == "UNRESOLVED"
    assert SlotStatus.FAILED == "FAILED"
    assert not hasattr(SlotStatus, "PASS_WITH_WARNING")


def test_unresolved_slot_exception_properties():
    """Verify UnresolvedSlotException cleanly packages slot and failure code."""
    slot = _make_slot(slot_id="slot_test_99")
    exc = UnresolvedSlotException(slot, failure_code="DOMAIN_INTEGRITY_VIOLATION")
    assert exc.slot.slot_id == "slot_test_99"
    assert exc.failure_code == "DOMAIN_INTEGRITY_VIOLATION"
    assert "slot_test_99" in str(exc)
    assert "DOMAIN_INTEGRITY_VIOLATION" in str(exc)


def test_orchestrator_raises_on_linter_exhaustion_when_flag_enabled():
    """
    Verify that SlotOrchestrator raises UnresolvedSlotException when
    AION_ENABLE_UNRESOLVED_HARD_BLOCK=true and retries exhaust on a failing check.
    """
    from core.generation.orchestrator import SlotOrchestrator
    from core.validation.common import CheckResult
    from core.validation.linter import LintReport

    orch = SlotOrchestrator()
    slot = _make_slot(slot_id="slot_fail_1")

    # Mock LLM to return valid JSON so schema parsing passes
    orch._call_llm = MagicMock(return_value='{"instruction": "Explain cloud.", "question_text": "Explain cloud.", "math_blocks": []}')

    # Mock linter to fail every time
    failing_report = LintReport(slot.slot_id, {
        "teacher_suitability": CheckResult.fail("TEACHER_SUITABILITY_FAILURE", "Pedagogically unsuitable.")
    })

    with patch("core.generation.orchestrator.run_linter", return_value=failing_report):
        with patch.dict(os.environ, {"AION_ENABLE_UNRESOLVED_HARD_BLOCK": "true"}):
            with pytest.raises(UnresolvedSlotException) as exc_info:
                orch.generate(slot, "Some evidence text")

            assert exc_info.value.slot.slot_id == "slot_fail_1"
            assert exc_info.value.slot.status == SlotStatus.UNRESOLVED.value
            assert exc_info.value.failure_code == "TEACHER_SUITABILITY_FAILURE"
