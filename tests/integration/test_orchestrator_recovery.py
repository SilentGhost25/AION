# tests/integration/test_orchestrator_recovery.py

import json
import pytest
from unittest.mock import patch
from core.contracts.question_slot import QuestionSlot
from core.contracts.budgets import AnswerBudget, QuestionBudget
from core.contracts.task_signature import TaskSignature
from core.generation.orchestrator import SlotOrchestrator
from core.generation.output_schema import QuestionOutput


def test_orchestrator_recovery_loop():
    """
    Test targeted recovery loop:
    1st attempt: LLM returns invalid JSON (schema failure).
    2nd attempt: LLM returns question text with wrong starting verb (bloom verb mismatch).
    3rd attempt: LLM returns a correct, compliant question.
    Expected outcome: SlotOrchestrator automatically recovers, retries, and returns the valid candidate.
    """
    answer_budget = AnswerBudget.from_marks_and_bloom(6, "L2")
    question_budget = QuestionBudget.from_bloom("L2", 6)
    task_signature = TaskSignature.from_bloom_marks_type("L2", 6, "descriptive")

    slot = QuestionSlot(
        slot_id="Q1",
        question_no=1,
        sub_label="",
        or_pair_id="mod1_OR_1",
        is_alternative=False,
        module_id=1,
        marks=6,
        bloom_level="L2",
        bloom_verb="Explain",
        bloom_operation="UNDERSTAND",
        co="CO1",
        difficulty="MIXED",
        question_type="descriptive",
        topic="test",
        evidence_ids=("chk1",),
        answer_budget=answer_budget,
        question_budget=question_budget,
        task_signature=task_signature,
        math_required=False,
        visual_required=False,
        generation_seed=123
    )

    attempt_counter = 0

    def mock_call_llm(prompt):
        nonlocal attempt_counter
        attempt_counter += 1
        
        if attempt_counter == 1:
            # Attempt 1: Invalid JSON to trigger SCHEMA_FAILURE
            return "{invalid_json}"
            
        elif attempt_counter == 2:
            # Attempt 2: Valid JSON, but violates Bloom starting verb check (expects 'Explain')
            return json.dumps({
                "instruction": "Outline the key principles.",
                "question_text": "Outline the key principles in detail.",
                "math_blocks": []
            })
            
        else:
            # Attempt 3: Fully compliant output
            return json.dumps({
                "instruction": "Explain the key principles and define the concepts.",
                "question_text": "Explain the key principles and define the concepts in detail.",
                "math_blocks": []
            })

    orchestrator = SlotOrchestrator(artifact=None)
    
    with patch.object(orchestrator, "_call_llm", side_effect=mock_call_llm):
        candidate = orchestrator.generate(slot, evidence_pack="Explain the key principles and define the concepts in detail mock evidence.", excluded_concepts=set())

    # Assertions
    assert attempt_counter == 3
    assert candidate.question_text == "Explain the key principles and define the concepts in detail."
    assert candidate.slot.slot_id == "Q1"
    
    # Assert orchestrator recorded the recovery logs properly
    session_log = orchestrator.session_log
    assert len(session_log) == 3
    
    # Attempt 1 log (Schema failure)
    assert session_log[0]["attempt"] == 1
    assert session_log[0]["code"] == "SCHEMA_FAILURE"
    
    # Attempt 2 log (Linter failure: Bloom verb)
    assert session_log[1]["attempt"] == 2
    assert session_log[1]["code"] == "BLOOM_MISMATCH"
    
    # Attempt 3 log (Pass)
    assert session_log[2]["attempt"] == 3
    assert session_log[2]["status"] == "PASS"
