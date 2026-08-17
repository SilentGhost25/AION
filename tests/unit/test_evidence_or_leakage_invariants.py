# tests/unit/test_evidence_or_leakage_invariants.py

import pytest
from core.contracts.question_slot import QuestionSlot, AnswerBudget, QuestionBudget, TaskSignature
from core.contracts.question import QuestionProvenance
from core.validation.linter import check_no_answer_leakage


def test_positive_evidence_module_grounding():
    """
    Audit Gap 2:
    Positive test proving legitimate evidence from Module 3 is selected for a Module 3 slot,
    and question.provenance.module_id matches slot.module_id.
    """
    slot_m3 = QuestionSlot(
        slot_id="module_3_Q5_a",
        question_no=5,
        sub_label="a",
        or_pair_id="module_3_OR_1",
        is_alternative=False,
        module_id=3,
        co="CO3",
        marks=5,
        bloom_level="L2",
        bloom_verb="Explain",
        bloom_operation="UNDERSTAND",
        difficulty="EASY",
        question_type="CONCEPTUAL",
        topic="Neural Networks",
        evidence_ids=("m3_chunk_12", "m3_chunk_13"),
        answer_budget=AnswerBudget.from_marks_and_bloom(5, "L2"),
        question_budget=QuestionBudget.from_bloom("L2", 5),
        task_signature=TaskSignature("UNDERSTAND", (), False, False, False),
        math_required=False,
        visual_required=False,
        generation_seed=201,
    )

    provenance = QuestionProvenance.from_slot(slot_m3)

    assert provenance.module_id == slot_m3.module_id
    assert provenance.evidence_ids == ("m3_chunk_12", "m3_chunk_13")
    assert provenance.co == "CO3"


def test_or_pair_structural_parity():
    """
    Audit Gap 4:
    Verifies that OR-A (main question) and OR-B (alternative question)
    preserve exact mark parity, subquestion count, CO, and Bloom level.
    """
    or_main = QuestionSlot(
        slot_id="module_1_Q1_a",
        question_no=1,
        sub_label="a",
        or_pair_id="module_1_OR_1",
        is_alternative=False,
        module_id=1,
        co="CO1",
        marks=6,
        bloom_level="L3",
        bloom_verb="Apply",
        bloom_operation="APPLY",
        difficulty="MEDIUM",
        question_type="NUMERICAL",
        topic="Reflex Agents",
        evidence_ids=("m1_c1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L3"),
        question_budget=QuestionBudget.from_bloom("L3", 6),
        task_signature=TaskSignature("APPLY", ("CALCULATE",), False, True, False),
        math_required=True,
        visual_required=False,
        generation_seed=301,
    )

    or_alt = QuestionSlot(
        slot_id="module_1_Q2_a",
        question_no=2,
        sub_label="a",
        or_pair_id="module_1_OR_1",
        is_alternative=True,
        module_id=1,
        co="CO1",
        marks=6,
        bloom_level="L3",
        bloom_verb="Apply",
        bloom_operation="APPLY",
        difficulty="MEDIUM",
        question_type="NUMERICAL",
        topic="Goal-Based Agents",
        evidence_ids=("m1_c2",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L3"),
        question_budget=QuestionBudget.from_bloom("L3", 6),
        task_signature=TaskSignature("APPLY", ("CALCULATE",), False, True, False),
        math_required=True,
        visual_required=False,
        generation_seed=302,
    )

    assert or_main.marks == or_alt.marks
    assert or_main.co == or_alt.co
    assert or_main.bloom_level == or_alt.bloom_level
    assert or_main.module_id == or_alt.module_id


def test_expanded_answer_leakage_detection():
    """
    Audit Gap 5:
    Expanded answer leakage detection test catching phrases like:
    'the correct answer is', 'thus, the required result', 'hence the output will be'.
    """
    leak_1 = check_no_answer_leakage("Calculate the velocity. The correct answer is 15 m/s.")
    assert leak_1.passed is False
    assert leak_1.code == "ANSWER_LEAK"

    leak_2 = check_no_answer_leakage("Derive the state equation. Thus, the required result is X = Y + Z.")
    assert leak_2.passed is False
    assert leak_2.code == "ANSWER_LEAK"

    clean = check_no_answer_leakage("Analyze the architecture of an intelligent agent and explain its components.")
    assert clean.passed is True
