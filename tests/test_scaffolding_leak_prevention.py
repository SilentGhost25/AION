import re
import pytest
from pathlib import Path

from core.contracts.module_identity import strip_module_header, MODULE_HEADER_PATTERN
from v0_1.main import _strip_module_header
from core.contracts.question_slot import QuestionSlot
from core.generation.orchestrator import SlotOrchestrator


@pytest.mark.parametrize("input_topic,expected", [
    # Exact bugs from review:
    ("MODULE 1: Orbital Mechanics", "Orbital Mechanics"),
    ("Module 2 - Satellite Subsystems", "Satellite Subsystems"),
    ("UNIT 3: Communication Links", "Communication Links"),
    ("module_1: Basics", "Basics"),
    ("Chapter 4 — Advanced Topics", "Advanced Topics"),
    ("CHAPTER5:Math", "Math"),
    ("   MODULE 1: Physics  ", "Physics"),
    # Non-matches (should NOT strip):
    ("Modular Arithmetic", "Modular Arithmetic"),
    ("The Module Pattern", "The Module Pattern"),
    ("", ""),
    (None, ""),
])
def test_module_header_stripping_contract(input_topic, expected):
    """Verify single source of truth regex in module_identity correctly handles all cases."""
    assert strip_module_header(input_topic) == expected
    # Verify alias in v0_1.main produces identical result
    assert _strip_module_header(input_topic) == expected


def test_sanitize_question_text_strips_module_headers():
    """Verify _sanitize_question_text removes leaked module headers from question stems."""
    orch = SlotOrchestrator()
    sample = "MODULE 1: Orbital Mechanics and Keplerian Dynamics governs the satellite motion."
    cleaned = orch._sanitize_question_text(sample)
    assert "MODULE 1:" not in cleaned
    assert cleaned == "Orbital Mechanics and Keplerian Dynamics governs the satellite motion."


from core.contracts.budgets import AnswerBudget, QuestionBudget
from core.contracts.task_signature import TaskSignature


def _make_test_slot(slot_id, module_id, q_no, sub_label, topic, verb, bloom, bloom_op, marks=6):
    return QuestionSlot(
        slot_id=slot_id,
        question_no=q_no,
        sub_label=sub_label,
        or_pair_id=f"module_{module_id}_OR_1",
        is_alternative=False,
        module_id=module_id,
        marks=marks,
        bloom_level=bloom,
        bloom_verb=verb,
        bloom_operation=bloom_op,
        co=f"CO{module_id}",
        difficulty="EASY" if marks <= 6 else "MEDIUM",
        question_type="THEORY",
        topic=topic,
        evidence_ids=(f"chunk_{module_id}",),
        answer_budget=AnswerBudget.from_marks_and_bloom(marks, bloom),
        question_budget=QuestionBudget.from_bloom(bloom, marks),
        task_signature=TaskSignature.from_bloom_marks_type(bloom, marks, "THEORY"),
        keywords=(),
    )


def test_fallback_template_does_not_leak_module_header():
    """Verify fallback question generation never echoes MODULE X: into question text."""
    orch = SlotOrchestrator()
    slot = _make_test_slot(
        slot_id="module_1_Q1_a",
        module_id=1,
        q_no=1,
        sub_label="a",
        topic="MODULE 1: Orbital Mechanics and Keplerian Dynamics",
        verb="List",
        bloom="L1",
        bloom_op="remember",
        marks=6,
    )

    candidate = orch._generate_template_fallback(slot, evidence_pack="Some dummy notes about satellite orbit.")
    assert candidate is not None
    q_text = candidate.question_text
    assert "MODULE 1:" not in q_text
    assert "MODULE 1" not in q_text
    assert "Orbital Mechanics and Keplerian Dynamics" in q_text
    assert q_text.startswith("List ")


def test_format_prompt_cleans_example_topic_and_contract_topic():
    """Verify _format_prompt uses clean_ex_topic without MODULE header."""
    orch = SlotOrchestrator()
    slot = _make_test_slot(
        slot_id="module_2_Q3_a",
        module_id=2,
        q_no=3,
        sub_label="a",
        topic="Module 2 - Satellite Subsystems and Antennas",
        verb="Explain",
        bloom="L2",
        bloom_op="understand",
        marks=6,
    )

    class DummyEvidencePack:
        combined_text = "Detailed notes on satellite power subsystems and communications payload."

    prompt = orch._format_prompt(slot, DummyEvidencePack(), extra_hints="")
    assert "Topic: Module 2 -" not in prompt
    assert "Topic: Satellite Subsystems and Antennas" in prompt
    assert "Module 2 -" not in prompt
