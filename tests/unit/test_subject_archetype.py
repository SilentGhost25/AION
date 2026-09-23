"""
tests/unit/test_subject_archetype.py
Unit tests verifying:
1. Subject 'IAI' resolves to 'ai_ml_data' with >= 1 domain directive.
2. Empty subject with topic 'IAI-MODULE-2' resolves to 'ai_ml_data' via short-alias bypass.
3. Completely unmapped subject returns the subject-neutral academic fallback directive (0 misses).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from aion_patch import resolve_subject_archetype, get_subject_directives, SUBJECT_ARCHETYPES


def test_iai_acronym_subject_resolution():
    """Case 1: 'IAI' as subject resolves directly to ai_ml_data archetype with directives."""
    arch = resolve_subject_archetype("IAI", "")
    assert arch == "ai_ml_data", f"Expected 'ai_ml_data', got {arch!r}"
    
    directives = get_subject_directives("IAI", "")
    assert len(directives) >= 1, "Expected >= 1 domain directive for IAI"
    assert any("Intelligent Agents" in d or "Search Algorithms" in d for d in directives)


def test_iai_topic_fallback_short_alias():
    """Case 2: Empty subject with topic 'IAI-MODULE-2' resolves to ai_ml_data via short alias."""
    arch = resolve_subject_archetype("", "IAI-MODULE-2")
    assert arch == "ai_ml_data", f"Expected 'ai_ml_data', got {arch!r}"
    
    directives = get_subject_directives("", "IAI-MODULE-2")
    assert len(directives) >= 1, "Expected >= 1 domain directive via topic fallback"


def test_unmapped_subject_returns_academic_fallback():
    """Case 3: Completely unmapped subject returns the subject-neutral academic fallback directive."""
    arch = resolve_subject_archetype("", "xyz_unmapped_random_course_topic")
    directives = get_subject_directives("", "xyz_unmapped_random_course_topic")
    assert len(directives) >= 1, f"Expected fallback directive for unmapped subject, got {directives}"
    assert "precise technical terminology" in directives[0].lower() or "academic" in directives[0].lower()


def test_fallback_directive_and_10m_multipart_prompt_generation():
    """Verify that unmapped subjects receive the fallback directive and 10M prompts mandate multi-part formatting."""
    from core.generation.orchestrator import SlotOrchestrator
    from aion_patch import _build_augmented_prompt
    
    orch = SlotOrchestrator(subject="XYZ_UNMAPPED")
    directives = get_subject_directives("XYZ_UNMAPPED", "Random Topic")
    assert len(directives) >= 1
    assert "precise technical terminology" in directives[0]
    
    from core.contracts.question_slot import QuestionSlot
    from core.contracts.budgets import AnswerBudget, QuestionBudget
    from core.contracts.task_signature import TaskSignature
    
    # Verify orchestrator builds 10M prompt with mandatory multi-part formatting instructions
    slot = QuestionSlot(
        slot_id="test_slot_10m",
        question_no=3,
        sub_label="b",
        or_pair_id="or_3",
        is_alternative=False,
        module_id=2,
        marks=10,
        bloom_level="L3",
        bloom_verb="Analyze",
        bloom_operation="EVALUATE",
        co="CO2",
        difficulty="MEDIUM",
        question_type="THEORY",
        topic="Heuristic Search Strategies",
        evidence_ids=("chunk_1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(10, "L3"),
        question_budget=QuestionBudget.from_bloom("L3", 10),
        task_signature=TaskSignature.from_bloom_marks_type("L3", 10, "THEORY"),
    )
    
    formatted_prompt = orch._format_prompt(slot, "Sample evidence on search trees.", "")
    assert "MANDATORY MULTI-PART FORMAT" in formatted_prompt
    assert "(i) [4 Marks]" in formatted_prompt
    assert "at least 80 words" in formatted_prompt
    
    # Verify aion_patch injects the fallback directive into the prompt for unmapped subjects
    augmented_prompt = _build_augmented_prompt(formatted_prompt, {"subject": "XYZ_UNMAPPED"})
    assert "precise technical terminology" in augmented_prompt
