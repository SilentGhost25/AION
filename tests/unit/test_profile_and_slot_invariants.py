# tests/unit/test_profile_and_slot_invariants.py

import os
import pytest

from core.contracts.question_slot import QuestionSlot, AnswerBudget, QuestionBudget, TaskSignature, QuestionContract
from core.generation.output_schema import QuestionOutput
from core.contracts.demand_profile import DemandProfile
from core.validation.demand_validator import DemandValidator
from runtime.profiles import PRODUCTION_PROFILE, LAPTOP_FAST_PROFILE
from runtime import set_active_profile
from v0_1.llm import get_best_llm


def test_production_profile_model_whitelist_enforcement():
    """
    Audit Item #1:
    PRODUCTION profile + unapproved model (e.g. qwen2.5:1.5b) MUST raise RuntimeError
    and refuse generation, preventing silent downgrade.
    """
    os.environ["AION_PROFILE"] = "PRODUCTION"
    os.environ["AION_MODEL"] = "qwen2.5:1.5b"
    
    set_active_profile(PRODUCTION_PROFILE)
    
    with pytest.raises(RuntimeError) as exc_info:
        get_best_llm()
        
    assert "PROFILE INTEGRITY VIOLATION" in str(exc_info.value)
    
    # Cleanup
    os.environ.pop("AION_MODEL", None)
    os.environ.pop("AION_PROFILE", None)
    set_active_profile(None)


def test_slot_retry_equivalence_and_non_drift():
    """
    Audit Item #3:
    Every retry attempt MUST preserve slot_id, module_id, co, bloom_level, marks,
    and seed increment from the original slot to prevent recovery drift.
    """
    original = QuestionSlot(
        slot_id="module_2_Q3_a",
        question_no=3,
        sub_label="a",
        or_pair_id="module_2_OR_1",
        is_alternative=False,
        module_id=2,
        co="CO2",
        marks=6,
        bloom_level="L3",
        bloom_verb="Apply",
        bloom_operation="Calculate",
        difficulty="MEDIUM",
        question_type="NUMERICAL",
        topic="Binary Search Trees",
        evidence_ids=("m2_c1", "m2_c2"),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L3"),
        question_budget=QuestionBudget.from_bloom("L3", 6),
        task_signature=TaskSignature(
            primary_operation="APPLY",
            allowed_secondary_operations=("CALCULATE",),
            requires_comparison=False,
            requires_calculation=True,
            requires_justification=False
        ),
        math_required=True,
        visual_required=False,
        generation_seed=101,
    )

    attempt_slot = original.make_attempt_slot(attempt=2)

    # Invariants
    assert attempt_slot.slot_id == original.slot_id
    assert attempt_slot.module_id == original.module_id
    assert attempt_slot.co == original.co
    assert attempt_slot.bloom_level == original.bloom_level
    assert attempt_slot.marks == original.marks
    assert attempt_slot.sub_label == original.sub_label
    assert attempt_slot.evidence_ids == original.evidence_ids
    assert attempt_slot.generation_seed == original.generation_seed + 2


def test_demand_validation_mark_proportionality():
    """
    Audit Item #6 & #7:
    Cognitive demand must scale proportionally with marks:
    - 2 marks: min_dimensions = 1
    - 5 marks: min_dimensions = 2 (L2) or 1
    - 10 marks: min_dimensions = 3 or 4
    """
    contract_2m = QuestionContract(
        slot_id="module_1_Q1_a",
        question_no=1,
        sub_label="a",
        module_id=1,
        marks=2,
        bloom_level="L1",
        bloom_verb="Recall",
        bloom_operation="REMEMBER",
        co="CO1",
        difficulty="EASY",
        question_type="CONCEPTUAL",
        topic="Agents",
        evidence_ids=("m1_c1",),
        task_signature=TaskSignature("REMEMBER", (), False, False, False),
        math_required=False,
        visual_required=False,
    )
    dp_2m = DemandProfile.from_contract(contract_2m)

    contract_10m = QuestionContract(
        slot_id="module_1_Q1_b",
        question_no=1,
        sub_label="b",
        module_id=1,
        marks=10,
        bloom_level="L4",
        bloom_verb="Analyze",
        bloom_operation="ANALYZE",
        co="CO1",
        difficulty="HARD",
        question_type="CONCEPTUAL",
        topic="Intelligent Agents",
        evidence_ids=("m1_c1",),
        task_signature=TaskSignature("ANALYZE", ("COMPARE", "JUSTIFY"), True, False, True),
        math_required=False,
        visual_required=False,
    )
    dp_10m = DemandProfile.from_contract(contract_10m)

    assert dp_2m.min_dimensions >= 1
    assert dp_10m.min_dimensions >= 3

    # Test single-clause output failing 10-mark demand validation
    single_clause_output = QuestionOutput(
        instruction="Explain the definition of an intelligent agent.",
        question_text="Explain the definition of an intelligent agent.",
        math_blocks=[],
    )
    result_fail = DemandValidator.validate(output=single_clause_output, contract=contract_10m)
    assert result_fail.passed is False
    assert "INSUFFICIENT_DECLARED_DIMENSIONS" in result_fail.code

    # Test multi-clause output passing 10-mark demand validation
    multi_clause_output = QuestionOutput(
        instruction=(
            "Analyze the architecture of an intelligent agent, compare reflex and goal-based agents, "
            "as well as calculate the optimal utility decision, and justify your design choices."
        ),
        question_text=(
            "Analyze the architecture of an intelligent agent, compare reflex and goal-based agents, "
            "as well as calculate the optimal utility decision, and justify your design choices."
        ),
        math_blocks=[],
    )
    result_pass = DemandValidator.validate(output=multi_clause_output, contract=contract_10m)
    assert result_pass.passed is True
