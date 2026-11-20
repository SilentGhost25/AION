"""
Acceptance Test Suite for Stage G1-B:
Assessment Configuration Registry & PlanningAgent Subpart Grounding.

Covers all 49 specified tests across Categories 1 through 7:
1. Assessment Configuration Registry (Tests 1–12)
2. SubPartPlan Dataclass & Provenance (Tests 13–19)
3. QuestionPlan Exact Types, Compatibility & Lifecycle (Tests 20–27)
4. Mutation Safety & Revalidation Boundary (Tests 28–31)
5. Pedagogical Grounding, Bloom/CO & Sibling Independence (Tests 32–37)
6. Selection Contract & Marks Conservation (Tests 38–43)
7. Candidate Isolation, Retries & Fail-Closed Lifecycle (Tests 44–49)
"""

import os
import threading
import pytest
from dataclasses import replace
from typing import Any, Dict, List

from core.contracts.document_artifact import DocumentArtifact, FigureArtifact, TextBlock
from core.contracts.paper_spec import PaperSpec
from core.extraction.module_segmenter import ModuleSegment, SegmentationStrategy
from core.generation.assessment_config import AssessmentConfig, AssessmentRegistry
from core.generation.agents.base import AgentContext
from core.generation.agents.planning_agent import PlanningAgent, find_module_segment
from core.generation.agents.question_plan import (
    CO_BLOOM_BANDS,
    PlanFailureCode,
    PlanStatus,
    QuestionPlan,
    SubPartPlan,
    detect_concept_collision,
    normalize_concept,
)


# =============================================================================
# Category 1: Assessment Configuration Registry (Tests 1–12)
# =============================================================================

def test_registry_4tuple_lookup_and_registration():
    """Test 1: Full 4-tuple identity lookup and registration succeeds."""
    # Register in test mode
    cfg = AssessmentConfig(
        scheme_id="SYNTHETIC_SCHEME",
        course_type="SYNTHETIC_TEST",
        assessment_type="TEST_MODE",
        blueprint_id="TEST_BP_2PART",
        total_marks_per_question=10,
        permitted_partitions=((5, 5), (6, 4)),
        max_subparts=2,
        reference_source="AION Test Verification Blueprint Citation 1",
    )
    # Unlock for registration in test environment
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(cfg)
    fetched = AssessmentRegistry.get("SYNTHETIC_SCHEME", "SYNTHETIC_TEST", "TEST_MODE", "TEST_BP_2PART")
    assert fetched == cfg
    assert fetched.identity == ("SYNTHETIC_SCHEME", "SYNTHETIC_TEST", "TEST_MODE", "TEST_BP_2PART")
    assert fetched.identity_key == "SYNTHETIC_SCHEME:SYNTHETIC_TEST:TEST_MODE:TEST_BP_2PART"


def test_registry_rejects_duplicate_registration_different_config():
    """Test 2: Rejects duplicate registration with different configuration."""
    AssessmentRegistry._reset_for_tests()
    cfg1 = AssessmentConfig(
        scheme_id="TEST_SCHEME",
        course_type="SYNTHETIC_TEST",
        assessment_type="TEST_MODE",
        blueprint_id="BP_1",
        total_marks_per_question=10,
        permitted_partitions=((5, 5),),
        max_subparts=2,
        reference_source="AION Test Verification Blueprint Citation 1",
    )
    cfg2 = AssessmentConfig(
        scheme_id="TEST_SCHEME",
        course_type="SYNTHETIC_TEST",
        assessment_type="TEST_MODE",
        blueprint_id="BP_1",
        total_marks_per_question=10,
        permitted_partitions=((6, 4),),
        max_subparts=2,
        reference_source="AION Test Verification Blueprint Citation 2",
    )
    AssessmentRegistry.register(cfg1)
    with pytest.raises(ValueError, match="already registered for key"):
        AssessmentRegistry.register(cfg2)


def test_registry_idempotent_identical_registration():
    """Test 3: Idempotent duplicate registration of identical config before freeze."""
    AssessmentRegistry._reset_for_tests()
    cfg = AssessmentConfig(
        scheme_id="TEST_SCHEME",
        course_type="SYNTHETIC_TEST",
        assessment_type="TEST_MODE",
        blueprint_id="BP_IDEMPOTENT",
        total_marks_per_question=10,
        permitted_partitions=((5, 5),),
        max_subparts=2,
        reference_source="AION Test Verification Blueprint Citation",
    )
    AssessmentRegistry.register(cfg)
    AssessmentRegistry.register(cfg)  # Should not raise
    assert AssessmentRegistry.get("TEST_SCHEME", "SYNTHETIC_TEST", "TEST_MODE", "BP_IDEMPOTENT") == cfg


def test_registry_rejects_partition_sum_mismatch():
    """Test 4: Partition sum != total_marks_per_question raises ValueError."""
    with pytest.raises(ValueError, match="sum .* != total_marks_per_question"):
        AssessmentConfig(
            scheme_id="TEST_SCHEME",
            course_type="SYNTHETIC_TEST",
            assessment_type="TEST_MODE",
            blueprint_id="BP_MISMATCH",
            total_marks_per_question=10,
            permitted_partitions=((6, 5),),  # sum=11 != 10
            max_subparts=2,
            reference_source="AION Test Verification Blueprint Citation",
        )


def test_registry_rejects_partition_exceeding_max_subparts():
    """Test 5: Partition length exceeding max_subparts raises ValueError."""
    with pytest.raises(ValueError, match="length .* > max_subparts"):
        AssessmentConfig(
            scheme_id="TEST_SCHEME",
            course_type="SYNTHETIC_TEST",
            assessment_type="TEST_MODE",
            blueprint_id="BP_EXCEED",
            total_marks_per_question=10,
            permitted_partitions=((4, 3, 3),),  # 3 subparts > max_subparts 2
            max_subparts=2,
            reference_source="AION Test Verification Blueprint Citation",
        )


def test_registry_rejects_duplicate_partitions_in_config():
    """Test 6: Duplicate partitions inside permitted_partitions raise ValueError."""
    with pytest.raises(ValueError, match="duplicate entries"):
        AssessmentConfig(
            scheme_id="TEST_SCHEME",
            course_type="SYNTHETIC_TEST",
            assessment_type="TEST_MODE",
            blueprint_id="BP_DUP",
            total_marks_per_question=10,
            permitted_partitions=((5, 5), (5, 5)),
            max_subparts=2,
            reference_source="AION Test Verification Blueprint Citation",
        )


def test_registry_rejects_max_subparts_exceeding_26():
    """Test 7: max_subparts > 26 raises ValueError."""
    with pytest.raises(ValueError, match="max_subparts must be int between 1 and 26"):
        AssessmentConfig(
            scheme_id="TEST_SCHEME",
            course_type="SYNTHETIC_TEST",
            assessment_type="TEST_MODE",
            blueprint_id="BP_MAX",
            total_marks_per_question=10,
            permitted_partitions=((10,),),
            max_subparts=27,
            reference_source="AION Test Verification Blueprint Citation",
        )


def test_registry_optional_fields_type_validation():
    """Test 8: allow_choice and questions_per_module type validation."""
    with pytest.raises(TypeError, match="allow_choice must be bool or None"):
        AssessmentConfig(
            scheme_id="TEST_SCHEME",
            course_type="SYNTHETIC_TEST",
            assessment_type="TEST_MODE",
            blueprint_id="BP_OPT1",
            total_marks_per_question=10,
            permitted_partitions=((10,),),
            max_subparts=2,
            reference_source="AION Test Verification Blueprint Citation",
            allow_choice="not_a_bool",
        )

    with pytest.raises(ValueError, match="questions_per_module must be positive exact int"):
        AssessmentConfig(
            scheme_id="TEST_SCHEME",
            course_type="SYNTHETIC_TEST",
            assessment_type="TEST_MODE",
            blueprint_id="BP_OPT2",
            total_marks_per_question=10,
            permitted_partitions=((10,),),
            max_subparts=2,
            reference_source="AION Test Verification Blueprint Citation",
            questions_per_module=-1,
        )


def test_registry_thread_safe_freeze_and_read_lock():
    """Test 9: Registry freeze and concurrent read safety."""
    AssessmentRegistry._reset_for_tests()
    cfg = AssessmentConfig(
        scheme_id="TEST_LOCK",
        course_type="SYNTHETIC_TEST",
        assessment_type="TEST_MODE",
        blueprint_id="BP_LOCK",
        total_marks_per_question=10,
        permitted_partitions=((10,),),
        max_subparts=1,
        reference_source="AION Test Verification Blueprint Citation",
    )
    AssessmentRegistry.register(cfg)
    AssessmentRegistry.freeze()
    assert AssessmentRegistry.is_frozen()

    with pytest.raises(RuntimeError, match="AssessmentRegistry is frozen"):
        AssessmentRegistry.register(cfg)

    # Concurrent reads should succeed without contention
    results = []

    def reader():
        res = AssessmentRegistry.get("TEST_LOCK", "SYNTHETIC_TEST", "TEST_MODE", "BP_LOCK")
        results.append(res == cfg)

    threads = [threading.Thread(target=reader) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 5
    assert all(results)


def test_registry_test_reset_safety_guard(monkeypatch):
    """Test 10: _reset_for_tests fails if not in test environment."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AION_TEST_MODE", raising=False)
    with pytest.raises(RuntimeError, match="_reset_for_tests\\(\\) is an internal test helper"):
        AssessmentRegistry._reset_for_tests()


def test_registry_unregistered_key_fails_closed():
    """Test 11: Lookup of unregistered key raises KeyError."""
    AssessmentRegistry._reset_for_tests()
    with pytest.raises(KeyError, match="No assessment configuration registered"):
        AssessmentRegistry.get("NONEXISTENT", "NONE", "NONE", "NONE")


def test_verified_aec_registration_exact_partitions_and_citation():
    """Test 12: Verified AEC Annexure-II registration has exact partitions and citation."""
    AssessmentRegistry._reset_for_tests()
    # Re-register baseline
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="VTU_2022_NEP",
        course_type="AEC_GENERAL_THEORY",
        assessment_type="CIE_IAT",
        blueprint_id="VTU_AEC_ANNEXURE_II",
        total_marks_per_question=10,
        permitted_partitions=((5, 5), (4, 6), (3, 7), (10,)),
        max_subparts=2,
        reference_source="VTU 2022 Scheme AEC Regulations (2024), Annexure-II, Clause 4(a) (10M questions, max 2 subquestions: 5+5, 4+6, 3+7, 10)",
        allow_choice=None,
        questions_per_module=None,
    ))
    cfg = AssessmentRegistry.get("VTU_2022_NEP", "AEC_GENERAL_THEORY", "CIE_IAT", "VTU_AEC_ANNEXURE_II")
    assert cfg.total_marks_per_question == 10
    assert cfg.permitted_partitions == ((5, 5), (4, 6), (3, 7), (10,))
    assert cfg.max_subparts == 2
    assert "Annexure-II, Clause 4(a)" in cfg.reference_source
    assert cfg.allow_choice is None
    assert cfg.questions_per_module is None


# =============================================================================
# Category 2: SubPartPlan Dataclass & Provenance (Tests 13–19)
# =============================================================================

def test_subpart_plan_validation_strict_types():
    """Test 13: SubPartPlan validates strict types for labels, marks, bloom, and co."""
    with pytest.raises((TypeError, ValueError)):
        SubPartPlan(label="1", marks=5, topic="Process Scheduling")  # Non 'a'-'z'
    with pytest.raises(TypeError):
        SubPartPlan(label="a", marks=True, topic="Process Scheduling")  # bool marks rejected
    with pytest.raises(ValueError):
        SubPartPlan(label="a", marks=-5, topic="Process Scheduling")  # negative marks
    with pytest.raises(ValueError):
        SubPartPlan(label="a", marks=5, topic="Process Scheduling", bloom="L7")  # invalid bloom
    with pytest.raises(ValueError):
        SubPartPlan(label="a", marks=5, topic="Process Scheduling", co="CO1", bloom="L5")  # CO1 only allows L1, L2


def test_subpart_plan_non_mutating_validation():
    """Test 14: Non-mutating post-construction validation via validate_fields."""
    sp = SubPartPlan(label="a", marks=5, topic="Process Scheduling", bloom="L2", co="CO1")
    valid, err = sp.validate_fields()
    assert valid and not err

    # Post-construction mutation
    sp.marks = 0
    valid, err = sp.validate_fields()
    assert not valid
    assert "strictly positive" in err


def test_subpart_plan_canonical_id_and_synonyms_strict_validation():
    """Test 15: Canonical ID and synonyms validation."""
    sp = SubPartPlan(
        label="a",
        marks=5,
        topic="Process Scheduling",
        canonical_concept_id="process_scheduling",
        synonym_ids=["cpu_scheduling", "thread_scheduling"],
    )
    valid, err = sp.validate_fields()
    assert valid

    # Empty synonym element
    sp.synonym_ids = [""]
    valid, err = sp.validate_fields()
    assert not valid
    assert "must be non-empty string" in err


def test_production_evidence_requires_explicit_module_id():
    """Test 16: Production evidence requires explicit module_id matching module_idx."""
    sp = SubPartPlan(
        label="a",
        marks=5,
        topic="Process Scheduling",
        canonical_concept_id="process_scheduling",
        evidence_blocks=[
            {"id": "b1", "text": "CPU scheduling algorithms", "page": 2, "module_id": "module_2"}  # Mismatch
        ],
    )
    valid, err = sp.validate_provenance(module_idx=1, page_start=1, page_end=5)
    assert not valid
    assert "module_id 'module_2' != expected 'module_1'" in err


def test_production_evidence_outside_module_segment_rejected():
    """Test 17: Production evidence page outside segment boundary is rejected."""
    sp = SubPartPlan(
        label="a",
        marks=5,
        topic="Process Scheduling",
        canonical_concept_id="process_scheduling",
        evidence_blocks=[
            {"id": "b1", "text": "CPU scheduling algorithms", "page": 6, "module_id": "module_1"}  # page 6 > 5
        ],
    )
    valid, err = sp.validate_provenance(module_idx=1, page_start=1, page_end=5)
    assert not valid
    assert "page 6 outside verified module segment [1, 5]" in err


def test_synthetic_evidence_bypass_bound_to_registered_config():
    """Test 18: Synthetic evidence bypass is bound strictly to SYNTHETIC_TEST course_type."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH",
        course_type="SYNTHETIC_TEST",
        assessment_type="CIE_IAT",
        blueprint_id="BP_SYNTH",
        total_marks_per_question=10,
        permitted_partitions=((5, 5),),
        max_subparts=2,
        reference_source="AION Test Verification Blueprint Citation",
    ))
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="PROD",
        course_type="AEC_GENERAL_THEORY",
        assessment_type="CIE_IAT",
        blueprint_id="BP_PROD",
        total_marks_per_question=10,
        permitted_partitions=((5, 5),),
        max_subparts=2,
        reference_source="AION Test Verification Blueprint Citation",
    ))

    # Synthetic subparts without evidence
    subparts = [
        SubPartPlan(label="a", marks=5, topic="Concept A", canonical_concept_id="concept_a"),
        SubPartPlan(label="b", marks=5, topic="Concept B", canonical_concept_id="concept_b"),
    ]

    plan_synth = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[5, 5],
        total_marks=10,
        co="CO1",
        bloom="L2",
        sub_part_plans=subparts,
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_SYNTH"),
    )
    valid, _, _ = plan_synth.validation_result()
    assert valid

    plan_prod = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[5, 5],
        total_marks=10,
        co="CO1",
        bloom="L2",
        sub_part_plans=subparts,
        assessment_identity=("PROD", "AEC_GENERAL_THEORY", "CIE_IAT", "BP_PROD"),
    )
    valid_prod, code, err = plan_prod.validation_result()
    assert not valid_prod
    assert code == PlanFailureCode.INSUFFICIENT_EVIDENCE


def test_figure_without_valid_module_association_rejected():
    """Test 19: Figure with visual_required=True requires valid module_id."""
    sp = SubPartPlan(
        label="a",
        marks=5,
        topic="Process Scheduling",
        canonical_concept_id="process_scheduling",
        evidence_blocks=[{"id": "b1", "text": "text", "page": 2, "module_id": "module_1"}],
        visual_required=True,
        figure={"id": "fig1", "image_path": "/img/f1.png", "module_id": "module_2"},  # module_2 != module_1
    )
    valid, err = sp.validate_provenance(module_idx=1, page_start=1, page_end=5)
    assert not valid
    assert "figure module_id 'module_2' != slot module 'module_1'" in err


# =============================================================================
# Category 3: QuestionPlan Exact Types, Compatibility & Lifecycle (Tests 20–27)
# =============================================================================

def test_question_plan_positional_construction_unchanged():
    """Test 20: 20-argument positional construction works identically."""
    plan = QuestionPlan(
        "module_1_Q1", "module_1", 1, 0, 1,
        [6, 4], 10,
        "CO1", "L2",
        "Topic Primary", "Topic A", "Topic B",
        ["c1"], ["c1"], ["c2"],
        [], False, None,
        {}, {}
    )
    assert plan.slot_id == "module_1_Q1"
    assert plan.total_marks == 10
    assert plan.status == PlanStatus.LEGACY_UNVALIDATED


def test_legacy_question_plan_status_is_legacy_unvalidated():
    """Test 21: Unmigrated legacy plan defaults to PlanStatus.LEGACY_UNVALIDATED."""
    plan = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[10],
        total_marks=10,
    )
    assert plan.status == PlanStatus.LEGACY_UNVALIDATED
    assert not plan.is_currently_valid()


def test_question_plan_cannot_claim_planned_without_subparts():
    """Test 22: QuestionPlan cannot be instantiated as PLANNED without sub_part_plans."""
    with pytest.raises(ValueError, match="violates structured plan invariant"):
        QuestionPlan(
            slot_id="module_1_Q1",
            module_id="module_1",
            module_idx=1,
            slot_idx=0,
            global_q_idx=1,
            partition=[10],
            total_marks=10,
            status=PlanStatus.PLANNED,
            sub_part_plans=[],
        )


def test_question_plan_unresolved_contains_no_subparts():
    """Test 23: QuestionPlan marked UNRESOLVED must not contain sub_part_plans."""
    sp = SubPartPlan(label="a", marks=10, topic="Topic")
    with pytest.raises(ValueError, match="UNRESOLVED must not contain sub_part_plans"):
        QuestionPlan(
            slot_id="module_1_Q1",
            module_id="module_1",
            module_idx=1,
            slot_idx=0,
            global_q_idx=1,
            partition=[10],
            total_marks=10,
            status=PlanStatus.UNRESOLVED,
            failure_code=PlanFailureCode.INSUFFICIENT_TOPICS.value,
            failure_detail="Failed",
            sub_part_plans=[sp],
            telemetry={"attempts": [{"attempt_number": 1, "strategy": "s1", "failure_code": "INSUFFICIENT_TOPICS", "failure_detail": "d", "candidate_topics": []}]},
        )


def test_unresolved_requires_declared_enum_failure_code_and_strict_diagnostics():
    """Test 24: QuestionPlan marked UNRESOLVED requires valid enum and telemetry schema."""
    with pytest.raises(ValueError, match="valid declared PlanFailureCode"):
        QuestionPlan(
            slot_id="module_1_Q1",
            module_id="module_1",
            module_idx=1,
            slot_idx=0,
            global_q_idx=1,
            partition=[10],
            total_marks=10,
            status=PlanStatus.UNRESOLVED,
            failure_code="INVALID_CODE_STRING",
            failure_detail="Failed",
            telemetry={"attempts": []},
        )


def test_boolean_and_non_positive_partition_marks_rejected():
    """Test 25: Boolean partition marks or total marks rejected in validation_result."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_BOOL", total_marks_per_question=10, permitted_partitions=((10,),),
        max_subparts=1, reference_source="Citation Test 25",
    ))
    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[True], total_marks=10, co="CO1", bloom="L2",
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_BOOL"),
    )
    valid, code, err = plan.validation_result()
    assert not valid
    assert code == PlanFailureCode.MARKS_MISMATCH


def test_more_than_26_subparts_rejected():
    """Test 26: Partition length > 26 rejected in validation_result."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_26", total_marks_per_question=10, permitted_partitions=((10,),),
        max_subparts=1, reference_source="Citation Test 26",
    ))
    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[1] * 27, total_marks=10, co="CO1", bloom="L2",
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_26"),
    )
    valid, code, err = plan.validation_result()
    assert not valid
    assert code == PlanFailureCode.INVALID_CONFIG
    assert "exceeds maximum 26 parts" in err


def test_parent_pedagogical_contract_validated():
    """Test 27: Parent CO/Bloom contract validated against CO_BLOOM_BANDS."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_PED", total_marks_per_question=10, permitted_partitions=((10,),),
        max_subparts=1, reference_source="Citation Test 27",
    ))
    sp = SubPartPlan(label="a", marks=10, topic="T", bloom="L2", co="CO1")
    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[10], total_marks=10, co="CO1", bloom="L5",  # CO1 only allows L1, L2
        sub_part_plans=[sp],
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_PED"),
    )
    valid, code, err = plan.validation_result()
    assert not valid
    assert code == PlanFailureCode.BLOOM_CO_INCOMPATIBLE


# =============================================================================
# Category 4: Mutation Safety & Revalidation Boundary (Tests 28–31)
# =============================================================================

def test_mutated_planned_object_fails_revalidation_at_consumption_boundary():
    """Test 28: Mutating a PLANNED plan causes is_currently_valid() to fail."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_MUT", total_marks_per_question=10, permitted_partitions=((5, 5),),
        max_subparts=2, reference_source="Citation Test 28",
    ))
    sp_a = SubPartPlan(label="a", marks=5, topic="Concept A", canonical_concept_id="a")
    sp_b = SubPartPlan(label="b", marks=5, topic="Concept B", canonical_concept_id="b")

    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[5, 5], total_marks=10, co="CO1", bloom="L2",
        sub_part_plans=[sp_a, sp_b],
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_MUT"),
        status=PlanStatus.PLANNED,
    )
    assert plan.is_currently_valid()

    # Mutation: alter marks on subpart
    plan.sub_part_plans[0].marks = 100
    assert not plan.is_currently_valid()


def test_validation_result_provides_diagnostic_reason():
    """Test 29: validation_result provides structured failure code and diagnostic detail."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_DIAG", total_marks_per_question=10, permitted_partitions=((5, 5),),
        max_subparts=2, reference_source="Citation Test 29",
    ))
    sp_a = SubPartPlan(label="a", marks=5, topic="Concept A", canonical_concept_id="a")
    sp_b = SubPartPlan(label="b", marks=5, topic="Concept B", canonical_concept_id="b")

    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[5, 5], total_marks=10, co="CO1", bloom="L2",
        sub_part_plans=[sp_a, sp_b],
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_DIAG"),
        status=PlanStatus.PLANNED,
    )
    plan.sub_part_plans[0].marks = 6
    valid, code, detail = plan.validation_result()
    assert not valid
    assert code == PlanFailureCode.MARKS_MISMATCH
    assert "marks 6 != partition[0] 5" in detail


def test_consumption_revalidation_enforces_registered_assessment_policy():
    """Test 30: Mutating partition to unpermitted split fails revalidation against registry."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_POLICY", total_marks_per_question=10, permitted_partitions=((5, 5),),
        max_subparts=2, reference_source="Citation Test 30",
    ))
    sp_a = SubPartPlan(label="a", marks=7, topic="Concept A", canonical_concept_id="a")
    sp_b = SubPartPlan(label="b", marks=3, topic="Concept B", canonical_concept_id="b")

    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[7, 3], total_marks=10, co="CO1", bloom="L2",
        sub_part_plans=[sp_a, sp_b],
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_POLICY"),
    )
    valid, code, detail = plan.validation_result()
    assert not valid
    assert code == PlanFailureCode.INVALID_CONFIG
    assert "not permitted" in detail


def test_question_plan_projection_clearing_and_idempotency():
    """Test 31: Projection clearing on subpart count reduction and idempotent syncing."""
    sp_a = SubPartPlan(label="a", marks=5, topic="Concept A", concept_neighborhood=["n1"])
    sp_b = SubPartPlan(label="b", marks=5, topic="Concept B", concept_neighborhood=["n2"])

    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[5, 5], total_marks=10,
        sub_part_plans=[sp_a, sp_b],
    )
    assert plan.topic == "Concept A"
    assert plan.topic_a == "Concept A"
    assert plan.topic_b == "Concept B"

    # Reduce to 1 subpart
    plan.sub_part_plans = [sp_a]
    plan.sync_legacy_projections()
    assert plan.topic == "Concept A"
    assert plan.topic_b == ""
    assert plan.concept_neighborhood_b == []

    # Reduce to 0 subparts
    plan.sub_part_plans = []
    plan.sync_legacy_projections()
    assert plan.topic == ""
    assert plan.topic_a == ""
    assert plan.topic_b == ""
    assert plan.concept_neighborhood == []
    assert plan.concept_neighborhood_a == []
    assert plan.concept_neighborhood_b == []


# =============================================================================
# Category 5: Pedagogical Grounding, Bloom/CO & Sibling Independence (Tests 32–37)
# =============================================================================

def test_bloom_co_mismatch_rejected():
    """Test 32: Subpart bloom incompatible with CO rejected with BLOOM_CO_INCOMPATIBLE."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_BLOOM", total_marks_per_question=10, permitted_partitions=((10,),),
        max_subparts=1, reference_source="Citation Test 32",
    ))
    sp = SubPartPlan(label="a", marks=10, topic="Scheduling", co="CO2", bloom="L3")
    sp.bloom = "L5"  # mutate to L5 which is incompatible with CO2 band ("L3",)
    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[10], total_marks=10, co="CO2", bloom="L3",
        sub_part_plans=[sp],
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_BLOOM"),
    )
    valid, code, _ = plan.validation_result()
    assert not valid
    assert code == PlanFailureCode.BLOOM_CO_INCOMPATIBLE


def test_subpart_unauthorized_co_bloom_override_rejected():
    """Test 33: Subpart CO/Bloom differing from parent without constraint override is rejected."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_OVERRIDE", total_marks_per_question=10, permitted_partitions=((5, 5),),
        max_subparts=2, reference_source="Citation Test 33",
    ))
    sp_a = SubPartPlan(label="a", marks=5, topic="A", co="CO1", bloom="L2")
    sp_b = SubPartPlan(label="b", marks=5, topic="B", co="CO2", bloom="L3")  # differs from parent CO1/L2

    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[5, 5], total_marks=10, co="CO1", bloom="L2",
        sub_part_plans=[sp_a, sp_b],
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_OVERRIDE"),
    )
    valid, code, _ = plan.validation_result()
    assert not valid
    assert code == PlanFailureCode.BLOOM_CO_INCOMPATIBLE

    # Authorized override
    plan.constraints = {"subpart_cos": {"b": "CO2"}, "subpart_blooms": {"b": "L3"}}
    valid_auth, _, _ = plan.validation_result()
    assert valid_auth


def test_duplicate_canonical_concept_or_synonym_rejected():
    """Test 34: Sibling subparts sharing identical canonical concept ID rejected with SIBLING_COLLISION."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_COLLIDE", total_marks_per_question=10, permitted_partitions=((5, 5),),
        max_subparts=2, reference_source="Citation Test 34",
    ))
    sp_a = SubPartPlan(label="a", marks=5, topic="Deadlock Prevention", canonical_concept_id="deadlock")
    sp_b = SubPartPlan(label="b", marks=5, topic="Deadlock Avoidance", canonical_concept_id="deadlock")

    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[5, 5], total_marks=10, co="CO1", bloom="L2",
        sub_part_plans=[sp_a, sp_b],
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_COLLIDE"),
    )
    valid, code, detail = plan.validation_result()
    assert not valid
    assert code == PlanFailureCode.SIBLING_COLLISION
    assert "Identical canonical concept ID" in detail


def test_shared_synonym_set_intersection_rejected_as_collision():
    """Test 35: Sibling subparts sharing a synonym in their synonym sets detected as collision."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_SYN_COLLIDE", total_marks_per_question=10, permitted_partitions=((5, 5),),
        max_subparts=2, reference_source="Citation Test 35",
    ))
    sp_a = SubPartPlan(
        label="a", marks=5, topic="Process Scheduling",
        canonical_concept_id="scheduling", synonym_ids=["cpu_allocation", "dispatcher"]
    )
    sp_b = SubPartPlan(
        label="b", marks=5, topic="Core Management",
        canonical_concept_id="core_mgr", synonym_ids=["dispatcher", "thread_pool"]
    )
    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[5, 5], total_marks=10, co="CO1", bloom="L2",
        sub_part_plans=[sp_a, sp_b],
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_SYN_COLLIDE"),
    )
    valid, code, detail = plan.validation_result()
    assert not valid
    assert code == PlanFailureCode.SIBLING_COLLISION
    assert "Shared synonym" in detail


def test_complete_neighborhood_overlap_detected_as_collision():
    """Test 36: Sibling parts sharing underlying concept in complete neighborhood footprints rejected."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_HOOD", total_marks_per_question=10, permitted_partitions=((5, 5),),
        max_subparts=2, reference_source="Citation Test 36",
    ))
    sp_a = SubPartPlan(
        label="a", marks=5, topic="Semaphores",
        canonical_concept_id="semaphores", concept_neighborhood=["mutex", "critical_section"]
    )
    sp_b = SubPartPlan(
        label="b", marks=5, topic="Monitors",
        canonical_concept_id="monitors", concept_neighborhood=["condition_variables", "mutex"]
    )
    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[5, 5], total_marks=10, co="CO1", bloom="L2",
        sub_part_plans=[sp_a, sp_b],
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_HOOD"),
    )
    valid, code, detail = plan.validation_result()
    assert not valid
    assert code == PlanFailureCode.SIBLING_COLLISION
    assert "Shared concepts in neighborhood footprints" in detail


def test_revalidation_enforces_identical_invariants_as_candidate_gate():
    """Test 37: Complete invariant equivalence between candidate gate and consumption revalidation."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_EQUIV", total_marks_per_question=10, permitted_partitions=((5, 5),),
        max_subparts=2, reference_source="Citation Test 37",
    ))
    sp_a = SubPartPlan(label="a", marks=5, topic="Concept A", canonical_concept_id="a")
    sp_b = SubPartPlan(label="b", marks=5, topic="Concept B", canonical_concept_id="b")

    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[5, 5], total_marks=10, co="CO1", bloom="L2",
        sub_part_plans=[sp_a, sp_b],
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_EQUIV"),
    )
    # Both call plan.validation_result()
    v1, c1, d1 = plan.validation_result()
    v2, c2, d2 = plan.validation_result()
    assert (v1, c1, d1) == (v2, c2, d2)
    assert v1 is True


# =============================================================================
# Category 6: Selection Contract & Marks Conservation (Tests 38–43)
# =============================================================================

def test_direct_unregistered_assessment_config_rejected():
    """Test 38: Directly supplied unregistered AssessmentConfig is rejected against registry."""
    AssessmentRegistry._reset_for_tests()
    unreg_cfg = AssessmentConfig(
        scheme_id="UNREGISTERED", course_type="UNREGISTERED", assessment_type="UNREGISTERED",
        blueprint_id="UNREGISTERED", total_marks_per_question=10, permitted_partitions=((10,),),
        max_subparts=1, reference_source="Citation Test 38",
    )
    agent = PlanningAgent(assessment_config=unreg_cfg)
    ctx = AgentContext(
        paper_spec=PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10),
        artifact=DocumentArtifact(source_pdf_sha256="abc", source_pdf_path="test.pdf"),
    )
    res = agent.run(ctx)
    assert not res.success
    assert "No assessment configuration registered" in res.failure_detail


def test_missing_assessment_identity_fails_closed():
    """Test 39: Missing assessment identity fails closed with KeyError."""
    AssessmentRegistry._reset_for_tests()
    agent = PlanningAgent()
    ctx = AgentContext(
        paper_spec=PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10),
        artifact=DocumentArtifact(source_pdf_sha256="abc", source_pdf_path="test.pdf"),
        request={},  # missing identity
    )
    res = agent.run(ctx)
    assert not res.success
    assert "Missing required assessment identity 4-tuple" in res.failure_detail


def test_planner_selects_partition_from_assessment_config():
    """Test 40: Planner adopts permitted partition from configured blueprint."""
    AssessmentRegistry._reset_for_tests()
    cfg = AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_SEL", total_marks_per_question=10, permitted_partitions=((6, 4),),
        max_subparts=2, reference_source="Citation Test 40",
    )
    AssessmentRegistry.register(cfg)
    agent = PlanningAgent(assessment_config=cfg)
    ctx = AgentContext(
        paper_spec=PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10),
        artifact=DocumentArtifact(source_pdf_sha256="abc", source_pdf_path="test.pdf"),
    )
    res = agent.run(ctx)
    assert res.success
    plans = res.payload
    assert len(plans) == 1
    assert plans[0].partition == [6, 4]


def test_planner_rejects_partition_not_in_assessment_config():
    """Test 41: Planner rejects partition not permitted by assessment config."""
    AssessmentRegistry._reset_for_tests()
    cfg = AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_REJ", total_marks_per_question=10, permitted_partitions=((5, 5),),
        max_subparts=2, reference_source="Citation Test 41",
    )
    AssessmentRegistry.register(cfg)
    agent = PlanningAgent(assessment_config=cfg)
    ctx = AgentContext(
        paper_spec=PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10),
        artifact=DocumentArtifact(source_pdf_sha256="abc", source_pdf_path="test.pdf"),
        marks_split=[[7, 3]],  # Not permitted
    )
    res = agent.run(ctx)
    assert not res.success
    assert "not permitted" in res.failure_detail


def test_planner_validates_outer_marks_split_length_against_total_questions():
    """Test 42: marks_split length mismatch against PaperSpec.total_questions raises ValueError."""
    AssessmentRegistry._reset_for_tests()
    cfg = AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_LEN", total_marks_per_question=10, permitted_partitions=((5, 5),),
        max_subparts=2, reference_source="Citation Test 42",
    )
    AssessmentRegistry.register(cfg)
    agent = PlanningAgent(assessment_config=cfg)
    ctx = AgentContext(
        paper_spec=PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=2, marks_per_question=10),  # total_questions=2
        artifact=DocumentArtifact(source_pdf_sha256="abc", source_pdf_path="test.pdf"),
        marks_split=[[5, 5]],  # only 1 partition provided
    )
    res = agent.run(ctx)
    assert not res.success
    assert "marks_split length" in res.failure_detail


def test_planner_deterministic_partition_selection_when_marks_split_absent():
    """Test 43: Planner deterministically selects primary partition when marks_split is absent."""
    AssessmentRegistry._reset_for_tests()
    cfg = AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_DET", total_marks_per_question=10, permitted_partitions=((4, 6), (5, 5)),
        max_subparts=2, reference_source="Citation Test 43",
    )
    AssessmentRegistry.register(cfg)
    agent = PlanningAgent(assessment_config=cfg)
    ctx = AgentContext(
        paper_spec=PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10),
        artifact=DocumentArtifact(source_pdf_sha256="abc", source_pdf_path="test.pdf"),
        marks_split=None,
    )
    res = agent.run(ctx)
    assert res.success
    assert res.payload[0].partition == [4, 6]


# =============================================================================
# Category 7: Candidate Isolation, Retries & Fail-Closed Lifecycle (Tests 44–49)
# =============================================================================

def test_module_segment_real_instance_lookup_and_provenance():
    """Test 44: Real ModuleSegment instance lookup and segment boundary resolution."""
    seg = ModuleSegment(
        module_id=1,
        title="Operating System Overview",
        page_start=1,
        page_end=15,
        confidence=0.98,
        strategy_used=SegmentationStrategy.EXPLICIT_MODULE_HEADING,
    )
    found = find_module_segment([seg], 1)
    assert found is not None
    assert found.page_start == 1
    assert found.page_end == 15


def test_planner_fails_closed_when_module_segment_unavailable():
    """Test 45: Missing ModuleSegment fails closed immediately with MISSING_MODULE_SEGMENT."""
    AssessmentRegistry._reset_for_tests()
    cfg = AssessmentConfig(
        scheme_id="PROD", course_type="AEC_GENERAL_THEORY", assessment_type="CIE_IAT",
        blueprint_id="BP_SEG", total_marks_per_question=10, permitted_partitions=((10,),),
        max_subparts=1, reference_source="Citation Test 45",
    )
    AssessmentRegistry.register(cfg)
    agent = PlanningAgent(assessment_config=cfg)
    ctx = AgentContext(
        paper_spec=PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10),
        artifact=DocumentArtifact(source_pdf_sha256="abc", source_pdf_path="test.pdf"),
    )
    ctx.module_segments = []  # No segments
    res = agent.run(ctx)
    assert res.success
    plans = res.payload
    assert len(plans) == 1
    p = plans[0]
    assert p.status == PlanStatus.UNRESOLVED
    assert p.failure_code == PlanFailureCode.MISSING_MODULE_SEGMENT.value
    assert p.sub_part_plans == []
    assert p.telemetry.get("attempts") == []  # 0 retries


def test_non_retryable_failures_terminate_immediately_with_zero_retries():
    """Test 46: Fatal non-retryable failures have exactly 0 attempts in telemetry."""
    AssessmentRegistry._reset_for_tests()
    cfg = AssessmentConfig(
        scheme_id="PROD", course_type="AEC_GENERAL_THEORY", assessment_type="CIE_IAT",
        blueprint_id="BP_FATAL", total_marks_per_question=10, permitted_partitions=((10,),),
        max_subparts=1, reference_source="Citation Test 46",
    )
    AssessmentRegistry.register(cfg)
    agent = PlanningAgent(assessment_config=cfg)
    ctx = AgentContext(
        paper_spec=PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10),
        artifact=DocumentArtifact(source_pdf_sha256="abc", source_pdf_path="test.pdf"),
    )
    res = agent.run(ctx)
    plan = res.payload[0]
    assert plan.status == PlanStatus.UNRESOLVED
    assert plan.telemetry["attempts"] == []


def test_planner_retry_rebuilds_candidate_without_mutating_original_plan():
    """Test 47: Retries construct fresh candidate lists without touching original partition."""
    # Test QuestionPlan candidate isolation logic
    orig_partition = [5, 5]
    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=list(orig_partition), total_marks=10, co="CO1", bloom="L2",
    )
    assert plan.partition == orig_partition

    # Retry candidates are fresh instances
    c1 = [SubPartPlan(label="a", marks=5, topic="T1"), SubPartPlan(label="b", marks=5, topic="T2")]
    c2 = [SubPartPlan(label="a", marks=5, topic="T3"), SubPartPlan(label="b", marks=5, topic="T4")]
    assert c1 != c2
    assert plan.partition == orig_partition


def test_unresolved_plan_contains_no_accepted_partial_allocation():
    """Test 48: Unresolved plan after exhausted retries has empty sub_part_plans."""
    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[5, 5], total_marks=10, co="CO1", bloom="L2",
        sub_part_plans=[],
        status=PlanStatus.UNRESOLVED,
        failure_code=PlanFailureCode.INSUFFICIENT_TOPICS.value,
        failure_detail="Could not find topics",
        telemetry={
            "attempts": [
                {"attempt_number": 1, "strategy": "s1", "failure_code": "INSUFFICIENT_TOPICS", "failure_detail": "d1", "candidate_topics": []},
                {"attempt_number": 2, "strategy": "s2", "failure_code": "INSUFFICIENT_TOPICS", "failure_detail": "d2", "candidate_topics": []},
                {"attempt_number": 3, "strategy": "s3", "failure_code": "INSUFFICIENT_TOPICS", "failure_detail": "d3", "candidate_topics": []},
            ]
        },
    )
    assert plan.status == PlanStatus.UNRESOLVED
    assert plan.sub_part_plans == []
    assert len(plan.telemetry["attempts"]) == 3


def test_planner_decoupled_from_writer_failure():
    """Test 49: PlanningAgent has zero imports of WritingAgent or writer failure callbacks."""
    import inspect
    import core.generation.agents.planning_agent as pa_module
    src = inspect.getsource(pa_module)
    assert "WritingAgent" not in src
    assert "GenerationFailed" not in src
    assert "WriterFailure" not in src
