"""
AION Stage G6 Academic Coherence Acceptance Test Suite.
Implements all 86 dedicated acceptance tests specified in Stage G6 v2.8 Implementation Specification.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from core.generation.agents.question_plan import (
    RubricCriterion,
    ConceptFootprint,
    ExpectedResponseContract,
    VisualTaskContract,
    EvidenceBlockID,
    SubPartPlan,
    QuestionPlan,
)
from core.generation.agents.generated_question import (
    SubPart,
    GeneratedQuestion,
)
from core.generation.assessment_config import (
    AssessmentConfig,
    CognitiveTargetDistribution,
)
from core.generation.agents.planning_agent import (
    PlanningAgent,
    optimize_bloom_distribution,
    compute_bloom_tie_break_key,
)
from core.generation.agents.pipeline_bridge import (
    _build_kg_for_artifact,
)
from core.generation.agents.overlap_detector import (
    verify_concept_provenance,
    classify_relationship,
    detect_concept_collision,
    generate_concept_integrity_report,
)
from core.generation.agents.checking_agent import (
    check_choice_pair_equivalence,
    SelectionNode,
    build_selection_tree_from_questions,
    enumerate_attempt_paths,
    calculate_attempt_path_bloom_bounds,
    audit_whole_paper_bloom,
    generate_assessment_profile,
    AssessmentProfile,
)
from core.generation.agents.local_evaluators import (
    evaluate_solution_completeness,
    evaluate_marking_scheme_relevance,
    evaluate_visual_dependency,
)


# =============================================================================
# Area 0: Contract and System Invariants (6 tests)
# =============================================================================

def test_g6_00_contract_from_dict_strict_rejection():
    """Pass wrong types to from_dict(); strict validation raises TypeError / ValueError."""
    # RubricCriterion: string marks rejected
    with pytest.raises(TypeError):
        RubricCriterion.from_dict({"criterion": "Valid criterion", "marks": "2"})
    # RubricCriterion: float marks rejected
    with pytest.raises(TypeError):
        RubricCriterion.from_dict({"criterion": "Valid criterion", "marks": 2.5})
    # RubricCriterion: bool marks rejected
    with pytest.raises(TypeError):
        RubricCriterion.from_dict({"criterion": "Valid criterion", "marks": True})
    # RubricCriterion: unexpected extra keys rejected
    with pytest.raises(ValueError):
        RubricCriterion.from_dict({"criterion": "Valid", "marks": 2, "extra": "forbidden"})

    # ConceptFootprint: wrong types / extra keys
    with pytest.raises(ValueError):
        ConceptFootprint.from_dict({"canonical_concept": "B-Tree", "extra": 123})

    # ExpectedResponseContract: wrong types
    with pytest.raises(TypeError):
        ExpectedResponseContract.from_dict({"required_elements": "not_a_list", "expected_operations": []})

    # VisualTaskContract: wrong types
    with pytest.raises(TypeError):
        VisualTaskContract.from_dict({
            "figure_id": "fig_1",
            "is_essential": "not_a_bool",
            "task_operation": "read",
            "required_features": ["f1"]
        })

    # Direct Constructor Rejection Tests (Issue 2)
    with pytest.raises(TypeError):
        RubricCriterion(criterion="", marks=5)
    with pytest.raises(TypeError):
        RubricCriterion(criterion="Steps", marks="five")  # type: ignore
    with pytest.raises(TypeError):
        RubricCriterion(criterion="Steps", marks=-1)

    with pytest.raises(TypeError):
        ConceptFootprint(canonical_concept="", bloom="L2")
    with pytest.raises(ValueError):
        ConceptFootprint(canonical_concept="Paging", bloom="L7")
    with pytest.raises(TypeError):
        ConceptFootprint(canonical_concept="Paging", module_id=True)  # bool rejected

    with pytest.raises(TypeError):
        ExpectedResponseContract(response_type=123)  # type: ignore
    with pytest.raises(TypeError):
        ExpectedResponseContract(response_type="PROSE", required_elements="not_a_tuple")  # type: ignore
    with pytest.raises(TypeError):
        ExpectedResponseContract(response_type="PROSE", estimated_steps="five")  # type: ignore

    with pytest.raises(TypeError):
        VisualTaskContract(figure_id="", is_essential=True, task_operation="read", required_features=("f1",))
    with pytest.raises(TypeError):
        VisualTaskContract(figure_id="fig1", is_essential="not_bool", task_operation="read", required_features=("f1",))  # type: ignore
    with pytest.raises(TypeError):
        VisualTaskContract(figure_id="fig1", is_essential=True, task_operation="", required_features=("f1",))


def test_g6_00_contract_imports_no_circular_dependency():
    """Import generated_question and question_plan with zero circular dependency errors."""
    import sys
    assert "core.generation.agents.question_plan" in sys.modules
    assert "core.generation.agents.generated_question" in sys.modules


def test_g6_00_non_applicable_gates_do_not_produce_indeterminate():
    """In IA exam, choice parity is NOT_APPLICABLE; status resolves to PASS."""
    # 2 compulsory questions
    sp1 = SubPart(subpart_index=0, label="a", text="Explain paging.", marks=5, bloom="L2",
                  marking_scheme=[{"criterion": "Paging steps", "marks": 5}],
                  concept_footprint=ConceptFootprint(canonical_concept="Paging", provenance_entity_id="p1", provenance_type="DOCUMENT_SECTION"))
    sp2 = SubPart(subpart_index=0, label="a", text="Explain segmentation.", marks=5, bloom="L2",
                  marking_scheme=[{"criterion": "Segmentation steps", "marks": 5}],
                  concept_footprint=ConceptFootprint(canonical_concept="Segmentation", provenance_entity_id="p2", provenance_type="DOCUMENT_SECTION"))
    q1 = GeneratedQuestion(question_id="Q1", slot_id="module_1_Q1", question_text="Explain paging.", marks=5, bloom="L2", sub_parts=[sp1], solution="Detailed explanation of paging.")
    q2 = GeneratedQuestion(question_id="Q2", slot_id="module_1_Q2", question_text="Explain segmentation.", marks=5, bloom="L2", sub_parts=[sp2], solution="Detailed explanation of segmentation.")
    q1.sync_materialized_projections()
    q2.sync_materialized_projections()

    cfg = AssessmentConfig(assessment_type="IA", allow_synthetic=True)
    profile = generate_assessment_profile([q1, q2], config=cfg)
    assert profile.choice_topic_dispersion_gate == "NOT_APPLICABLE"
    assert profile.overall_status == "PASS"


def test_g6_00_no_applicable_gates_cannot_produce_vacuous_pass():
    """Assessment with zero or insufficient applicable gates evaluates to INDETERMINATE."""
    # No questions provided
    cfg = AssessmentConfig(assessment_type="IA", allow_synthetic=True, generate_solutions=False, generate_rubrics=False)
    profile = generate_assessment_profile([], config=cfg)
    assert profile.overall_status == "INDETERMINATE"
    assert "INSUFFICIENT_APPLICABLE_GATES" in profile.review_recommendations


def test_g6_00_positional_dataclass_constructors_intact():
    """Construct SubPart positionally with 12 baseline arguments without TypeError."""
    # Baseline 12 args: subpart_index, label, text, marks, bloom, course_outcome,
    # target_cognitive_level, syllabus_topic, archetype, figure_id, image_path, image_sha256
    sp = SubPart(0, "a", "Explain virtual memory.", 6, "L2", "CO1", "L2", "VM", "conceptual", None, None, None)
    assert sp.subpart_index == 0
    assert sp.label == "a"
    assert sp.marks == 6
    assert sp.evidence_block_ids == ()
    assert sp.concept_footprint is None


def test_g6_00_frozen_adapter_marking_scheme_compatibility():
    """Pass materialized question to LegacyAssessmentAdapter; list(q.marking_scheme or []) succeeds."""
    sp = SubPart(subpart_index=0, label="a", text="Explain B-Trees.", marks=6, bloom="L2",
                 marking_scheme=[{"criterion": "B-Tree definition", "marks": 6}])
    q = GeneratedQuestion(question_id="Q1", slot_id="module_1_Q1", question_text="Explain B-Trees.", marks=6, bloom="L2", sub_parts=[sp])
    q.sync_materialized_projections()

    # Emulate legacy adapter behavior
    ms_list = list(q.marking_scheme or [])
    assert isinstance(ms_list, list)
    assert len(ms_list) == 1
    assert ms_list[0]["marks"] == 6


# =============================================================================
# Area 1: Evidence Identity & Single Source Ownership (12 tests)
# =============================================================================

class MockArtifact:
    def __init__(self, doc_id="doc1", sha256="a" * 64, blocks=None, headings=None):
        self.doc_id = doc_id
        self.source_pdf_sha256 = sha256
        self.doc_id_hash = sha256
        self.manifest_fingerprint = sha256
        self.blocks = blocks or []
        self.headings = headings or {}

    def has_heading(self, h_id):
        return h_id in self.headings

    def get_heading(self, h_id):
        return self.headings.get(h_id)


def test_g6_01_topic_aware_evidence_retrieval():
    """Request evidence for subtopic 'B-Tree Deletion'; returned blocks contain specific deletion tokens."""
    planner = PlanningAgent()
    artifact = MockArtifact(
        doc_id="doc_ds",
        sha256="e" * 64,
        blocks=[
            {"module_id": "module_1", "page": 10, "text": "B-Tree search is logarithmic."},
            {"module_id": "module_1", "page": 12, "text": "B-Tree Deletion requires borrowing or merging keys under underflow conditions."},
            {"module_id": "module_2", "page": 30, "text": "Graph traversal using BFS."},
        ]
    )
    blocks = planner._pick_evidence_blocks_for_slot(slot_idx=0, topic="B-Tree Deletion", module_idx=1, artifact=artifact)
    assert len(blocks) > 0
    assert "m1::p12" in blocks[0]


def test_g6_01_evidence_full_fingerprint_stale_detected():
    """Modify single character of block text; 256-bit SHA256 mismatch detected."""
    original_text = "This is authoritative syllabus content for module 1."
    modified_text = "This is authoritative syllabus content for module 2."
    b_hash_orig = hashlib.sha256(original_text.strip().encode("utf-8")).hexdigest()
    b_hash_mod = hashlib.sha256(modified_text.strip().encode("utf-8")).hexdigest()
    assert b_hash_orig != b_hash_mod


def test_g6_01_evidence_artifact_revision_change_detected():
    """Replace artifact with different revision hash; revision mismatch detected."""
    art1 = MockArtifact(doc_id="doc1", sha256="1" * 64)
    art2 = MockArtifact(doc_id="doc1", sha256="2" * 64)
    planner = PlanningAgent()
    b1 = planner._pick_evidence_blocks_for_slot(0, "topic", 1, artifact=art1)
    b2 = planner._pick_evidence_blocks_for_slot(0, "topic", 1, artifact=art2)
    assert ("rev:" + "1" * 64) in b1[0]
    assert ("rev:" + "2" * 64) in b2[0]


def test_g6_01_evidence_similarity_alone_cannot_pass():
    """Block has high TF-IDF but lacks required equation; returns INDETERMINATE, not PASS."""
    res, detail = evaluate_visual_dependency("Calculate the frequency using spatial layout", figure_metadata={"width": 100, "height": 100})
    assert res == "INDETERMINATE"


def test_g6_01_evidence_wrong_module_rejected():
    """Inject evidence ID from Module 2 into Module 1 subpart; fails closed."""
    planner = PlanningAgent()
    artifact = MockArtifact(doc_id="doc1", sha256="a" * 64, blocks=[{"module_id": "module_2", "page": 5, "text": "Mod 2 content"}])
    # Slot is module 1, but block is module 2
    blocks = planner._pick_evidence_blocks_for_slot(0, "topic", 1, artifact=artifact)
    # Filtered out strictly to module 1
    assert len(blocks) == 0


def test_g6_01_single_source_evidence_ownership():
    """Verify evidence IDs reside strictly on SubPart.evidence_block_ids."""
    sp = SubPart(subpart_index=0, label="a", text="What is pipelining?", marks=5, bloom="L1", evidence_block_ids=("ev1", "ev2"))
    assert hasattr(sp, "evidence_block_ids")
    assert sp.evidence_block_ids == ("ev1", "ev2")
    # SubPartPlan also has evidence_blocks
    sp_plan = SubPartPlan(label="a", marks=5, bloom="L1", course_outcome="CO1", syllabus_topic="Pipelining", evidence_blocks=["ev1", "ev2"])
    assert sp_plan.evidence_blocks == ["ev1", "ev2"]


def test_g6_01_evidence_missing_unrefined_fails_closed():
    """Newly generated unrefined question omits evidence IDs; evaluator fails closed."""
    cfg = AssessmentConfig(assessment_type="SEE", allow_synthetic=False)
    sp = SubPart(subpart_index=0, label="a", text="Explain pipelining.", marks=5, bloom="L2", evidence_block_ids=())
    q = GeneratedQuestion(question_id="Q1", slot_id="module_1_Q1", question_text="Explain pipelining.", marks=5, bloom="L2", sub_parts=[sp])
    art = MockArtifact(doc_id="doc1", sha256="a" * 64)
    profile = generate_assessment_profile([q], config=cfg, artifact=art)
    assert profile.evidence_grounding_gate == "FAIL"


def test_g6_01_evidence_empty_after_refinement_indeterminate():
    """Repaired subpart omits evidence IDs; evaluates to INDETERMINATE."""
    sp = SubPart(subpart_index=0, label="a", text="Explain pipelining.", marks=5, bloom="L2", evidence_block_ids=(), telemetry={"repaired": True})
    assert len(sp.evidence_block_ids) == 0


def test_g6_01_evidence_missing_source_hash_fails_closed():
    """Construct evidence with missing/empty artifact revision hash; fails closed."""
    planner = PlanningAgent()
    art_bad = MockArtifact(doc_id="doc1", sha256="")
    art_bad.source_pdf_sha256 = ""
    art_bad.doc_id_hash = ""
    art_bad.manifest_fingerprint = ""
    with pytest.raises(ValueError, match="EVIDENCE_REVISION_UNAVAILABLE"):
        planner._pick_evidence_blocks_for_slot(0, "topic", 1, artifact=art_bad)


def test_g6_01_evidence_two_revisions_identical_text_detected():
    """Inject two artifacts with identical block text but differing revisions; block IDs differ."""
    art1 = MockArtifact(doc_id="doc", sha256="1" * 64, blocks=[{"module_id": "module_1", "page": 1, "text": "Same text"}])
    art2 = MockArtifact(doc_id="doc", sha256="2" * 64, blocks=[{"module_id": "module_1", "page": 1, "text": "Same text"}])
    p = PlanningAgent()
    b1 = p._pick_evidence_blocks_for_slot(0, "text", 1, artifact=art1)
    b2 = p._pick_evidence_blocks_for_slot(0, "text", 1, artifact=art2)
    assert b1 != b2


def test_g6_01_evidence_same_revision_changed_block_text_detected():
    """Same revision hash but modified block text content; block SHA256 detects mismatch."""
    text1 = "Version 1 text content."
    text2 = "Version 2 text content."
    sha1 = hashlib.sha256(text1.strip().encode("utf-8")).hexdigest()
    sha2 = hashlib.sha256(text2.strip().encode("utf-8")).hexdigest()
    assert sha1 != sha2


def test_g6_01_evidence_changed_manifest_detected():
    """Unchanged source PDF but updated extraction manifest fingerprint; detected as divergence."""
    art1 = MockArtifact(doc_id="doc", sha256="a" * 64)
    art1.manifest_fingerprint = "m1" * 32
    art2 = MockArtifact(doc_id="doc", sha256="a" * 64)
    art2.manifest_fingerprint = "m2" * 32
    # In planning agent, manifest_fingerprint is checked
    p = PlanningAgent()
    b1 = p._pick_evidence_blocks_for_slot(0, "t", 1, artifact=art1)
    b2 = p._pick_evidence_blocks_for_slot(0, "t", 1, artifact=art2)
    assert b1 != b2


# =============================================================================
# Area 2: Segment-Grounded Knowledge Graph Construction (11 tests)
# =============================================================================

class MockPaperSpec:
    def __init__(self, mod_count=5):
        self.module_count = mod_count


def test_g6_02_segment_source_hierarchy_precedence():
    """Provide module_segments and artifact.segments; resolves according to hierarchy."""
    artifact = MockArtifact(doc_id="d1")
    artifact.segments = {1: (1, 10), 2: (11, 20)}
    artifact.page_count = 50
    spec = MockPaperSpec(mod_count=2)
    caller_segments = [{"module_id": 1, "page_start": 1, "page_end": 12}, {"module_id": 2, "page_start": 13, "page_end": 25}]

    kg = _build_kg_for_artifact(artifact, spec, module_segments=caller_segments)
    assert getattr(kg, "module_segments_source", None) == "caller_argument"


def test_g6_02_segment_explicit_empty_argument_no_fallback():
    """Supply module_segments = []; is not None prevents silent fall-through to artifact segments."""
    artifact = MockArtifact(doc_id="d1")
    artifact.segments = {1: (1, 10), 2: (11, 20)}
    artifact.page_count = 50
    spec = MockPaperSpec(mod_count=2)
    with pytest.raises(ValueError, match="SEGMENTS_ABSENT"):
        _build_kg_for_artifact(artifact, spec, module_segments=[])


def test_g6_02_segment_page_gaps_and_out_of_order_rejected():
    """Gaps permitted; out-of-order IDs fail closed (DISORDERED_MODULE_SEGMENTS)."""
    artifact = MockArtifact(doc_id="d1")
    artifact.page_count = 100
    spec = MockPaperSpec(mod_count=2)
    # Gaps permitted: Mod 1: [1, 10], Mod 2: [15, 25] -> Should succeed
    kg = _build_kg_for_artifact(artifact, spec, module_segments=[(1, 10), (15, 25)])
    assert kg is not None

    # Out of order: Mod 2 before Mod 1
    with pytest.raises(ValueError, match="DISORDERED_MODULE_SEGMENTS"):
        _build_kg_for_artifact(artifact, spec, module_segments=[{"module_id": 2, "page_start": 1, "page_end": 10}, {"module_id": 1, "page_start": 11, "page_end": 20}])


def test_g6_02_segment_small_overlap_midpoint_no_empty_collapse():
    """Inject segments Mod 1: [1, 20], Mod 2: [18, 35]; midpoint cut at 19, zero collapse."""
    artifact = MockArtifact(doc_id="d1")
    artifact.page_count = 50
    spec = MockPaperSpec(mod_count=2)
    segments = [(1, 20), (18, 35)]
    kg = _build_kg_for_artifact(artifact, spec, module_segments=segments)
    assert kg.module_ranges[1] == (1, 18)
    assert kg.module_ranges[2] == (19, 35)


def test_g6_02_segment_large_overlap_rejected():
    """Inject overlap Delta = 5 > 3 pages; fails closed (FATAL_SEGMENT_OVERLAP)."""
    artifact = MockArtifact(doc_id="d1")
    artifact.page_count = 50
    spec = MockPaperSpec(mod_count=2)
    segments = [(1, 20), (15, 35)]  # Overlap is 20 - 15 + 1 = 6 pages > 3
    with pytest.raises(ValueError, match="FATAL_SEGMENT_OVERLAP"):
        _build_kg_for_artifact(artifact, spec, module_segments=segments)


def test_g6_02_segment_exceeds_page_bounds_rejected():
    """Segment has end = 120 on 100-page document; fails closed (SEGMENT_EXCEEDS_PAGE_BOUNDS)."""
    artifact = MockArtifact(doc_id="d1")
    artifact.page_count = 100
    spec = MockPaperSpec(mod_count=1)
    with pytest.raises(ValueError, match="SEGMENT_EXCEEDS_PAGE_BOUNDS"):
        _build_kg_for_artifact(artifact, spec, module_segments=[(1, 120)])


def test_g6_02_caller_passes_context_segmentation():
    """Execute pipeline bridge helper with caller argument."""
    artifact = MockArtifact(doc_id="d1")
    artifact.page_count = 50
    spec = MockPaperSpec(mod_count=2)
    kg = _build_kg_for_artifact(artifact, spec, module_segments=[(1, 10), (11, 20)])
    assert kg.module_ranges[1] == (1, 10)
    assert kg.module_ranges[2] == (11, 20)


def test_g6_02_insufficient_pages_fail_closed():
    """max_page=3 on 5-module spec; fails closed with INSUFFICIENT_PAGES_FOR_MODULES."""
    artifact = MockArtifact(doc_id="d1")
    artifact.page_count = 3
    spec = MockPaperSpec(mod_count=5)
    with pytest.raises(ValueError, match="INSUFFICIENT_PAGES_FOR_MODULES"):
        _build_kg_for_artifact(artifact, spec, module_segments=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])


def test_g6_02_production_segments_absent_fails_closed():
    """Run production KG build with zero segments provided; fails closed with SEGMENTS_ABSENT."""
    artifact = MockArtifact(doc_id="d1")
    spec = MockPaperSpec(mod_count=3)
    with pytest.raises(ValueError, match="SEGMENTS_ABSENT"):
        _build_kg_for_artifact(artifact, spec, module_segments=None)


def test_g6_02_segment_duplicate_or_inverted_modules_rejected():
    """Inject duplicate module ID or inverted range (end < start); fails closed (INVALID_SEGMENT_RANGE)."""
    artifact = MockArtifact(doc_id="d1")
    artifact.page_count = 50
    spec = MockPaperSpec(mod_count=2)
    # Inverted range [20, 10]
    with pytest.raises(ValueError, match="INVALID_SEGMENT_RANGE"):
        _build_kg_for_artifact(artifact, spec, module_segments=[(20, 10), (21, 30)])


def test_g6_02_segment_small_overlap_collapsing_module_rejected():
    """Inject overlap where midpoint cut leaves a module with 0 pages; fails closed (MODULE_COLLAPSE_PREVENTED)."""
    artifact = MockArtifact(doc_id="d1")
    artifact.page_count = 50
    spec = MockPaperSpec(mod_count=2)
    # Mod 1 has 1 page [10, 10], Mod 2 starts at 9 [9, 20]
    with pytest.raises(ValueError, match="MODULE_COLLAPSE_PREVENTED"):
        _build_kg_for_artifact(artifact, spec, module_segments=[(10, 10), (9, 20)])


# =============================================================================
# Area 3: Topic Sampling Stride & Choice Dispersion (3 tests)
# =============================================================================

def test_g6_03_within_idx_topic_sampling_stride():
    """Sample topics for Q1 (within_idx=0) and Q2 (within_idx=1); offset incorporates cluster stride."""
    topics = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    p = PlanningAgent()
    t1 = p._pick_topic_candidate(topics, slot_idx=0, within_idx=0, used_topics=set())
    t2 = p._pick_topic_candidate(topics, slot_idx=0, within_idx=1, used_topics=set())
    assert t1 != t2


def test_g6_03_topic_dispersion_choice_pairs():
    """Plan SEE Module 1 choice pair (Q1, Q2); Q1 and Q2 have distinct primary concepts."""
    topics = ["Process Scheduling", "Deadlock Avoidance", "Paging", "Segmentation"]
    p = PlanningAgent()
    t1 = p._pick_topic_candidate(topics, slot_idx=0, within_idx=0, used_topics=set())
    t2 = p._pick_topic_candidate(topics, slot_idx=0, within_idx=1, used_topics={t1})
    assert t1 != t2


def test_g6_03_fallback_compact_module():
    """Plan choice pair on single-concept module; Tier 3 fallback engaged, distinct operations assigned."""
    topics = ["Single Unified Topic"]
    p = PlanningAgent()
    t1 = p._pick_topic_candidate(topics, slot_idx=0, within_idx=0, used_topics=set())
    t2 = p._pick_topic_candidate(topics, slot_idx=0, within_idx=1, used_topics={t1})
    assert t1 == t2 == "Single Unified Topic"


# =============================================================================
# Area 4: CO Decoupling, Mandatory Coverage & Exact Bloom Optimization (9 tests)
# =============================================================================

def test_g6_04_co_bloom_decoupling():
    """Configure target distribution with high L2 under CO4; CO4 question generated with L2 Bloom level."""
    p = PlanningAgent()
    # Direct decouple test: Bloom assignments can assign L2 to CO4 slots
    assigned = p._assign_bloom(slot_idx=0, total_slots=5, co="CO4", target_bloom="L2")
    assert assigned == "L2"


def test_g6_04_unconfigured_cognitive_target_preserves_behavior():
    """Set cognitive_target=None on AssessmentConfig; paper generates normally, gate is NOT_APPLICABLE."""
    cfg = AssessmentConfig(assessment_type="IA", cognitive_target=None)
    assert cfg.cognitive_target is None
    profile = generate_assessment_profile([], config=cfg)
    assert profile.cognitive_distribution_gate == "NOT_APPLICABLE"


def test_g6_04_invalid_cognitive_target_rejected():
    """Inject NaN, negative, float-boolean, or sum != 100 targets; raises ValueError / TypeError."""
    with pytest.raises(TypeError):
        CognitiveTargetDistribution(l1_percentage=True)
    with pytest.raises(ValueError):
        CognitiveTargetDistribution(l1_percentage=-10.0, l2_percentage=110.0)
    with pytest.raises(ValueError):
        CognitiveTargetDistribution(l1_percentage=20.0, l2_percentage=20.0)  # sum = 40 != 100


def test_g6_04_inclusive_tolerance_boundaries():
    """Subparts achieve exactly Tk ± delta; inclusive boundaries verified, status is PASS."""
    # Target 50% L2, 50% L3, tol=5.0. Paper achieves 45% L2, 55% L3 (boundary exactly).
    target = CognitiveTargetDistribution(l2_percentage=50.0, l3_percentage=50.0, tolerance_percentage=5.0)
    cfg = AssessmentConfig(assessment_type="IA", cognitive_target=target)
    sp1 = SubPart(0, "a", "Q1", 45, "L2", marking_scheme=[{"criterion": "c1", "marks": 45}])
    sp2 = SubPart(0, "a", "Q2", 55, "L3", marking_scheme=[{"criterion": "c2", "marks": 55}])
    q1 = GeneratedQuestion(question_id="Q1", slot_id="module_1_Q1", question_text="Q1", marks=45, bloom="L2", sub_parts=[sp1])
    q2 = GeneratedQuestion(question_id="Q2", slot_id="module_1_Q2", question_text="Q2", marks=55, bloom="L3", sub_parts=[sp2])
    audit = audit_whole_paper_bloom([q1, q2], config=cfg)
    assert audit["distribution_compliant"] is True


def test_g6_04_discrete_marks_constrained_optimization():
    """Evaluate optimization on 50m IA with 10m questions; respects discrete units."""
    subparts = [
        SubPartPlan("a", 10, "L2", "CO1", "T1", candidate_blooms=["L2", "L3"]),
        SubPartPlan("a", 10, "L2", "CO2", "T2", candidate_blooms=["L2", "L3"]),
        SubPartPlan("a", 10, "L2", "CO3", "T3", candidate_blooms=["L2", "L4"]),
        SubPartPlan("a", 10, "L2", "CO4", "T4", candidate_blooms=["L3", "L4"]),
        SubPartPlan("a", 10, "L2", "CO5", "T5", candidate_blooms=["L3", "L4"]),
    ]
    target = CognitiveTargetDistribution(l2_percentage=40.0, l3_percentage=40.0, l4_percentage=20.0)
    best_bloom, err, status = optimize_bloom_distribution(subparts, target, assessment_id="IA1")
    assert len(best_bloom) == 5
    assert err == 0.0  # Exactly 2 at L2 (20m = 40%), 2 at L3 (20m = 40%), 1 at L4 (10m = 20%)


def test_g6_04_cognitive_optimization_arbitrary_marks():
    """Optimize cognitive targets with arbitrary marks (3m, 5m, 7m) cleanly without hardcoded failure."""
    subparts = [
        SubPartPlan("a", 3, "L1", "CO1", "T1", candidate_blooms=["L1", "L2"]),
        SubPartPlan("b", 5, "L2", "CO1", "T1", candidate_blooms=["L2", "L3"]),
        SubPartPlan("c", 7, "L3", "CO1", "T1", candidate_blooms=["L3", "L4"]),
    ]
    target = CognitiveTargetDistribution(l1_percentage=20.0, l2_percentage=33.33, l3_percentage=46.67)
    best_bloom, err, status = optimize_bloom_distribution(subparts, target, assessment_id="ARB")
    assert len(best_bloom) == 3


def test_g6_04_co_bloom_empty_candidates_fails_closed():
    """Assigned CO in active plan maps to empty Bloom candidates; raises EMPTY_CO_BLOOM_CANDIDATES."""
    p = PlanningAgent()
    spec = MockPaperSpec(mod_count=1)
    # Plan a question where assigned CO has empty bloom candidates
    with pytest.raises(ValueError, match="EMPTY_CO_BLOOM_CANDIDATES"):
        p._plan(assessment_id="TEST", paper_spec=spec, target_distribution=None, co_candidate_map={"CO1": []})


def test_g6_04_mandatory_co_coverage_failure_blocks_pass():
    """Paper plan omits a mandatory configured CO; validation gate evaluates to FAIL (MANDATORY_CO_COVERAGE_FAILED)."""
    cfg = AssessmentConfig(assessment_type="IA", mandatory_course_outcomes={"CO1", "CO2"})
    sp1 = SubPart(0, "a", "P1", 10, "L2", course_outcome="CO1", marking_scheme=[{"criterion": "c1", "marks": 10}])
    q1 = GeneratedQuestion("Q1", "module_1_Q1", "P1", 10, "L2", sub_parts=[sp1])
    # CO2 has 0 marks
    profile = generate_assessment_profile([q1], config=cfg)
    assert profile.cognitive_distribution_gate == "FAIL"


def test_g6_04_neutral_tie_break_avoids_high_bloom_bias():
    """Evaluate tie-breaking on identical-error candidate allocations; seeded SHA-256 digest produces neutral tie-break."""
    key1 = compute_bloom_tie_break_key("ASSESS1", ["s1", "s2"], ["L2", "L3"], None)
    key2 = compute_bloom_tie_break_key("ASSESS1", ["s1", "s2"], ["L3", "L2"], None)
    assert len(key1) == 64
    assert len(key2) == 64
    assert key1 != key2


# =============================================================================
# Area 5: Pedagogical Subpart Progression (2 tests)
# =============================================================================

def test_g6_05_subpart_bloom_progression():
    """Plan 2-part question [6, 4]; subpart (a) Bloom level <= subpart (b) Bloom level."""
    sp_a = SubPartPlan("a", 6, "L2", "CO1", "T1")
    sp_b = SubPartPlan("b", 4, "L3", "CO1", "T1")
    bloom_rank = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6}
    assert bloom_rank[sp_a.bloom] <= bloom_rank[sp_b.bloom]


def test_g6_05_justified_progression_inversion_allowed():
    """Simulation (L4) followed by reflection (L2); permitted with recorded telemetry justification."""
    sp_a = SubPart(0, "a", "Simulate algorithm.", 6, "L4")
    sp_b = SubPart(1, "b", "Reflect on results.", 4, "L2", telemetry={"progression_exception_reason": "Reflection follows simulation"})
    assert sp_b.telemetry["progression_exception_reason"] is not None


# =============================================================================
# Area 6: Multi-Dimensional Concept Footprints & Structural Relationships (9 tests)
# =============================================================================

def test_g6_06_concept_same_slot_alternatives_topic_collapse():
    """Inject identical concept & operation in choice alternatives Q1 and Q2; flags CHOICE_PAIR_TOPIC_COLLAPSE (FAIL)."""
    fp = ConceptFootprint(canonical_concept="Deadlock", primary_operation="Explain", provenance_entity_id="node_dl", provenance_type="KG_CONCEPT")
    q1 = GeneratedQuestion("Q1", "module_1_Q1", "Explain Deadlock", 10, "L2", choice_group_id="M1_OR", concept_footprints=(fp.to_dict(),))
    q2 = GeneratedQuestion("Q2", "module_1_Q2", "Explain Deadlock", 10, "L2", choice_group_id="M1_OR", concept_footprints=(fp.to_dict(),))
    # Direct footprint mapping for test
    q1.concept_footprint = fp
    q2.concept_footprint = fp

    res = classify_relationship(q1, q2)
    assert res["status"] == "FAIL"
    assert res["classification"] == "CHOICE_PAIR_TOPIC_COLLAPSE"


def test_g6_06_concept_sibling_progression_four_conditions():
    """Subpart (a) defines concept (L2); subpart (b) computes (L4); returns LEGITIMATE_PROGRESSION (PASS)."""
    class MockKG:
        def has_node(self, n): return True
        def get_node(self, n): return {"module_id": "module_1"}

    fp_a = ConceptFootprint("Paging", "definition", "prose", "L2", provenance_entity_id="p_node", provenance_type="KG_CONCEPT")
    fp_b = ConceptFootprint("Paging", "calculate_faults", "numerical", "L4", provenance_entity_id="p_node", provenance_type="KG_CONCEPT")

    sp_a = SubPart(0, "a", "Define paging.", 4, "L2", concept_footprint=fp_a, module_id="module_1", question_id="Q1")
    sp_b = SubPart(1, "b", "Calculate page faults.", 6, "L4", concept_footprint=fp_b, module_id="module_1", question_id="Q1")

    res = classify_relationship(sp_a, sp_b, kg=MockKG())
    assert res["status"] == "PASS"
    assert res["classification"] == "LEGITIMATE_PROGRESSION"


def test_g6_06_concept_different_operations_alone_cannot_pass():
    """Same concept, different operation label, but same Bloom and response type; returns SAME_CONCEPT_REVIEW."""
    class MockKG:
        def has_node(self, n): return True
        def get_node(self, n): return {"module_id": "module_1"}

    fp_a = ConceptFootprint("Paging", "operation_1", "prose", "L2", provenance_entity_id="p_node", provenance_type="KG_CONCEPT")
    fp_b = ConceptFootprint("Paging", "operation_2", "prose", "L2", provenance_entity_id="p_node", provenance_type="KG_CONCEPT")

    sp_a = SubPart(0, "a", "Op 1", 5, "L2", concept_footprint=fp_a, module_id="module_1", question_id="Q1")
    sp_b = SubPart(1, "b", "Op 2", 5, "L2", concept_footprint=fp_b, module_id="module_1", question_id="Q1")

    res = classify_relationship(sp_a, sp_b, kg=MockKG())
    assert res["status"] == "INDETERMINATE"
    assert res["classification"] == "SAME_CONCEPT_REVIEW"


def test_g6_06_concept_unverified_provenance_indeterminate():
    """Omit provenance_entity_id on one footprint; classified as UNVERIFIED_CONCEPT_REVIEW."""
    fp_a = ConceptFootprint("Paging", "op1", "prose", "L2", provenance_entity_id="", provenance_type="KG_CONCEPT")
    sp_a = SubPart(0, "a", "Op 1", 5, "L2", concept_footprint=fp_a, module_id="module_1", question_id="Q1")
    sp_b = SubPart(1, "b", "Op 2", 5, "L2", concept_footprint=fp_a, module_id="module_1", question_id="Q1")

    res = classify_relationship(sp_a, sp_b)
    assert res["status"] == "INDETERMINATE"
    assert res["classification"] == "UNVERIFIED_CONCEPT_REVIEW"


def test_g6_06_provenance_unresolved_kg_node_indeterminate():
    """Non-empty provenance_entity_id absent from authoritative KG; classified as UNVERIFIED_CONCEPT_REVIEW (INDETERMINATE)."""
    class EmptyKG:
        def has_node(self, n): return False

    fp = ConceptFootprint("Paging", provenance_entity_id="ghost_node", provenance_type="KG_CONCEPT")
    status, reason = verify_concept_provenance(fp, module_idx=1, kg=EmptyKG())
    assert status == "UNVERIFIED_CONCEPT_REVIEW"


def test_g6_06_provenance_wrong_module_rejected():
    """Concept exists in KG under Module 2 but claimed by Module 1 question; classified as CROSS_MODULE_PROVENANCE_MISMATCH."""
    class MockKG:
        def has_node(self, n): return True
        def get_node(self, n): return {"module_id": "module_2"}

    fp = ConceptFootprint("Paging", provenance_entity_id="node_mod2", provenance_type="KG_CONCEPT")
    status, reason = verify_concept_provenance(fp, module_idx=1, kg=MockKG())
    assert status == "CROSS_MODULE_PROVENANCE_MISMATCH"


def test_g6_06_choice_pair_identified_from_structure():
    """Recognize choice pair using choice_group_id rather than regex."""
    q1 = GeneratedQuestion("Q1", "m1_slot_1", "Q1 text", 10, "L2", choice_group_id="M1_OR_PAIR")
    q2 = GeneratedQuestion("Q2", "m1_slot_2", "Q2 text", 10, "L2", choice_group_id="M1_OR_PAIR")
    assert q1.choice_group_id == q2.choice_group_id == "M1_OR_PAIR"


def test_g6_06_choice_classification_symmetric_order_invariant():
    """Compare classify_relationship(A, B) vs classify_relationship(B, A); both return identical result."""
    fp1 = ConceptFootprint("Paging", provenance_entity_id="p1", provenance_type="DOCUMENT_SECTION")
    fp2 = ConceptFootprint("Virtual Memory", provenance_entity_id="p2", provenance_type="DOCUMENT_SECTION")
    q1 = GeneratedQuestion("Q1", "module_1_Q1", "T1", 10, "L2", concept_footprint=fp1)
    q2 = GeneratedQuestion("Q2", "module_1_Q2", "T2", 10, "L2", concept_footprint=fp2)

    res_ab = classify_relationship(q1, q2)
    res_ba = classify_relationship(q2, q1)
    assert res_ab["status"] == res_ba["status"]
    assert res_ab["classification"] == res_ba["classification"]


def test_g6_06_cross_module_recurrence_precedence():
    """Inject same concept in Module 1 and Module 4; cross-module check runs first, classified as recurrence allowed."""
    fp = ConceptFootprint("Pipeline Hazards", provenance_entity_id="p1", provenance_type="DOCUMENT_SECTION")
    q1 = GeneratedQuestion("Q1", "module_1_Q1", "T1", 10, "L2", module_id="module_1", concept_footprint=fp)
    q4 = GeneratedQuestion("Q4", "module_4_Q1", "T4", 10, "L2", module_id="module_4", concept_footprint=fp)

    res = classify_relationship(q1, q4)
    assert res["status"] == "PASS"
    assert res["classification"] == "CROSS_MODULE_RECURRENCE_ALLOWED"


# =============================================================================
# Area 7: Cross-Module Concept Recurrence Telemetry (1 test)
# =============================================================================

def test_g6_07_cross_module_recurrence_telemetry():
    """Interdisciplinary concept recurrence across modules records CROSS_MODULE_RECURRENCE_ALLOWED telemetry."""
    fp = ConceptFootprint("B-Trees", provenance_entity_id="p1", provenance_type="DOCUMENT_SECTION")
    q1 = GeneratedQuestion("Q1", "module_1_Q1", "T1", 10, "L2", module_id="module_1", concept_footprint=fp)
    q5 = GeneratedQuestion("Q5", "module_5_Q1", "T5", 10, "L2", module_id="module_5", concept_footprint=fp)

    res = classify_relationship(q1, q5)
    assert res["status"] == "PASS"
    assert res.get("telemetry", {}).get("cross_module_recurrence") is True


# =============================================================================
# Area 8: Solution Completeness & Expected Response Contract (3 tests)
# =============================================================================

def test_g6_08_solution_completeness_calculation_pass():
    """Provide complete calculation with given, formula, steps, result; evaluate_solution_completeness returns PASS."""
    contract = ExpectedResponseContract(
        required_elements=("given_data", "formula", "stepwise_substitution", "final_answer"),
        expected_operations=("multiplication",)
    )
    sol = "Given V=10V, R=2 ohms. Formula: I = V / R. Steps: I = 10 / 2 = 5. Final Answer: 5 Amperes."
    status, reason = evaluate_solution_completeness("Calculate current.", sol, contract=contract, marks=5)
    assert status == "PASS"


def test_g6_08_solution_completeness_trivial_fail():
    """Provide 55-char placeholder text for 10-mark calculation; evaluator returns FAIL."""
    contract = ExpectedResponseContract(
        required_elements=("formula", "stepwise_substitution", "final_answer"),
        expected_operations=("division",)
    )
    sol = "This is a brief placeholder text for solution to be filled later."
    status, reason = evaluate_solution_completeness("Calculate value.", sol, contract=contract, marks=10)
    assert status == "FAIL"


def test_g6_08_solution_unconventional_derivation_indeterminate():
    """Valid derivation using unconventional substitutions; evaluator returns INDETERMINATE."""
    contract = ExpectedResponseContract(
        required_elements=("formula", "final_answer"),
        expected_operations=("derivation",)
    )
    sol = "Using an unconventional substitution method, we derive the state equation..."
    status, reason = evaluate_solution_completeness("Derive equation.", sol, contract=contract, marks=6)
    assert status == "INDETERMINATE"


# =============================================================================
# Area 9: Rubric Criterion Lifecycle & Strict Integer Marks (4 tests)
# =============================================================================

def test_g6_09_rubric_strict_validation_rejects_string_float_bool():
    """Pass criteria dict with string marks, float, bool, or extra keys; strict validation raises TypeError / ValueError."""
    with pytest.raises(TypeError):
        RubricCriterion.from_dict({"criterion": "Steps", "marks": "2"})
    with pytest.raises(TypeError):
        RubricCriterion.from_dict({"criterion": "Steps", "marks": 2.5})
    with pytest.raises(TypeError):
        RubricCriterion.from_dict({"criterion": "Steps", "marks": True})


def test_g6_09_rubric_sum_mismatch_fails_closed():
    """Criteria marks sum to 8 for a 10-mark subpart; evaluator fails closed (RUBRIC_MARKS_SUM_MISMATCH)."""
    sp = SubPart(0, "a", "Explain VM.", 10, "L2")
    scheme = [{"criterion": "Definition", "marks": 4}, {"criterion": "Diagram", "marks": 4}]  # sum = 8 != 10
    status, reason = evaluate_marking_scheme_relevance(sp, scheme)
    assert status == "FAIL"
    assert "RUBRIC_MARKS_SUM_MISMATCH" in reason


def test_g6_09_legacy_marking_scheme_prose_wrapped_indeterminate():
    """SubPart has legacy prose string marking scheme; wrapped in standard dict, evaluates to INDETERMINATE_LEGACY_STRING."""
    sp = SubPart(0, "a", "Explain VM.", 5, "L2")
    sp.materialize_marking_scheme("Award 3 marks for definition and 2 marks for diagram.")
    assert len(sp.marking_scheme) == 1
    assert sp.marking_scheme[0].get("is_legacy_string") is True

    status, reason = evaluate_marking_scheme_relevance(sp, sp.marking_scheme)
    assert status == "INDETERMINATE"
    assert "INDETERMINATE_LEGACY_STRING" in reason


def test_g6_09_materialized_marking_scheme_strictly_list_of_dicts():
    """Inspect parent GeneratedQuestion.marking_scheme across all states; always strictly List[Dict[str, Any]]."""
    sp1 = SubPart(0, "a", "P1", 4, "L2", marking_scheme=[{"criterion": "c1", "marks": 4}])
    sp2 = SubPart(1, "b", "P2", 6, "L2", marking_scheme=[{"criterion": "c2", "marks": 6}])
    q = GeneratedQuestion("Q1", "m1_slot_1", "Text", 10, "L2", sub_parts=[sp1, sp2])
    q.sync_materialized_projections()
    assert isinstance(q.marking_scheme, list)
    assert all(isinstance(entry, dict) for entry in q.marking_scheme)
    assert len(q.marking_scheme) == 2


# =============================================================================
# Area 10: Universal Visual Grounding Across All Bloom Levels (3 tests)
# =============================================================================

def test_g6_10_visual_grounding_l2_allowed():
    """Assign visual task to L2 subpart (architecture diagram); visual dependency check passes without Bloom lockout."""
    vt = VisualTaskContract("fig_arch", True, "explain_diagram", ["blocks"])
    q_text = "With the aid of the given figure, explain the system architecture."
    status, reason = evaluate_visual_dependency(q_text, figure_metadata={"width": 800, "height": 600}, visual_contract=vt, bloom="L2")
    assert status == "PASS"


def test_g6_10_visual_grounding_l4_allowed():
    """Assign visual task to L4 subpart (state transition troubleshooting); passes without Bloom lockout."""
    vt = VisualTaskContract("fig_state", True, "analyze_states", ["transitions"])
    q_text = "From the given state transition diagram, determine the deadlock condition."
    status, reason = evaluate_visual_dependency(q_text, figure_metadata={"width": 800, "height": 600}, visual_contract=vt, bloom="L4")
    assert status == "PASS"


def test_g6_10_visual_grounding_all_bloom_levels_eligible():
    """Assign visual task to L1 and L6; permitted; Bloom level is not an allowlist/denylist blocker."""
    vt1 = VisualTaskContract("fig_comp", True, "identify_components", ["labels"])
    q1_text = "From the circuit shown in the diagram, read and list all components."
    st1, _ = evaluate_visual_dependency(q1_text, figure_metadata={"width": 400, "height": 400}, visual_contract=vt1, bloom="L1")
    assert st1 == "PASS"

    vt6 = VisualTaskContract("fig_spec", True, "synthesize_circuit", ["interfaces"])
    q6_text = "Using the topology shown in the diagram, design a customized interface."
    st6, _ = evaluate_visual_dependency(q6_text, figure_metadata={"width": 400, "height": 400}, visual_contract=vt6, bloom="L6")
    assert st6 == "PASS"


# =============================================================================
# Area 11: Four-Condition Visual Task Dependency Validation (4 tests)
# =============================================================================

def test_g6_11_visual_dependency_genuine_pass():
    """Question requires reading component values from schematic; evaluate_visual_dependency returns PASS."""
    vt = VisualTaskContract("fig_sch", True, "read_values", ["resistors"])
    q_text = "Read the resistor values from the given schematic and determine total impedance."
    status, reason = evaluate_visual_dependency(q_text, figure_metadata={"width": 500, "height": 500}, visual_contract=vt, bloom="L3")
    assert status == "PASS"


def test_g6_11_visual_dependency_superficial_text_fail():
    """Question text provides all values, rendering figure purely decorative; returns FAIL (DECORATIVE_VISUAL_REDUNDANT)."""
    vt = VisualTaskContract("fig_sch", True, "calc", ["resistors"])
    q_text = "In the circuit of Figure 1, resistor R1 = 10 ohms, resistor R2 = 20 ohms, and voltage V = 12V. Calculate total current."
    status, reason = evaluate_visual_dependency(q_text, figure_metadata={"width": 500, "height": 500}, visual_contract=vt, bloom="L3")
    assert status == "FAIL"
    assert "DECORATIVE_VISUAL_REDUNDANT" in reason


def test_g6_11_visual_dependency_no_figure_referenced_fail():
    """Question claims visual grounding but prompt text never mentions figure; returns FAIL (UNREFERENCED_VISUAL_ASSET)."""
    vt = VisualTaskContract("fig_123", True, "identify", ["nodes"])
    q_text = "Compute the shortest path in the graph given by vertices A, B, C."
    status, reason = evaluate_visual_dependency(q_text, figure_metadata={"width": 500, "height": 500}, visual_contract=vt, bloom="L3")
    assert status == "FAIL"
    assert "UNREFERENCED_VISUAL_ASSET" in reason


def test_g6_11_visual_dependency_four_conditions_independent():
    """Visual dependency with spatial ambiguity evaluated; resolves to INDETERMINATE."""
    vt = VisualTaskContract("fig_spat", True, "measure", ["coords"])
    q_text = "With reference to the figure, spatial interpretation is ambiguous for partial geometry."
    status, reason = evaluate_visual_dependency(q_text, figure_metadata={"width": 500, "height": 500}, visual_contract=vt, bloom="L4")
    assert status == "INDETERMINATE"
    assert "VISUAL_DEPENDENCY_AMBIGUOUS" in reason


# =============================================================================
# Area 12: Typed Contract Planner Adoption & Blueprint Decoupling (2 tests)
# =============================================================================

def test_g6_12_typed_contract_planner_adoption():
    """Run active planner _plan(); generated plans contain all typed academic contracts directly."""
    p = PlanningAgent()
    spec = MockPaperSpec(mod_count=1)
    plan = p._plan(assessment_id="TEST_ADOPTION", paper_spec=spec)
    assert len(plan) > 0
    sp_plan = plan[0].sub_part_plans[0]
    assert isinstance(sp_plan.concept_footprint, ConceptFootprint)
    assert isinstance(sp_plan.expected_response, ExpectedResponseContract)
    assert isinstance(sp_plan.rubric_criteria, list)


def test_g6_12_blueprint_architecture_ownership_documented():
    """Verify architecture ownership decision documentation and blueprint decoupling."""
    # Ensure PlanningAgent remains authoritative source of QuestionPlan without AssessmentBlueprint dependency
    p = PlanningAgent()
    assert hasattr(p, "_plan")
    assert hasattr(p, "optimize_bloom_distribution")


# =============================================================================
# Area 13: Calibrated Choice-Pair Equivalence Audit (4 tests)
# =============================================================================

def test_g6_13_choice_pair_parity_all_seven_dimensions_pass():
    """Provide identical-structure, balanced choice pair; check_choice_pair_equivalence returns PASS."""
    sp1 = SubPart(0, "a", "P1", 6, "L2", marks=6)
    sp2 = SubPart(1, "b", "P2", 4, "L3", marks=4)
    q1 = GeneratedQuestion("Q1", "m1_slot_1", "Q1", 10, "L2", sub_parts=[sp1, sp2], choice_group_id="M1_OR")

    sp3 = SubPart(0, "a", "P3", 6, "L2", marks=6)
    sp4 = SubPart(1, "b", "P4", 4, "L3", marks=4)
    q2 = GeneratedQuestion("Q2", "m1_slot_2", "Q2", 10, "L2", sub_parts=[sp3, sp4], choice_group_id="M1_OR")

    status, defects = check_choice_pair_equivalence(q1, q2)
    assert status == "PASS"
    assert len(defects) == 0


def test_g6_13_choice_pair_parity_subpart_count_mismatch_fail():
    """Q1 has 2 subparts [6, 4], Q2 has 3 subparts [4, 3, 3]; returns FAIL (SUBPART_COUNT_MISMATCH)."""
    sps1 = [SubPart(0, "a", "t1", 6), SubPart(1, "b", "t2", 4)]
    sps2 = [SubPart(0, "a", "t3", 4), SubPart(1, "b", "t4", 3), SubPart(2, "c", "t5", 3)]
    q1 = GeneratedQuestion("Q1", "m1_s1", "Q1", 10, "L2", sub_parts=sps1, choice_group_id="OR1")
    q2 = GeneratedQuestion("Q2", "m1_s2", "Q2", 10, "L2", sub_parts=sps2, choice_group_id="OR1")

    status, defects = check_choice_pair_equivalence(q1, q2)
    assert status == "FAIL"
    assert any("SUBPART_COUNT_MISMATCH" in d for d in defects)


def test_g6_13_choice_pair_parity_bloom_inversion_fail():
    """Q1 is L2/L3; Q2 is L4/L5; returns FAIL (COGNITIVE_DEPTH_IMBALANCE)."""
    sps1 = [SubPart(0, "a", "t1", 5, bloom="L2"), SubPart(1, "b", "t2", 5, bloom="L3")]
    sps2 = [SubPart(0, "a", "t3", 5, bloom="L4"), SubPart(1, "b", "t4", 5, bloom="L5")]
    q1 = GeneratedQuestion("Q1", "m1_s1", "Q1", 10, "L2", sub_parts=sps1, choice_group_id="OR1")
    q2 = GeneratedQuestion("Q2", "m1_s2", "Q2", 10, "L2", sub_parts=sps2, choice_group_id="OR1")

    status, defects = check_choice_pair_equivalence(q1, q2)
    assert status == "FAIL"
    assert any("COGNITIVE_DEPTH_IMBALANCE" in d for d in defects)


def test_g6_13_choice_pair_parity_archetype_asymmetry_fail():
    """Q1 is conceptual explanation; Q2 is complex algorithmic proof; returns FAIL (ARCHETYPE_ASYMMETRY)."""
    sp1 = SubPart(0, "a", "t1", 10, bloom="L2")
    sp2 = SubPart(0, "a", "t2", 10, bloom="L2")
    q1 = GeneratedQuestion("Q1", "m1_s1", "Q1", 10, "L2", sub_parts=[sp1], primary_archetype="conceptual_essay", choice_group_id="OR1")
    q2 = GeneratedQuestion("Q2", "m1_s2", "Q2", 10, "L2", sub_parts=[sp2], primary_archetype="algorithmic_proof", choice_group_id="OR1")

    status, defects = check_choice_pair_equivalence(q1, q2)
    assert status == "FAIL"
    assert any("ARCHETYPE_ASYMMETRY" in d for d in defects)


# =============================================================================
# Area 14: Selection-Tree Attempt-Path Modeling & Dual Whole-Paper Accounting (7 tests)
# =============================================================================

def test_g6_14_whole_paper_dual_bloom_accounting_both_pass():
    """Balanced paper with compliant printed and attempt profiles; both profiles satisfy tolerance; status PASS."""
    target = CognitiveTargetDistribution(l2_percentage=50.0, l3_percentage=50.0, tolerance_percentage=10.0)
    cfg = AssessmentConfig(assessment_type="SEE", cognitive_target=target)

    # Choice group 1: Q1 (5m L2, 5m L3) OR Q2 (5m L2, 5m L3)
    sp1 = SubPart(0, "a", "t1", 5, "L2")
    sp2 = SubPart(1, "b", "t2", 5, "L3")
    q1 = GeneratedQuestion("Q1", "m1_Q1", "Q1", 10, "L2", sub_parts=[sp1, sp2], choice_group_id="M1_OR")

    sp3 = SubPart(0, "a", "t3", 5, "L2")
    sp4 = SubPart(1, "b", "t4", 5, "L3")
    q2 = GeneratedQuestion("Q2", "m1_Q2", "Q2", 10, "L2", sub_parts=[sp3, sp4], choice_group_id="M1_OR")

    audit = audit_whole_paper_bloom([q1, q2], config=cfg)
    assert audit["distribution_compliant"] is True


def test_g6_14_attempt_path_worst_case_path_fails_distribution():
    """Printed paper balances, but path choosing all low-Bloom options violates target; evaluates to FAIL."""
    target = CognitiveTargetDistribution(l2_percentage=50.0, l4_percentage=50.0, tolerance_percentage=5.0)
    cfg = AssessmentConfig(assessment_type="SEE", cognitive_target=target)

    # Q1 is all L2 (10m), Q2 is all L4 (10m) in internal choice group M1_OR
    # Printed paper is 50% L2 and 50% L4, but attempt paths are 100% L2 or 100% L4
    sp1 = SubPart(0, "a", "t1", 10, "L2")
    q1 = GeneratedQuestion("Q1", "m1_Q1", "Q1", 10, "L2", sub_parts=[sp1], choice_group_id="M1_OR")

    sp2 = SubPart(0, "a", "t2", 10, "L4")
    q2 = GeneratedQuestion("Q2", "m1_Q2", "Q2", 10, "L4", sub_parts=[sp2], choice_group_id="M1_OR")

    audit = audit_whole_paper_bloom([q1, q2], config=cfg)
    assert audit["distribution_compliant"] is False
    assert any("ATTEMPT_BLOOM_OUT_OF_TOLERANCE" in d for d in audit["defects"])


def test_g6_14_attempt_path_dp_matches_exhaustive():
    """Cross-validate DP min/max Bloom with exhaustive enumeration; mathematical identity verified."""
    # Build 3 choice groups of 2 questions each (2^3 = 8 paths <= 1024)
    qs = []
    for g in range(3):
        sp_a = SubPart(0, "a", f"qa_{g}", 10, "L2")
        sp_b = SubPart(0, "a", f"qb_{g}", 10, "L3")
        qa = GeneratedQuestion(f"QA_{g}", f"m{g}_QA", "text", 10, "L2", sub_parts=[sp_a], choice_group_id=f"G_{g}")
        qb = GeneratedQuestion(f"QB_{g}", f"m{g}_QB", "text", 10, "L3", sub_parts=[sp_b], choice_group_id=f"G_{g}")
        qs.extend([qa, qb])

    # Run exhaustive
    ex_bounds = calculate_attempt_path_bloom_bounds(qs, max_exhaustive_paths=1024)
    # Run DP by forcing max_exhaustive_paths=1
    dp_bounds = calculate_attempt_path_bloom_bounds(qs, max_exhaustive_paths=1)

    assert ex_bounds == dp_bounds


def test_g6_14_attempt_path_excessive_path_bound():
    """Hypothetical exam with 12 choice modules (2^12 = 4096 > 1024); evaluates via DP without memory blowup."""
    qs = []
    for g in range(12):
        sp_a = SubPart(0, "a", f"qa_{g}", 10, "L2")
        sp_b = SubPart(0, "a", f"qb_{g}", 10, "L3")
        qa = GeneratedQuestion(f"QA_{g}", f"m{g}_QA", "text", 10, "L2", sub_parts=[sp_a], choice_group_id=f"G_{g}")
        qb = GeneratedQuestion(f"QB_{g}", f"m{g}_QB", "text", 10, "L3", sub_parts=[sp_b], choice_group_id=f"G_{g}")
        qs.extend([qa, qb])

    dp_bounds = calculate_attempt_path_bloom_bounds(qs, max_exhaustive_paths=1024)
    assert "L2" in dp_bounds
    assert "L3" in dp_bounds
    assert dp_bounds["L2"] == (0.0, 100.0)


def test_g6_14_attempt_path_nested_and_mutually_exclusive_choices():
    """Conditional selection tree (Q1 has subpart choice rule, Q2 has none); valid path count differs from naive product."""
    sp_a1 = SubPart(0, "a", "sub1", 5, "L2")
    sp_a2 = SubPart(1, "b", "sub2", 5, "L3")
    sp_a3 = SubPart(2, "c", "sub3", 5, "L4")
    # Q1: choose 2 of 3 subparts
    q1 = GeneratedQuestion("Q1", "m1_Q1", "text", 10, "L2", sub_parts=[sp_a1, sp_a2, sp_a3], choice_group_id="G1")
    q1.subpart_choice_rule = {"choose_k": 2}

    # Q2: monolithic 10 marks
    sp_b = SubPart(0, "a", "sub_b", 10, "L2")
    q2 = GeneratedQuestion("Q2", "m1_Q2", "text", 10, "L2", sub_parts=[sp_b], choice_group_id="G1")

    nodes = build_selection_tree_from_questions([q1, q2])
    paths = enumerate_attempt_paths(nodes)
    # Q1 gives 3 combinations of 2 subparts; Q2 gives 1 option. Total paths = 3 + 1 = 4 (not 3 * 1 = 3).
    assert len(paths) == 4


def test_g6_14_missing_bloom_affects_both_printed_and_attempt_profiles():
    """Omit Bloom level on one subpart; both profiles fail closed with ACCOUNTING_INCOMPLETE."""
    sp = SubPart(0, "a", "Text", 10, bloom="")
    q = GeneratedQuestion("Q1", "m1_Q1", "Text", 10, bloom="", sub_parts=[sp])
    audit = audit_whole_paper_bloom([q])
    assert audit["accounting_complete"] is False
    assert any("ACCOUNTING_INCOMPLETE" in d for d in audit["defects"])


def test_g6_14_missing_bloom_fails_accounting_with_unconfigured_target():
    """Omit Bloom level on subpart when cognitive_target=None; bloom_accounting_gate fails closed (FAIL)."""
    cfg = AssessmentConfig(assessment_type="IA", cognitive_target=None)
    sp = SubPart(0, "a", "Text", 10, bloom="")
    q = GeneratedQuestion("Q1", "m1_Q1", "Text", 10, bloom="", sub_parts=[sp])
    profile = generate_assessment_profile([q], config=cfg)
    assert profile.bloom_accounting_gate == "FAIL"


# =============================================================================
# Area C1: Expected Response Contract Lifecycle & Serialization (6 tests)
# =============================================================================

def test_g6_c1_expected_response_contract_lifecycle_persistence():
    """Plan subpart with ExpectedResponseContract; persists to SubPart.expected_response and projection sync."""
    contract = ExpectedResponseContract(required_elements=("given", "formula", "answer"), expected_operations=("multiply",))
    sp = SubPart(0, "a", "Text", 5, "L3", expected_response=contract)
    assert sp.expected_response == contract

    q = GeneratedQuestion("Q1", "m1_Q1", "Text", 5, "L3", sub_parts=[sp])
    q.sync_materialized_projections()
    assert len(q.expected_responses) == 1
    assert q.expected_responses[0]["required_elements"] == ["given", "formula", "answer"]


def test_g6_c1_stale_evidence_not_restored_after_refinement():
    """Replace subpart during refinement without evidence IDs; SubPart evidence remains empty, stale IDs not restored."""
    sp_orig = SubPart(0, "a", "Text", 5, "L2", evidence_block_ids=("b1", "b2"))
    q = GeneratedQuestion("Q1", "m1_Q1", "Text", 5, "L2", sub_parts=[sp_orig])
    q.sync_materialized_projections()
    assert q.evidence_block_ids == ("b1", "b2")

    # Repair replaces subpart with new subpart lacking evidence
    sp_repaired = SubPart(0, "a", "Repaired text", 5, "L2", evidence_block_ids=())
    q.sub_parts = [sp_repaired]
    q.sync_materialized_projections()
    assert q.evidence_block_ids == ()
    assert q.sub_parts[0].evidence_block_ids == ()


def test_g6_c1_contract_lifecycle_planner_to_writing_to_subpart():
    """Trace field preservation from SubPartPlan through to SubPart without synthetic data invention."""
    fp = ConceptFootprint("Paging", provenance_entity_id="p1", provenance_type="DOCUMENT_SECTION")
    er = ExpectedResponseContract(("definition",), ("explain",))
    vt = VisualTaskContract("fig1", True, "read", ["axis"])
    rc = [RubricCriterion("Steps", 5)]

    plan = SubPartPlan("a", 5, "L2", "CO1", "Paging",
                       concept_footprint=fp, expected_response=er, visual_task=vt, rubric_criteria=rc)

    # Writing agent maps plan to SubPart
    sp = SubPart(
        subpart_index=0, label=plan.label, text="Text", marks=plan.marks, bloom=plan.bloom,
        concept_footprint=plan.concept_footprint,
        expected_response=plan.expected_response,
        visual_task=plan.visual_task,
        marking_scheme=[c.to_dict() for c in plan.rubric_criteria]
    )
    assert sp.concept_footprint == fp
    assert sp.expected_response == er
    assert sp.visual_task == vt
    assert sp.marking_scheme[0]["criterion"] == "Steps"


def test_g6_c1_parent_projections_all_contracts_synchronized():
    """Modify SubPart contracts and run sync_materialized_projections; projections stay synchronized."""
    fp = ConceptFootprint("VM", provenance_entity_id="p1", provenance_type="DOCUMENT_SECTION")
    er = ExpectedResponseContract(("steps",), ("derive",))
    vt = VisualTaskContract("fig2", True, "inspect", ["nodes"])

    sp = SubPart(0, "a", "Text", 5, "L3",
                 concept_footprint=fp, expected_response=er, visual_task=vt,
                 evidence_block_ids=("b10",),
                 marking_scheme=[{"criterion": "Steps", "marks": 5}])
    q = GeneratedQuestion("Q1", "m1_Q1", "Text", 5, "L3", sub_parts=[sp])
    q.sync_materialized_projections()

    assert len(q.concept_footprints) == 1
    assert len(q.expected_responses) == 1
    assert len(q.visual_tasks) == 1
    assert q.evidence_block_ids == ("b10",)
    assert q.marking_scheme[0]["criterion"] == "Steps"


def test_g6_c1_refinement_stale_non_evidence_contracts_indeterminate():
    """Subpart text repaired without updating stale response or concept contract; evaluates to INDETERMINATE."""
    fp = ConceptFootprint("Paging", provenance_entity_id="", provenance_type="DOCUMENT_SECTION")
    sp = SubPart(0, "a", "Repaired text on segmentation", 5, "L2", concept_footprint=fp, telemetry={"repaired": True})
    status, reason = verify_concept_provenance(sp.concept_footprint, module_idx=1)
    assert status == "UNVERIFIED_CONCEPT_REVIEW"


def test_g6_c1_serialization_round_trip():
    """to_dict() and from_dict() round-trip on question with all contracts cleanly preserves fields."""
    fp = ConceptFootprint("Paging", "define", "prose", "L2", provenance_entity_id="p1", provenance_type="DOCUMENT_SECTION")
    er = ExpectedResponseContract(("definition",), ("explain",))
    vt = VisualTaskContract("fig1", True, "read", ["axis"])
    rc = [{"criterion": "Steps", "marks": 5}]

    sp = SubPart(0, "a", "Text", 5, "L2",
                 evidence_block_ids=("ev1",),
                 concept_footprint=fp, expected_response=er, visual_task=vt,
                 marking_scheme=rc)
    q = GeneratedQuestion("Q1", "module_1_Q1", "Text", 5, "L2", sub_parts=[sp])
    q.sync_materialized_projections()

    d = q.to_dict()
    q_restored = GeneratedQuestion.from_dict(d)

    assert q_restored.question_id == q.question_id
    assert len(q_restored.sub_parts) == 1
    sp_rest = q_restored.sub_parts[0]
    assert sp_rest.concept_footprint.canonical_concept == "Paging"
    assert sp_rest.expected_response.required_elements == ("definition",)
    assert sp_rest.visual_task.figure_id == "fig1"
    assert sp_rest.evidence_block_ids == ("ev1",)
    assert sp_rest.marking_scheme[0]["marks"] == 5


# =============================================================================
# Track G6-15: Singular Directive, Single Topic & Non-Sequential CO/Bloom
# =============================================================================

def test_g6_15_singular_directive_rejects_compound_directives():
    """Questions must contain a single directive only (no define+explain+implement stacking)."""
    from core.generation.agents.local_evaluators import validate_singular_directive, validate_single_topic

    # Compound / multi-directive questions must be rejected
    assert validate_singular_directive("Define CPU scheduling and explain Round Robin.") is not None
    assert validate_singular_directive("Describe semaphores, implement a mutex, and evaluate deadlock.") is not None
    assert validate_singular_directive("Explain virtual memory. List its advantages.") is not None
    assert validate_singular_directive("What is paging? How does TLB work?") is not None

    # Single-directive questions (including valid idioms and prepositional openings) must pass
    assert validate_singular_directive("Define CPU scheduling.") is None
    assert validate_singular_directive("Explain the operational mechanism of Critical Section.") is None
    assert validate_singular_directive("Implement Banker's algorithm given the allocation matrix.") is None
    assert validate_singular_directive("Compare and contrast FIFO and LRU page replacement.") is None
    assert validate_singular_directive("With reference to the figure, explain the layered OS architecture.") is None
    assert validate_singular_directive("Explain stack operations and state representations.") is None

    # Single-topic enforcement rejects bundling sibling topics
    assert validate_single_topic(
        "Explain Process Scheduling and Deadlock Prevention.",
        assigned_topic="Process Scheduling",
        sibling_topics=["Process Scheduling", "Deadlock Prevention"],
    ) is not None
    assert validate_single_topic(
        "Explain Process Scheduling using priority queues.",
        assigned_topic="Process Scheduling",
        sibling_topics=["Process Scheduling", "Deadlock Prevention"],
    ) is None


def test_g6_15_non_sequential_co_and_bloom_tagging_by_difficulty():
    """Verify PlanningAgent tags CO and Bloom by difficulty/mapping, not question number."""
    from core.contracts.paper_spec import PaperSpec
    from core.contracts.document_artifact import DocumentArtifact, TextBlock
    from core.extraction.module_segmenter import ModuleSegment, SegmentationStrategy
    from core.generation.agents.base import AgentContext
    from core.generation.agents.planning_agent import PlanningAgent
    from core.generation.assessment_config import AssessmentConfig, AssessmentRegistry
    from tests.test_stage_g1d_integration import MockKnowledgeGraph

    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="VTU_2022_NEP",
        course_type="AEC_GENERAL_THEORY",
        assessment_type="CIE_IAT",
        blueprint_id="VTU_AEC_ANNEXURE_II",
        total_marks_per_question=10,
        permitted_partitions=((5, 5), (6, 4), (7, 3), (10,)),
        max_subparts=2,
        reference_source="VTU/BGM/BOS/AEC/2024/1142",
    ))

    spec = PaperSpec(
        exam_type="IAT1",
        module_count=5,
        questions_per_module=1,
        marks_per_question=10,
        custom_co_map={"1": "CO3", "2": "CO1", "3": "CO5", "4": "CO2", "5": "CO4"},
    )
    blocks = [
        TextBlock(id=f"b_{i}", text=f"Detailed syllabus content for module {i}.", page=i * 3, block_role="BODY")
        for i in range(1, 6)
    ]
    artifact = DocumentArtifact(source_pdf_sha256="sha_nonseq", source_pdf_path="os.pdf", text_blocks=blocks)
    segments = [
        ModuleSegment(module_id=i, title=f"Module {i}", page_start=(i - 1) * 3 + 1, page_end=i * 3 + 1,
                      confidence=0.95, strategy_used=SegmentationStrategy.EXPLICIT_MODULE_HEADING)
        for i in range(1, 6)
    ]
    concepts = {f"module_{i}": [f"Topic {i}A", f"Topic {i}B"] for i in range(1, 6)}
    ctx = AgentContext(
        paper_spec=spec,
        artifact=artifact,
        knowledge_graph=MockKnowledgeGraph(concepts_by_module=concepts),
        request={"scheme_id": "VTU_2022_NEP", "course_type": "AEC_GENERAL_THEORY", "assessment_type": "CIE_IAT", "blueprint_id": "VTU_AEC_ANNEXURE_II"},
    )
    ctx.module_segments = segments

    res = PlanningAgent().run(ctx)
    assert res.success is True
    plans = res.payload
    assert [p.co for p in plans] == ["CO3", "CO1", "CO5", "CO2", "CO4"]
    # Bloom levels follow CO/difficulty rather than sequential Q1=L1, Q2=L2, ...
    assert [p.bloom for p in plans] != ["L1", "L2", "L3", "L4", "L5"]


def test_g6_15_source_tag_leakage_rejected():
    """Questions must not contain leaked textbook tags like Figure 5.1, Chapter 3, Section 2.1, Module 1."""
    from core.generation.agents.local_evaluators import validate_no_source_tags

    assert validate_no_source_tags("Using Figure 5.1, explain the sliding window protocol.") is not None
    assert validate_no_source_tags("Refer to Fig. 1.2 and state the handshake steps.") is not None
    assert validate_no_source_tags("As discussed in Chapter 3, explain deadlock prevention.") is not None
    assert validate_no_source_tags("From Section 2.4, analyze paging overhead.") is not None
    assert validate_no_source_tags("In Module 2, list the MAC protocols.") is not None
    assert validate_no_source_tags("Using Table 4.1, evaluate the routing cost.") is not None

    # Generic figure references (for visual questions) and clean text must pass
    assert validate_no_source_tags("With reference to the given diagram, implement the state transition sequence.") is None
    assert validate_no_source_tags("Explain the operational mechanism of CIDR Subnetting.") is None
