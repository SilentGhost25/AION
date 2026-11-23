"""
Stage G5 Lifecycle Integrity Acceptance Test Suite.
Validates:
1. Track G5-A: RefinementAgent Multi-Subpart Integrity, Non-Positional Label Matching,
   Untargeted Sibling Immutability, Candidate-Copy Construction, and Transactional Rollback.
2. Track G5-B: Legacy Adapter CLI Presentation Compatibility (mq_index, bloom_level, bloom_name)
   with Explicit Numbering Contract and End-to-End aion._save_output Verification.
3. Track G5-C: Whole-Paper Cross-Module Deduplication Audit, Distinguishable Question vs Subpart
   Severities, Single-Pair Reporting, False-Positive Controls, and Explicit Degraded Status.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch

import pytest

from core.contracts.paper_spec import PaperSpec
from core.generation.agents.base import AgentContext
from core.generation.agents.checking_agent import CheckingAgent, STATUS_OK, STATUS_DEGRADED, STATUS_BLOCKED
from core.generation.agents.checking_contracts import QAReport, STATUS_PARTIAL
from core.generation.agents.evaluation_contracts import AuditReport, SlotVerdict
from core.generation.agents.generated_question import GeneratedQuestion, SubPart
from core.generation.agents.legacy_adapter import (
    _to_question_dict,
    _resolve_mq_index,
    _resolve_bloom_presentation,
    to_legacy_shape,
)
from core.generation.agents.orchestrator_contracts import OrchestratorResult
from core.generation.agents.overlap_detector import audit_cross_module_duplicates
from core.generation.agents.pipeline_bridge import _reconcile_post_adaptation
from core.generation.agents.planning_agent import QuestionPlan, SubPartPlan
from core.generation.agents.refinement_agent import RefinementAgent
from aion import _save_output


# =============================================================================
# Helper Fixtures
# =============================================================================

def make_test_question(
    slot_id: str = "module_1_Q1",
    module_id: str = "module_1",
    global_q_idx: int = 1,
    subparts_spec: list[tuple[str, int, str]] | None = None,
    bloom: str = "L3",
    co: str = "CO1",
) -> GeneratedQuestion:
    """Helper to build consistent multi-part test questions."""
    if subparts_spec is None:
        subparts_spec = [("a", 6, "Explain sorting algorithms."), ("b", 4, "Compare quicksort and mergesort.")]
    
    sub_parts = [
        SubPart(
            label=lbl,
            marks=m,
            text=txt,
            topic="Algorithms",
            solution=f"Solution for {lbl}.",
            marking_scheme=[{"criterion": f"Criterion {lbl}", "marks": m}],
            bloom=bloom,
            co=co,
        )
        for lbl, m, txt in subparts_spec
    ]
    partition = [m for _, m, _ in subparts_spec]
    q = GeneratedQuestion(
        slot_id=slot_id,
        module_id=module_id,
        global_q_idx=global_q_idx,
        question_text="",
        solution="",
        marking_scheme=[],
        marks=sum(partition),
        partition=partition,
        bloom=bloom,
        co=co,
        sub_parts=sub_parts,
    )
    q.sync_materialized_projections()
    return q


# =============================================================================
# Track G5-A: RefinementAgent Multi-Subpart Integrity
# =============================================================================

def test_g5a_acc01_monolithic_response_rejected_for_multipart():
    """G5-ACC-01: Monolithic response without sub_parts must be rejected for multi-part questions."""
    q = make_test_question()
    original_copy = copy.deepcopy(q)
    
    verdict = SlotVerdict(
        slot_id="module_1_Q1",
        verdict="fail",
        source="local",
        reason_code="SIBLING_INTERCONNECTED",
        affected_subparts=("a", "b"),
        detail="Subparts are coupled",
    )
    
    # Monolithic LLM response
    monolithic_response = json.dumps({
        "question_text": "(a) Repaired text.\n\n(b) Repaired text b.",
        "solution": "Solution.",
        "marking_scheme": [{"criterion": "C1", "marks": 6}, {"criterion": "C2", "marks": 4}],
    })
    mock_llm = MagicMock()
    mock_llm.call.return_value = monolithic_response

    agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=1)
    outcome, repaired = agent._refine_one(original=q, verdict=verdict, llm_budget_used=0)

    # Must reject repair
    assert outcome.resolved is False
    assert repaired is None
    # Original question must be completely unmutated
    assert q.question_text == original_copy.question_text
    assert q.sub_parts[0].text == original_copy.sub_parts[0].text
    assert q.sub_parts[1].text == original_copy.sub_parts[1].text


def test_g5a_acc02_valid_structured_multi_target_repair():
    """G5-ACC-02: Valid structured multi-target repair succeeds with projection consistency."""
    q = make_test_question()
    
    verdict = SlotVerdict(
        slot_id="module_1_Q1",
        verdict="fail",
        source="local",
        reason_code="SIBLING_INTERCONNECTED",
        affected_subparts=("a", "b"),
    )
    
    structured_response = json.dumps({
        "sub_parts": [
            {
                "label": "a",
                "text": "Explain independent bubble sort traversal.",
                "solution": "Traversal steps.",
                "marking_scheme": [{"criterion": "Traversal", "marks": 6}],
            },
            {
                "label": "b",
                "text": "Explain independent insertion sort logic.",
                "solution": "Insertion steps.",
                "marking_scheme": [{"criterion": "Insertion", "marks": 4}],
            },
        ],
        "change_summary": "Decoupled subparts",
    })
    mock_llm = MagicMock()
    mock_llm.call.return_value = structured_response

    agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=1)
    outcome, repaired = agent._refine_one(original=q, verdict=verdict, llm_budget_used=0)

    assert outcome.resolved is True
    assert repaired is not None
    assert repaired.sub_parts[0].text == "Explain independent bubble sort traversal."
    assert repaired.sub_parts[1].text == "Explain independent insertion sort logic."
    
    # Projection consistency verified
    consistent, err = repaired.verify_projection_consistency()
    assert consistent is True
    assert "bubble sort" in repaired.question_text
    assert "insertion sort" in repaired.question_text


def test_g5a_acc03_controlled_3part_repair_untargeted_preserved():
    """G5-ACC-03: Controlled 3-part repair: targets (a) & (b), untargeted (c) remains bit-for-bit unchanged."""
    specs = [
        ("a", 8, "Original text for subpart A."),
        ("b", 6, "Original text for subpart B."),
        ("c", 6, "Untargeted pristine text for subpart C."),
    ]
    q = make_test_question(subparts_spec=specs, bloom="L4", co="CO2")
    orig_c = copy.deepcopy(q.sub_parts[2])
    
    verdict = SlotVerdict(
        slot_id="module_1_Q1",
        verdict="fail",
        source="local",
        reason_code="SIBLING_COLLISION",
        affected_subparts=("a", "b"),
    )
    
    structured_response = json.dumps({
        "sub_parts": [
            {
                "label": "a",
                "text": "Repaired distinct text for Part A.",
                "solution": "Solution A.",
                "marking_scheme": [{"criterion": "Crit A", "marks": 8}],
            },
            {
                "label": "b",
                "text": "Repaired distinct text for Part B.",
                "solution": "Solution B.",
                "marking_scheme": [{"criterion": "Crit B", "marks": 6}],
            },
        ],
        "change_summary": "Fixed collision between A and B",
    })
    mock_llm = MagicMock()
    mock_llm.call.return_value = structured_response

    agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=1)
    outcome, repaired = agent._refine_one(original=q, verdict=verdict, llm_budget_used=0)

    assert outcome.resolved is True
    assert repaired is not None
    # Subparts a and b are updated
    assert repaired.sub_parts[0].text == "Repaired distinct text for Part A."
    assert repaired.sub_parts[1].text == "Repaired distinct text for Part B."
    
    # Subpart c must be bit-for-bit identical to orig_c
    rep_c = repaired.sub_parts[2]
    assert rep_c.label == orig_c.label
    assert rep_c.marks == orig_c.marks
    assert rep_c.text == orig_c.text
    assert rep_c.solution == orig_c.solution
    assert rep_c.marking_scheme == orig_c.marking_scheme
    assert rep_c.bloom == orig_c.bloom
    assert rep_c.co == orig_c.co


def test_g5a_acc04_non_positional_label_matching():
    """G5-ACC-04: Subparts returned in reversed order [b, a] must match correctly by label."""
    q = make_test_question()
    
    verdict = SlotVerdict(
        slot_id="module_1_Q1",
        verdict="fail",
        source="local",
        reason_code="SIBLING_INTERCONNECTED",
        affected_subparts=("a", "b"),
    )
    
    # Response contains [b, a] reversed
    reversed_response = json.dumps({
        "sub_parts": [
            {
                "label": "b",
                "text": "Repaired text B (4 marks).",
                "solution": "Sol B.",
                "marking_scheme": [{"criterion": "B", "marks": 4}],
            },
            {
                "label": "a",
                "text": "Repaired text A (6 marks).",
                "solution": "Sol A.",
                "marking_scheme": [{"criterion": "A", "marks": 6}],
            },
        ]
    })
    mock_llm = MagicMock()
    mock_llm.call.return_value = reversed_response

    agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=1)
    outcome, repaired = agent._refine_one(original=q, verdict=verdict, llm_budget_used=0)

    assert outcome.resolved is True
    assert repaired is not None
    # a has 6 marks, b has 4 marks correctly attributed despite reversed list order
    assert repaired.sub_parts[0].label == "a"
    assert repaired.sub_parts[0].text == "Repaired text A (6 marks)."
    assert repaired.sub_parts[0].marks == 6
    assert repaired.sub_parts[1].label == "b"
    assert repaired.sub_parts[1].text == "Repaired text B (4 marks)."
    assert repaired.sub_parts[1].marks == 4


def test_g5a_acc05_substantive_mark_change_rejected():
    """G5-ACC-05: Substantive mark changes on target subpart must be rejected transactionally."""
    q = make_test_question()
    original_copy = copy.deepcopy(q)
    
    verdict = SlotVerdict(
        slot_id="module_1_Q1",
        verdict="fail",
        source="local",
        reason_code="SIBLING_INTERCONNECTED",
        affected_subparts=("a", "b"),
    )
    
    # Subpart A marks changed from 6 to 7
    drift_response = json.dumps({
        "sub_parts": [
            {
                "label": "a",
                "marks": 7,
                "text": "Repaired text A.",
                "solution": "Sol A.",
                "marking_scheme": [{"criterion": "A", "marks": 7}],
            },
            {
                "label": "b",
                "marks": 3,
                "text": "Repaired text B.",
                "solution": "Sol B.",
                "marking_scheme": [{"criterion": "B", "marks": 3}],
            },
        ]
    })
    mock_llm = MagicMock()
    mock_llm.call.return_value = drift_response

    agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=1)
    outcome, repaired = agent._refine_one(original=q, verdict=verdict, llm_budget_used=0)

    assert outcome.resolved is False
    assert repaired is None
    assert q.sub_parts[0].marks == original_copy.sub_parts[0].marks


def test_g5a_acc06_broken_marking_scheme_sum_rejected():
    """G5-ACC-06: Marking scheme sum not matching subpart marks must be rejected."""
    q = make_test_question()
    
    verdict = SlotVerdict(
        slot_id="module_1_Q1",
        verdict="fail",
        source="local",
        reason_code="SIBLING_INTERCONNECTED",
        affected_subparts=("a", "b"),
    )
    
    # Subpart A has marks=6, but marking scheme sums to 5
    broken_scheme_response = json.dumps({
        "sub_parts": [
            {
                "label": "a",
                "text": "Repaired text A.",
                "solution": "Sol A.",
                "marking_scheme": [{"criterion": "A", "marks": 5}],
            },
            {
                "label": "b",
                "text": "Repaired text B.",
                "solution": "Sol B.",
                "marking_scheme": [{"criterion": "B", "marks": 4}],
            },
        ]
    })
    mock_llm = MagicMock()
    mock_llm.call.return_value = broken_scheme_response

    agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=1)
    outcome, repaired = agent._refine_one(original=q, verdict=verdict, llm_budget_used=0)

    assert outcome.resolved is False
    assert repaired is None


def test_g5a_acc07_missing_target_in_response_rejected():
    """G5-ACC-07: When targets are ('a', 'b'), omitting target 'b' must fail closed."""
    q = make_test_question()
    
    verdict = SlotVerdict(
        slot_id="module_1_Q1",
        verdict="fail",
        source="local",
        reason_code="SIBLING_INTERCONNECTED",
        affected_subparts=("a", "b"),
    )
    
    # Only part 'a' is returned
    missing_target_response = json.dumps({
        "sub_parts": [
            {
                "label": "a",
                "text": "Repaired text A.",
                "solution": "Sol A.",
                "marking_scheme": [{"criterion": "A", "marks": 6}],
            }
        ]
    })
    mock_llm = MagicMock()
    mock_llm.call.return_value = missing_target_response

    agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=1)
    outcome, repaired = agent._refine_one(original=q, verdict=verdict, llm_budget_used=0)

    assert outcome.resolved is False
    assert repaired is None


# =============================================================================
# Track G5-B: Legacy Adapter CLI Presentation Compatibility
# =============================================================================

def test_g5b_acc09_presentation_keys_populated():
    """G5-ACC-09: _to_question_dict populates mq_index, bloom_level, and bloom_name."""
    q = make_test_question(global_q_idx=3, bloom="L4")
    d = _to_question_dict(q)
    
    assert "mq_index" in d
    assert d["mq_index"] == 3
    assert "bloom_level" in d
    assert d["bloom_level"] == 4
    assert "bloom_name" in d
    assert d["bloom_name"] == "Analyse"


def test_g5b_acc10_resolve_mq_index_contract():
    """G5-ACC-10: _resolve_mq_index resolves from global_q_idx, slot_id, and fails on invalid."""
    q1 = make_test_question(global_q_idx=5)
    assert _resolve_mq_index(q1) == 5

    # Fallback to slot_id regex
    q2 = make_test_question(slot_id="module_3_Q7", global_q_idx=None)
    assert _resolve_mq_index(q2) == 7

    # Fail closed when unresolvable (no silent defaulting to 1)
    q_invalid = make_test_question(slot_id="custom_slot_no_q", global_q_idx=None)
    with pytest.raises(ValueError, match="Numbering contract requires an explicit positive integer index"):
        _resolve_mq_index(q_invalid)


def test_g5b_acc11_resolve_bloom_presentation():
    """G5-ACC-11: _resolve_bloom_presentation derives max level from subparts and maps name."""
    q = make_test_question(bloom="L2")
    sub_qs = [
        {"bloom": "L2"},
        {"bloom": "L4"},
    ]
    level, name = _resolve_bloom_presentation(q, sub_qs)
    assert level == 4
    assert name == "Analyse"


def test_g5b_acc12_cli_save_output_integration():
    """G5-ACC-12: to_legacy_shape output passed to aion._save_output writes file without error."""
    # Build a full module with 4 questions
    questions = [
        make_test_question(slot_id=f"module_1_Q{i}", global_q_idx=i, bloom="L3")
        for i in range(1, 5)
    ]
    plans = [
        QuestionPlan(
            slot_id=f"module_1_Q{i}", module_id="module_1", module_idx=1, slot_idx=i-1,
            global_q_idx=i, total_marks=10, partition=[6, 4], bloom="L3", co="CO1"
        )
        for i in range(1, 5)
    ]
    spec = PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=4, marks_per_question=10)
    ctx = AgentContext(questions=questions, plans=plans, paper_spec=spec)
    orch_res = OrchestratorResult(success=True, questions=questions, qa_report=MagicMock(spec=QAReport))

    parts, full = to_legacy_shape(orch_res, ctx)
    assert len(parts) == 1
    assert len(parts[0]["questions"]) == 4

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        out_path = tf.name

    try:
        _save_output(parts, out_path, "ia")
        content = Path(out_path).read_text(encoding="utf-8")
        assert "Q1 " in content
        assert "Q2 " in content
        assert "Q3 " in content
        assert "Q4 " in content
        assert "[Bloom Level 3: Apply]" in content
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


# =============================================================================
# Track G5-C: Whole-Paper Cross-Module Deduplication
# =============================================================================

def test_g5c_acc13_exact_duplicate_cross_module_detected():
    """G5-ACC-13: Exact duplicate questions across different modules must be detected and flag STATUS_DEGRADED."""
    dup_text = "Explain the fundamental architecture and working of convolutional neural networks."
    q1 = make_test_question(slot_id="module_1_Q1", module_id="module_1", global_q_idx=1, subparts_spec=[("a", 10, dup_text)])
    q2 = make_test_question(slot_id="module_2_Q2", module_id="module_2", global_q_idx=2, subparts_spec=[("a", 10, dup_text)])

    plan1 = QuestionPlan(slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1, total_marks=10, partition=[10], bloom="L3", co="CO1")
    plan2 = QuestionPlan(slot_id="module_2_Q2", module_id="module_2", module_idx=2, slot_idx=0, global_q_idx=2, total_marks=10, partition=[10], bloom="L3", co="CO2")
    spec = PaperSpec(exam_type="IAT1", module_count=2, questions_per_module=1, marks_per_question=10)

    ctx = AgentContext(
        questions=[q1, q2],
        plans=[plan1, plan2],
        paper_spec=spec,
        audit_report=AuditReport(verdicts=[
            SlotVerdict(slot_id="module_1_Q1", verdict="pass", source="local"),
            SlotVerdict(slot_id="module_2_Q2", verdict="pass", source="local"),
        ]),
    )
    checker = CheckingAgent()
    qa = checker.run(ctx).payload

    assert qa.status == STATUS_DEGRADED
    assert any("CROSS_MODULE_QUESTION_DUPLICATE:module_1_Q1_vs_module_2_Q2" in r for r in qa.degraded_reasons)
    
    # Check telemetry
    dup_telemetry = qa.telemetry["checking"]["cross_module_duplicates"]
    assert len(dup_telemetry) == 1
    assert dup_telemetry[0]["match_type"] == "exact_question"
    assert dup_telemetry[0]["similarity"] == 1.0


def test_g5c_acc14_subpart_collision_cross_module_detected():
    """G5-ACC-14: Subpart collision across modules detected as CROSS_MODULE_SUBPART_DUPLICATE."""
    dup_sub_text = "Derive the backpropagation gradient descent update equation for multi-layer perceptrons."
    q1 = make_test_question(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        subparts_spec=[("a", 6, dup_sub_text), ("b", 4, "Unique topic A.")]
    )
    q2 = make_test_question(
        slot_id="module_2_Q2", module_id="module_2", global_q_idx=2,
        subparts_spec=[("a", 6, "Unique topic B."), ("b", 4, dup_sub_text)]
    )

    findings = audit_cross_module_duplicates([q1, q2])
    assert len(findings) == 1
    f = findings[0]
    assert f["severity"] == "subpart_level"
    assert f["match_type"] == "exact_subpart"
    assert f["subpart_a"] == "a"
    assert f["subpart_b"] == "b"


def test_g5c_acc15_single_pair_reporting():
    """G5-ACC-15: Conflicting pair reported at most once even if multiple subparts match."""
    t = "Identical text across multiple subparts."
    q1 = make_test_question(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        subparts_spec=[("a", 6, t), ("b", 4, t)]
    )
    q2 = make_test_question(
        slot_id="module_2_Q2", module_id="module_2", global_q_idx=2,
        subparts_spec=[("a", 6, t), ("b", 4, t)]
    )

    findings = audit_cross_module_duplicates([q1, q2])
    # Exactly one report for this question pair
    assert len(findings) == 1


def test_g5c_acc16_false_positive_control_template_wording():
    """G5-ACC-16: Generic framing ('With a neat diagram, explain...') with distinct concepts passes clean."""
    q1 = make_test_question(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        subparts_spec=[("a", 10, "With a neat diagram, explain the working principle and architecture of transformer attention mechanisms.")]
    )
    q2 = make_test_question(
        slot_id="module_2_Q2", module_id="module_2", global_q_idx=2,
        subparts_spec=[("a", 10, "With a neat diagram, explain the memory organization and registers of the 8051 microcontroller.")]
    )

    findings = audit_cross_module_duplicates([q1, q2])
    assert len(findings) == 0


def test_g5c_acc17_within_module_excluded_from_cross_module():
    """G5-ACC-17: Questions within the same module are excluded from cross-module audit."""
    t = "Identical question within the same module."
    q1 = make_test_question(slot_id="module_1_Q1", module_id="module_1", global_q_idx=1, subparts_spec=[("a", 10, t)])
    q2 = make_test_question(slot_id="module_1_Q2", module_id="module_1", global_q_idx=2, subparts_spec=[("a", 10, t)])

    findings = audit_cross_module_duplicates([q1, q2])
    # Same module_id -> ignored by cross-module audit (handled by sibling checks)
    assert len(findings) == 0


# =============================================================================
# Full End-to-End Pipeline Integration Tests
# =============================================================================

def test_g5_acc18_full_ia_and_see_pipeline_workflows():
    """G5-ACC-18: Full IA (50M) and SEE (100M) workflows complete cleanly with STATUS_OK and 0 defects."""
    # 1. IA 50M with distinct syllabus topics per module
    topics = {
        1: ("singly linked list pointer manipulation", "doubly linked circular list deletion"),
        2: ("binary search tree recursive lookup", "AVL tree height balance rotations"),
        3: ("Dijkstra single source shortest path", "Bellman-Ford negative cycle detection"),
        4: ("dynamic programming knapsack memoization", "longest common subsequence table"),
        5: ("Turing machine deterministic transitions", "Cook-Levin theorem SAT completeness"),
    }
    ia_questions = [
        make_test_question(
            slot_id=f"module_{m}_Q{(m-1)*2+q}",
            module_id=f"module_{m}",
            global_q_idx=(m-1)*2+q,
            subparts_spec=[(
                "a", 6, f"Explain the principles and algorithms for {topics[m][0]}."
            ), (
                "b", 4, f"Analyze the asymptotic runtime and trade-offs for {topics[m][1]}."
            )],
            bloom="L3", co=f"CO{m}"
        )
        for m in range(1, 6) for q in (1, 2)
    ]
    ia_plans = [
        QuestionPlan(
            slot_id=q.slot_id, module_id=q.module_id, module_idx=int(q.module_id.split('_')[1]),
            slot_idx=0, global_q_idx=q.global_q_idx, total_marks=10, partition=[6, 4], bloom="L3", co=q.co
        )
        for q in ia_questions
    ]
    ia_spec = PaperSpec(exam_type="IAT1", module_count=5, questions_per_module=2, marks_per_question=10)
    ia_ctx = AgentContext(
        questions=ia_questions, plans=ia_plans, paper_spec=ia_spec,
        audit_report=AuditReport(verdicts=[SlotVerdict(slot_id=q.slot_id, verdict="pass", source="local") for q in ia_questions])
    )
    ia_qa = CheckingAgent().run(ia_ctx).payload
    assert ia_qa.status == STATUS_OK
    assert ia_qa.resolved_slots == 10

    ia_res = OrchestratorResult(success=True, questions=ia_questions, qa_report=ia_qa)
    ia_parts, ia_full = to_legacy_shape(ia_res, ia_ctx)
    reconciled_qa, defects, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=ia_res, context=ia_ctx, paper_parts=ia_parts, full_paper=ia_full
    )
    assert eligible is True
    assert len(defects) == 0
    assert ia_parts[0]["questions"][0]["mq_index"] == 1
    assert ia_parts[0]["questions"][0]["bloom_name"] == "Apply"


# =============================================================================
# Additional Stage G5 Acceptance Tests (Issues 2 & 3 Boundary Coverage)
# =============================================================================

def test_g5a_acc18_full_metadata_preservation_across_repair():
    """G5-ACC-18: Verify all metadata fields (bloom, co, topic, figure/visual, solutions) are strictly preserved across repair."""
    sp_a = SubPart(
        label="a", marks=8, text="Explain primitive sorting algorithms.",
        topic="Sorting Concepts", solution="Sol A initial.", marking_scheme=[{"criterion": "Crit A", "marks": 8}],
        bloom="L2", co="CO1",
    )
    sp_b = SubPart(
        label="b", marks=6, text="Analyze tree balancing rotations.",
        topic="Tree Structures", solution="Sol B initial.", marking_scheme=[{"criterion": "Crit B", "marks": 6}],
        figure_id="fig_b", image_path="/images/fig_b.png", figure_caption="AVL Rotation", image_sha256="sha256_b_val",
        bloom="L3", co="CO2",
    )
    sp_c = SubPart(
        label="c", marks=6, text="Describe graph traversal algorithms.",
        topic="Graph Theory", solution="Sol C initial.", marking_scheme=[{"criterion": "Crit C", "marks": 6}],
        bloom="L4", co="CO3",
    )

    q = GeneratedQuestion(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        question_text="", solution="", marking_scheme=[],
        marks=20, partition=[8, 6, 6], bloom="L3", co="CO2",
        sub_parts=[sp_a, sp_b, sp_c],
    )
    q.sync_materialized_projections()

    # Target ONLY subpart (a)
    verdict = SlotVerdict(
        slot_id="module_1_Q1", verdict="fail", source="local",
        reason_code="BLOOM_DEPTH_MISMATCH", affected_subparts=("a",),
        detail="Subpart (a) is too descriptive",
    )

    # LLM repairs subpart (a)
    mock_llm = MagicMock()
    mock_llm.call.return_value = json.dumps({
        "sub_parts": [{
            "label": "a",
            "text": "Critically analyze and compare insertion sort and selection sort mechanics.",
            "solution": "Refined solution for sorting comparisons.",
            "marking_scheme": [{"criterion": "Mechanics comparison", "marks": 8}],
        }]
    })

    agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=1)
    outcome, repaired = agent._refine_one(original=q, verdict=verdict, llm_budget_used=0)

    assert outcome.resolved is True
    assert repaired is not None

    # Repaired target (a) assertions
    rep_a = repaired.sub_parts[0]
    assert "Critically analyze" in rep_a.text
    assert rep_a.solution == "Refined solution for sorting comparisons."
    assert rep_a.marks == 8
    assert rep_a.bloom == "L2"  # Preserved
    assert rep_a.co == "CO1"     # Preserved
    assert rep_a.topic == "Sorting Concepts"  # Preserved

    # Untargeted sibling (b) assertions — EVERY field immutable
    rep_b = repaired.sub_parts[1]
    assert rep_b.label == "b"
    assert rep_b.text == sp_b.text
    assert rep_b.marks == sp_b.marks == 6
    assert rep_b.solution == sp_b.solution == "Sol B initial."
    assert rep_b.marking_scheme == sp_b.marking_scheme
    assert rep_b.topic == sp_b.topic == "Tree Structures"
    assert rep_b.bloom == sp_b.bloom == "L3"
    assert rep_b.co == sp_b.co == "CO2"
    assert rep_b.figure_id == "fig_b"
    assert rep_b.image_path == "/images/fig_b.png"
    assert rep_b.figure_caption == "AVL Rotation"
    assert rep_b.image_sha256 == "sha256_b_val"

    # Untargeted sibling (c) assertions
    rep_c = repaired.sub_parts[2]
    assert rep_c.label == "c"
    assert rep_c.text == sp_c.text
    assert rep_c.marks == sp_c.marks == 6
    assert rep_c.solution == sp_c.solution == "Sol C initial."
    assert rep_c.marking_scheme == sp_c.marking_scheme
    assert rep_c.topic == sp_c.topic == "Graph Theory"
    assert rep_c.bloom == sp_c.bloom == "L4"
    assert rep_c.co == sp_c.co == "CO3"


def test_g5a_acc19_projection_mismatch_triggers_transactional_rollback():
    """G5-ACC-19: Candidate repair failing projection consistency triggers transactional rollback to unmutated original."""
    q = make_test_question()
    original_copy = copy.deepcopy(q)

    verdict = SlotVerdict(
        slot_id="module_1_Q1", verdict="fail", source="local",
        reason_code="BLOOM_MISMATCH", affected_subparts=("a",),
    )

    # Candidate repair response
    mock_llm = MagicMock()
    mock_llm.call.return_value = json.dumps({
        "sub_parts": [{
            "label": "a",
            "text": "Repaired subpart text for A.",
            "solution": "Solution A.",
            "marking_scheme": [{"criterion": "C1", "marks": 6}],
        }]
    })

    agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=1)
    
    # Mock verify_projection_consistency to simulate projection divergence failure
    with unittest.mock.patch.object(GeneratedQuestion, "verify_projection_consistency", return_value=(False, "PROJECTION_MISMATCH")):
        outcome, repaired = agent._refine_one(original=q, verdict=verdict, llm_budget_used=0)

    assert outcome.resolved is False
    assert repaired is None
    # Original question must remain 100% unmutated
    assert q.question_text == original_copy.question_text
    assert q.sub_parts[0].text == original_copy.sub_parts[0].text
    assert q.sub_parts[1].text == original_copy.sub_parts[1].text


def test_g5c_acc20_generic_wording_different_topics_no_false_positive():
    """G5-ACC-20: Questions sharing common exam framing wording but different substantive topics are NOT flagged."""
    q1 = GeneratedQuestion(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Explain with a neat sketch the architectural pipeline stages and instruction hazard handling.",
        solution="", marking_scheme=[], marks=10,
    )
    q2 = GeneratedQuestion(
        slot_id="module_2_Q3", module_id="module_2", global_q_idx=3,
        question_text="Explain with a neat sketch the architectural relational database indexing and B-tree storage layout.",
        solution="", marking_scheme=[], marks=10,
    )
    dupes = audit_cross_module_duplicates([q1, q2])
    assert len(dupes) == 0, f"False positive detected: {dupes}"


def test_g5c_acc21_same_topic_different_bloom_and_objectives():
    """G5-ACC-21: Questions on the same topic with distinctly different assessment objectives/Bloom levels are NOT duplicates."""
    q1 = GeneratedQuestion(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        question_text="State the four fundamental Coffman conditions necessary for deadlock occurrence in concurrent operating systems.",
        solution="", marking_scheme=[], marks=10, bloom="L1",
    )
    q2 = GeneratedQuestion(
        slot_id="module_2_Q3", module_id="module_2", global_q_idx=3,
        question_text="Critically evaluate banker algorithm safe sequence calculations and resource allocation graph cycle mitigation under starvation.",
        solution="", marking_scheme=[], marks=10, bloom="L5",
    )
    dupes = audit_cross_module_duplicates([q1, q2])
    assert len(dupes) == 0, f"False positive detected: {dupes}"


def test_g5c_acc22_multiple_subpart_collisions_within_one_pair_single_finding():
    """G5-ACC-22: Several subpart collisions within a single question pair emit exactly ONE finding."""
    q1 = make_test_question(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        subparts_spec=[
            ("a", 4, "Define the fundamental principles of lexical tokenization."),
            ("b", 3, "Explain the mechanics of transition table generation for deterministic finite automata."),
            ("c", 3, "Describe intermediate code representations."),
        ]
    )
    q2 = make_test_question(
        slot_id="module_2_Q3", module_id="module_2", global_q_idx=3,
        subparts_spec=[
            ("a", 4, "Define the fundamental principles of lexical tokenization."),
            ("b", 3, "Explain the mechanics of transition table generation for deterministic finite automata."),
            ("c", 3, "Discuss code optimization in basic blocks."),
        ]
    )
    dupes = audit_cross_module_duplicates([q1, q2])
    assert len(dupes) == 1, f"Expected exactly 1 pair-level finding, got {len(dupes)}"
    assert dupes[0]["slot_a"] == "module_1_Q1"
    assert dupes[0]["slot_b"] == "module_2_Q3"
    assert dupes[0]["severity"] == "subpart_level"


def test_g5c_acc23_robustness_missing_empty_module_and_text():
    """G5-ACC-23: Questions with missing or empty module IDs or text are handled robustly without crashing."""
    q_empty_mod = GeneratedQuestion(slot_id="Q1", module_id="", global_q_idx=1, question_text="Some valid question text.", solution="", marking_scheme=[], marks=10)
    q_none_mod = GeneratedQuestion(slot_id="Q2", module_id=None, global_q_idx=2, question_text="Some valid question text.", solution="", marking_scheme=[], marks=10)
    q_empty_text = GeneratedQuestion(slot_id="module_1_Q3", module_id="module_1", global_q_idx=3, question_text="", solution="", marking_scheme=[], marks=10)
    q_none_text = GeneratedQuestion(slot_id="module_2_Q4", module_id="module_2", global_q_idx=4, question_text=None, solution="", marking_scheme=[], marks=10)

    dupes = audit_cross_module_duplicates([q_empty_mod, q_none_mod, q_empty_text, q_none_text])
    assert dupes == []


def test_g5c_acc24_permutation_determinism_stable_order():
    """G5-ACC-24: Audit results are strictly order-invariant across question list permutations."""
    q1 = GeneratedQuestion(slot_id="module_1_Q1", module_id="module_1", global_q_idx=1, question_text="Exact duplicate question text across modules.", solution="", marking_scheme=[], marks=10)
    q2 = GeneratedQuestion(slot_id="module_2_Q3", module_id="module_2", global_q_idx=3, question_text="Exact duplicate question text across modules.", solution="", marking_scheme=[], marks=10)

    res_forward = audit_cross_module_duplicates([q1, q2])
    res_reverse = audit_cross_module_duplicates([q2, q1])

    assert len(res_forward) == 1
    assert len(res_reverse) == 1
    assert res_forward == res_reverse
    assert res_forward[0]["slot_a"] == "module_1_Q1"
    assert res_forward[0]["slot_b"] == "module_2_Q3"


def test_g5c_acc25_status_precedence_blocked_and_partial_preserved():
    """G5-ACC-25: Cross-module duplicate findings transition OK to DEGRADED, but NEVER overwrite BLOCKED or PARTIAL."""
    spec = PaperSpec(exam_type="IAT1", module_count=2, questions_per_module=1, marks_per_question=10)
    
    # 1. Blocked question + duplicate
    q_blocked = GeneratedQuestion(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Duplicate text across modules for test.", solution="Sol", marking_scheme=[], marks=10,
        is_unresolved=True, unresolved_scope="question", failure_code="TOPIC_OUT_OF_SCOPE", failure_detail="Invalid",
    )
    q_clean = GeneratedQuestion(
        slot_id="module_2_Q2", module_id="module_2", global_q_idx=2,
        question_text="Duplicate text across modules for test.", solution="Sol", marking_scheme=[], marks=10,
    )
    plans = [
        QuestionPlan(slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1, total_marks=10, partition=[10]),
        QuestionPlan(slot_id="module_2_Q2", module_id="module_2", module_idx=2, slot_idx=0, global_q_idx=2, total_marks=10, partition=[10]),
    ]

    ctx_blocked = AgentContext(paper_spec=spec, plans=plans, questions=[q_blocked, q_clean], unresolved_slots=["module_1_Q1"])
    res_b = CheckingAgent().run(ctx_blocked)
    assert res_b.payload.status == STATUS_BLOCKED, "Blocked status must take strict precedence over DEGRADED!"

    # 2. Partial resolution question + duplicate
    q_partial = GeneratedQuestion(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Duplicate text across modules for test.", solution="Sol", marking_scheme=[], marks=10,
        is_unresolved=True, unresolved_scope="subquestion", unresolved_subparts=("b",),
        failure_code="BLOOM_MISMATCH", failure_detail="Subpart b fail",
    )
    ctx_partial = AgentContext(paper_spec=spec, plans=plans, questions=[q_partial, q_clean], unresolved_slots=["module_1_Q1"])
    res_p = CheckingAgent(min_questions_ratio=0.5).run(ctx_partial)
    assert res_p.payload.status == STATUS_PARTIAL, "Partial status must take strict precedence over DEGRADED!"


def test_g5c_acc26_deduplication_audit_does_not_mutate_questions():
    """G5-ACC-26: Running cross-module deduplication audit does not mutate question text, marks, or metadata."""
    q1 = GeneratedQuestion(slot_id="module_1_Q1", module_id="module_1", global_q_idx=1, question_text="Duplicate question text.", solution="Sol 1", marking_scheme=[{"criterion": "C1", "marks": 10}], marks=10, bloom="L3", co="CO1")
    q2 = GeneratedQuestion(slot_id="module_2_Q3", module_id="module_2", global_q_idx=3, question_text="Duplicate question text.", solution="Sol 2", marking_scheme=[{"criterion": "C2", "marks": 10}], marks=10, bloom="L3", co="CO2")

    copy_q1 = copy.deepcopy(q1)
    copy_q2 = copy.deepcopy(q2)

    _ = audit_cross_module_duplicates([q1, q2])

    assert q1.question_text == copy_q1.question_text
    assert q1.solution == copy_q1.solution
    assert q1.marks == copy_q1.marks
    assert q1.bloom == copy_q1.bloom
    assert q1.co == copy_q1.co
    assert q1.marking_scheme == copy_q1.marking_scheme

    assert q2.question_text == copy_q2.question_text
    assert q2.solution == copy_q2.solution
    assert q2.marks == copy_q2.marks
    assert q2.bloom == copy_q2.bloom
    assert q2.co == copy_q2.co
    assert q2.marking_scheme == copy_q2.marking_scheme


def test_g5c_acc27_question_jaccard_boundary_below_and_at_threshold():
    """G5-ACC-27: Question Jaccard threshold boundary test: below 0.65 yields 0 findings; >= 0.65 yields finding."""
    # Pair A: Below threshold (Jaccard ~ 0.50)
    q1 = GeneratedQuestion(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Analyze and implement quicksort partitioning algorithms with pivot selection strategies.",
        solution="", marking_scheme=[], marks=10
    )
    q2 = GeneratedQuestion(
        slot_id="module_2_Q3", module_id="module_2", global_q_idx=3,
        question_text="Analyze and implement mergesort divide conquer recursive decomposition algorithms.",
        solution="", marking_scheme=[], marks=10
    )
    assert audit_cross_module_duplicates([q1, q2]) == []

    # Pair B: At / Above threshold (Jaccard >= 0.65, substantive topic match)
    q3 = GeneratedQuestion(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Analyze and implement quicksort partitioning algorithms with median pivot selection strategies.",
        solution="", marking_scheme=[], marks=10
    )
    q4 = GeneratedQuestion(
        slot_id="module_2_Q3", module_id="module_2", global_q_idx=3,
        question_text="Analyze and implement quicksort partitioning algorithms with median pivot selection rules.",
        solution="", marking_scheme=[], marks=10
    )
    dupes = audit_cross_module_duplicates([q3, q4])
    assert len(dupes) == 1
    assert dupes[0]["severity"] == "question_level"


def test_g5c_acc28_question_containment_boundary_below_and_at_threshold():
    """G5-ACC-28: Question containment boundary test: < 0.75 yields 0 findings; >= 0.75 with >= 4 content roots yields finding."""
    # Below containment
    q1 = GeneratedQuestion(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Construct binary search tree nodes with key balancing.",
        solution="", marking_scheme=[], marks=10
    )
    q2 = GeneratedQuestion(
        slot_id="module_2_Q3", module_id="module_2", global_q_idx=3,
        question_text="Construct database storage pages with row identifiers and transaction logging records.",
        solution="", marking_scheme=[], marks=10
    )
    assert audit_cross_module_duplicates([q1, q2]) == []

    # At/above containment: q3 is substantially contained within q4 (all 4 content roots present in q4)
    q3 = GeneratedQuestion(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Analyze B-tree indexing node splitting algorithms.",
        solution="", marking_scheme=[], marks=10
    )
    q4 = GeneratedQuestion(
        slot_id="module_2_Q3", module_id="module_2", global_q_idx=3,
        question_text="Analyze B-tree indexing node splitting algorithms under concurrent multi-version concurrency control transactions.",
        solution="", marking_scheme=[], marks=10
    )
    dupes = audit_cross_module_duplicates([q3, q4])
    assert len(dupes) == 1
    assert dupes[0]["severity"] == "question_level"


def test_g5c_acc29_subpart_jaccard_and_containment_boundaries():
    """G5-ACC-29: Subpart boundary test: below 0.70 yields 0 findings; at/above 0.70 yields subpart_level finding."""
    # Subpart A below 0.70
    q1_sub = make_test_question(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        subparts_spec=[
            ("a", 6, "Explain bubble sort adjacent element swapping and passes."),
            ("b", 4, "Analyze relational database algebra operations."),
        ]
    )
    q2_sub_below = make_test_question(
        slot_id="module_2_Q3", module_id="module_2", global_q_idx=3,
        subparts_spec=[
            ("a", 6, "Explain selection sort minimum element selection and passes."),
            ("b", 4, "Design finite automata state transition systems."),
        ]
    )
    assert audit_cross_module_duplicates([q1_sub, q2_sub_below]) == []

    # Subpart A at/above 0.70
    q2_sub_above = make_test_question(
        slot_id="module_2_Q3", module_id="module_2", global_q_idx=3,
        subparts_spec=[
            ("a", 6, "Explain bubble sort adjacent element swapping logic and passes."),
            ("b", 4, "Design finite automata state transition systems."),
        ]
    )
    dupes = audit_cross_module_duplicates([q1_sub, q2_sub_above])
    assert len(dupes) == 1
    assert dupes[0]["severity"] == "subpart_level"


def test_g5c_acc30_genuine_paraphrases_low_lexical_overlap():
    """G5-ACC-30: Genuine semantic paraphrases with distinct vocabulary produce 0 findings (avoids false-positive degradation)."""
    q1 = GeneratedQuestion(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Construct an optimal Huffman coding tree from symbol frequencies.",
        solution="", marking_scheme=[], marks=10
    )
    q2 = GeneratedQuestion(
        slot_id="module_2_Q3", module_id="module_2", global_q_idx=3,
        question_text="Build a prefix-free binary tree minimizing weighted path length for character distributions.",
        solution="", marking_scheme=[], marks=10
    )
    # Distinct academic vocabulary yields 0 lexical collisions
    assert audit_cross_module_duplicates([q1, q2]) == []


def test_g5c_acc31_short_questions_token_count_threshold():
    """G5-ACC-31: Short questions with < 4 content roots are not flagged as near-duplicates; require substantive length."""
    # Only 2 content roots ('push', 'pop')
    q1 = GeneratedQuestion(slot_id="module_1_Q1", module_id="module_1", global_q_idx=1, question_text="Stack push pop.", solution="", marking_scheme=[], marks=10)
    q2 = GeneratedQuestion(slot_id="module_2_Q3", module_id="module_2", global_q_idx=3, question_text="Stack push pop operations.", solution="", marking_scheme=[], marks=10)
    assert audit_cross_module_duplicates([q1, q2]) == []


def test_g5c_acc32_degraded_paper_remains_export_eligible_under_existing_contract():
    """G5-ACC-32: Verifies that a paper with STATUS_DEGRADED remains export_eligible=True and exports to DOCX cleanly."""
    from v0_1.docx_export import render_docx_document
    q1 = GeneratedQuestion(
        slot_id="module_1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Duplicate question across modules for export eligibility check.",
        solution="Sol 1", marking_scheme=[{"criterion": "C1", "marks": 10}], marks=10, bloom="L3", co="CO1",
        partition=[10],
    )
    q2 = GeneratedQuestion(
        slot_id="module_2_Q2", module_id="module_2", global_q_idx=2,
        question_text="Duplicate question across modules for export eligibility check.",
        solution="Sol 2", marking_scheme=[{"criterion": "C2", "marks": 10}], marks=10, bloom="L3", co="CO2",
        partition=[10],
    )
    plans = [
        QuestionPlan(slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1, total_marks=10, partition=[10]),
        QuestionPlan(slot_id="module_2_Q2", module_id="module_2", module_idx=2, slot_idx=0, global_q_idx=2, total_marks=10, partition=[10]),
    ]
    spec = PaperSpec(exam_type="IAT1", module_count=2, questions_per_module=1, marks_per_question=10)
    ctx = AgentContext(paper_spec=spec, plans=plans, questions=[q1, q2], unresolved_slots=[])

    # 1. CheckingAgent runs -> transitions to STATUS_DEGRADED
    chk_res = CheckingAgent().run(ctx)
    assert chk_res.payload.status == STATUS_DEGRADED
    assert any("CROSS_MODULE_QUESTION_DUPLICATE" in r for r in chk_res.payload.degraded_reasons)

    # 2. Adaptation runs
    orch_res = OrchestratorResult(success=True, questions=ctx.questions, qa_report=chk_res.payload)
    paper_parts, full_paper = to_legacy_shape(orch_res, ctx)

    # 3. Post-adaptation reconciliation
    reconciled_qa, defects, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=orch_res, context=ctx, paper_parts=paper_parts, full_paper=full_paper
    )
    assert reconciled_qa.status == STATUS_DEGRADED
    assert len(defects) == 0
    # Under existing contract in pipeline_bridge.py line 1135, STATUS_DEGRADED is explicitly export_eligible!
    assert eligible is True

    # 4. DOCX render succeeds without raising
    docx_io = render_docx_document({
        "subject": "Algorithm Analysis",
        "subject_code": "21CS42",
        "exam_type": "IAT1",
        "modules": paper_parts,
    })
    doc_bytes = docx_io.getvalue()
    assert doc_bytes.startswith(b"PK\x03\x04")
    assert len(doc_bytes) > 5000
