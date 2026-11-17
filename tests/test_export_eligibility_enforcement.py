"""
Unit and Integration Tests for Phase 4 Stage 2C:
CheckingAgent Canonical Slot Accounting, Export Eligibility Enforcement & Status Synchronization.

Covers:
- Resolvability checking (_is_resolvable)
- Authoritative expected slot validation (_get_validated_expected_slots)
- Two-tier accounting & strict partition invariants (_account_slots)
- Cross-field Stage 2A lifecycle consistency validation
- Precedence of fatal integrity defects over partial status
- Positive controls for valid question-level and subquestion-level failures
- Missing question object handling with and without registry entry
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import pytest

from core.contracts.paper_spec import PaperSpec
from core.generation.agents.base import AgentContext
from core.generation.agents.checking_agent import CheckingAgent, _is_positive_int
from core.generation.agents.checking_contracts import (
    BLOCK_REASON_INSUFFICIENT_QUESTIONS,
    BLOCK_REASON_UNRESOLVED_SLOTS,
    BLOCK_REASON_WRITING_FAILURES,
    DEGRADED_REASON_UNVERIFIED_EVALUATION,
    STATUS_BLOCKED,
    STATUS_DEGRADED,
    STATUS_OK,
    STATUS_PARTIAL,
)
from core.generation.agents.generated_question import GeneratedQuestion


# -----------------------------------------------------------------------------
# Test Helpers & Fixtures
# -----------------------------------------------------------------------------


@dataclass
class DummyPlan:
    slot_id: Any


def make_clean_question(
    slot_id: str,
    module_id: str = "module_1",
    global_q_idx: int = 1,
    question_text: str = "Explain the architecture of a transformer model.",
    solution: str = "The transformer model consists of an encoder and a decoder stack...",
) -> GeneratedQuestion:
    return GeneratedQuestion(
        slot_id=slot_id,
        module_id=module_id,
        global_q_idx=global_q_idx,
        question_text=question_text,
        solution=solution,
        marking_scheme=[{"points": 10, "criterion": "Complete and accurate explanation"}],
        marks=10,
        bloom="L2",
        co="CO1",
        is_unresolved=False,
        unresolved_scope=None,
        unresolved_subparts=(),
        failure_code=None,
        failure_detail=None,
    )


def make_10_plans_and_questions():
    """Generates standard 5-module x 2-question plans and clean questions."""
    plans = []
    questions = []
    for m in range(1, 6):
        for q in range(1, 3):
            g_idx = (m - 1) * 2 + q
            sid = f"module_{m}_Q{g_idx}"
            plans.append(DummyPlan(slot_id=sid))
            questions.append(
                make_clean_question(
                    slot_id=sid,
                    module_id=f"module_{m}",
                    global_q_idx=g_idx,
                )
            )
    spec = PaperSpec(exam_type="IAT1", module_count=5, questions_per_module=2, marks_per_question=10)
    return spec, plans, questions


# -----------------------------------------------------------------------------
# Tests 1 - 4: Resolvability Function
# -----------------------------------------------------------------------------


def test_checking_agent_is_resolvable_rejects_unresolved_flag():
    agent = CheckingAgent()
    q = make_clean_question(slot_id="module_1_Q1")
    q.is_unresolved = True
    assert agent._is_resolvable(q) is False


def test_checking_agent_is_resolvable_rejects_unresolved_subparts():
    agent = CheckingAgent()
    q = make_clean_question(slot_id="module_1_Q1")
    q.unresolved_subparts = ("b",)
    assert agent._is_resolvable(q) is False


def test_checking_agent_is_resolvable_rejects_unresolved_scope_alone():
    agent = CheckingAgent()
    q = make_clean_question(slot_id="module_1_Q1")
    q.unresolved_scope = "question"
    assert agent._is_resolvable(q) is False


def test_checking_agent_is_resolvable_requires_text_and_solution():
    agent = CheckingAgent()

    # None question
    assert agent._is_resolvable(None) is False

    # Empty / whitespace text
    q1 = make_clean_question(slot_id="module_1_Q1", question_text="")
    assert agent._is_resolvable(q1) is False
    q2 = make_clean_question(slot_id="module_1_Q1", question_text="   ")
    assert agent._is_resolvable(q2) is False

    # Empty / whitespace solution
    q3 = make_clean_question(slot_id="module_1_Q1", solution="")
    assert agent._is_resolvable(q3) is False
    q4 = make_clean_question(slot_id="module_1_Q1", solution="   ")
    assert agent._is_resolvable(q4) is False

    # Non-string text or solution
    q5 = make_clean_question(slot_id="module_1_Q1")
    q5.question_text = None
    assert agent._is_resolvable(q5) is False

    # Clean question
    clean_q = make_clean_question(slot_id="module_1_Q1")
    assert agent._is_resolvable(clean_q) is True


# -----------------------------------------------------------------------------
# Tests 5 - 8: Plan Validation
# -----------------------------------------------------------------------------


def test_checking_agent_plans_rejects_duplicates():
    agent = CheckingAgent()
    plans = [DummyPlan("module_1_Q1"), DummyPlan("module_1_Q1")]
    ctx = AgentContext(plans=plans, paper_spec=None)
    with pytest.raises(ValueError, match="duplicate expected slot_id"):
        agent._get_validated_expected_slots(ctx)


def test_checking_agent_invalid_expected_slot_id_fails_validation():
    agent = CheckingAgent()

    # slot_id is None
    ctx1 = AgentContext(plans=[DummyPlan(None)], paper_spec=None)
    with pytest.raises(ValueError, match="has slot_id=None"):
        agent._get_validated_expected_slots(ctx1)

    # slot_id is empty
    ctx2 = AgentContext(plans=[DummyPlan("")], paper_spec=None)
    with pytest.raises(ValueError, match="has empty slot_id"):
        agent._get_validated_expected_slots(ctx2)

    # slot_id is whitespace
    ctx3 = AgentContext(plans=[DummyPlan("   ")], paper_spec=None)
    with pytest.raises(ValueError, match="has empty slot_id"):
        agent._get_validated_expected_slots(ctx3)

    # slot_id is not a string
    ctx4 = AgentContext(plans=[DummyPlan(123)], paper_spec=None)
    with pytest.raises(ValueError, match="slot_id is not a string"):
        agent._get_validated_expected_slots(ctx4)


def test_checking_agent_empty_authoritative_plans_fails_closed():
    agent = CheckingAgent()
    spec = PaperSpec(exam_type="IAT1", module_count=5, questions_per_module=2, marks_per_question=10)
    ctx = AgentContext(plans=[], paper_spec=spec)
    with pytest.raises(ValueError, match="EMPTY_AUTHORITATIVE_PLAN"):
        agent._get_validated_expected_slots(ctx)


def test_checking_agent_plan_count_mismatch_fails_closed():
    agent = CheckingAgent()
    spec = PaperSpec(exam_type="IAT1", module_count=5, questions_per_module=2, marks_per_question=10)  # expects 10 slots
    plans = [DummyPlan(f"module_1_Q{i}") for i in range(1, 6)]  # only 5 slots
    ctx = AgentContext(plans=plans, paper_spec=spec)
    with pytest.raises(ValueError, match="Plan count mismatch"):
        agent._get_validated_expected_slots(ctx)


# -----------------------------------------------------------------------------
# Tests 9 - 12: Partition Invariants & Malformed Slot IDs
# -----------------------------------------------------------------------------


def test_checking_agent_malformed_observed_slot_id_controlled_defect():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()
    # Unhashable list as slot_id
    questions[0].slot_id = []

    ctx = AgentContext(plans=plans, paper_spec=spec, questions=questions, unresolved_slots=[])
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any("MALFORMED_OBSERVED_SLOT_ID:question[0]:type_list" in r for r in res.payload.blocked_reasons)


def test_checking_agent_unexpected_slot_id_cannot_produce_status_ok():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()
    # Extra unexpected question
    extra_q = make_clean_question(slot_id="unexpected_module_X_Q99")
    questions.append(extra_q)

    ctx = AgentContext(plans=plans, paper_spec=spec, questions=questions, unresolved_slots=[])
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any("UNEXPECTED_SLOT_ID:unexpected_module_X_Q99" in r for r in res.payload.blocked_reasons)


def test_checking_agent_unexpected_unresolved_id_prevents_status_ok():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["unexpected_external_slot_Q99"],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any("UNEXPECTED_UNRESOLVED_REGISTRY_ID:unexpected_external_slot_Q99" in r for r in res.payload.blocked_reasons)


def test_checking_agent_strict_partition_invariants():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Slot 9: partially resolved (subquestion scope with subpart)
    questions[8].is_unresolved = True
    questions[8].unresolved_scope = "subquestion"
    questions[8].unresolved_subparts = ("b",)
    questions[8].failure_code = "BLOOM_DEPTH_MISMATCH"
    questions[8].failure_detail = "Subpart (b) Bloom mismatch"

    # Slot 10: blocked (question scope)
    questions[9].is_unresolved = True
    questions[9].unresolved_scope = "question"
    questions[9].unresolved_subparts = ()
    questions[9].failure_code = "TOPIC_OUT_OF_SCOPE"
    questions[9].failure_detail = "Out of syllabus"

    unresolved_registry = ["module_5_Q9", "module_5_Q10"]
    ctx = AgentContext(plans=plans, paper_spec=spec, questions=questions, unresolved_slots=unresolved_registry)

    (
        sorted_f,
        sorted_p,
        sorted_b,
        sorted_unexp_obs,
        sorted_unexp_reg,
        fatal_integrity_defects,
        all_accounting_defects,
        telemetry,
    ) = agent._account_slots(ctx)

    expected_ids = [p.slot_id for p in plans]
    expected_set = set(expected_ids)

    # Union must equal expected set E
    assert (set(sorted_f) | set(sorted_p) | set(sorted_b)) == expected_set
    # Pairwise disjoint
    assert len(set(sorted_f) & set(sorted_p)) == 0
    assert len(set(sorted_f) & set(sorted_b)) == 0
    assert len(set(sorted_p) & set(sorted_b)) == 0

    assert len(sorted_f) == 8
    assert len(sorted_p) == 1
    assert len(sorted_b) == 1
    assert sorted_p == ["module_5_Q9"]
    assert sorted_b == ["module_5_Q10"]
    assert len(fatal_integrity_defects) == 0


# -----------------------------------------------------------------------------
# Tests 13 - 15: Bidirectional Registry Reconciliation
# -----------------------------------------------------------------------------


def test_checking_agent_unresolved_id_without_question_is_blocked():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Remove question for module_1_Q2, but register in unresolved_slots
    missing_sid = "module_1_Q2"
    questions = [q for q in questions if q.slot_id != missing_sid]

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=[missing_sid],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any(f"UNRESOLVED_SLOT_WITHOUT_QUESTION:{missing_sid}" in r for r in res.payload.blocked_reasons)


def test_checking_agent_unresolved_registry_contradiction_clean_question():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Question is clean, but registered in unresolved_slots
    contradicted_sid = "module_1_Q1"
    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=[contradicted_sid],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any(
        f"UNRESOLVED_REGISTRY_CONTRADICTION:{contradicted_sid}:registry_unresolved_but_question_clean" in r
        for r in res.payload.blocked_reasons
    )


def test_checking_agent_unresolved_question_absent_from_registry_reported():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Question claims unresolved, but registry is empty
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].unresolved_subparts = ()
    questions[0].failure_code = "BLOOM_MISMATCH"
    questions[0].failure_detail = "L1 instead of L3"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=[],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any(
        "UNRESOLVED_REGISTRY_CONTRADICTION:module_1_Q1:question_unresolved_but_absent_from_registry" in r
        for r in res.payload.blocked_reasons
    )


# -----------------------------------------------------------------------------
# Tests 16 - 18: Scope & Registry Handling
# -----------------------------------------------------------------------------


def test_checking_agent_missing_unresolved_scope_is_blocked():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    questions[0].is_unresolved = True
    questions[0].unresolved_scope = None  # Missing scope
    questions[0].unresolved_subparts = ()
    questions[0].failure_code = "BLOOM_MISMATCH"
    questions[0].failure_detail = "Missing scope"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any("INCONSISTENT_UNRESOLVED_METADATA:module_1_Q1:unresolved_true_with_missing_scope" in r for r in res.payload.blocked_reasons)


def test_checking_agent_subquestion_scope_without_subparts_is_blocked():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = ()  # Empty subparts for subquestion scope
    questions[0].failure_code = "BLOOM_MISMATCH"
    questions[0].failure_detail = "Subquestion scope without subparts"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any("INCONSISTENT_UNRESOLVED_METADATA:module_1_Q1:subquestion_scope_without_subparts" in r for r in res.payload.blocked_reasons)


def test_checking_agent_duplicate_unresolved_registry_entry_deterministic():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].unresolved_subparts = ()
    questions[0].failure_code = "BLOOM_MISMATCH"
    questions[0].failure_detail = "Deterministic deduplication test"

    # Duplicate in registry
    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1", "module_1_Q1"],
    )
    (
        sorted_f,
        sorted_p,
        sorted_b,
        sorted_unexp_obs,
        sorted_unexp_reg,
        fatal_defects,
        all_defects,
        telemetry,
    ) = agent._account_slots(ctx)

    assert len(fatal_defects) == 0
    assert sorted_b == ["module_1_Q1"]


# -----------------------------------------------------------------------------
# Tests 19 - 21: Status Preservation & Thresholds
# -----------------------------------------------------------------------------


def test_checking_agent_degraded_evidence_source_preserves_status():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    mock_audit = type("Audit", (), {"degraded_mode": True, "verdicts": []})()
    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=[],
        audit_report=mock_audit,
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_DEGRADED
    assert DEGRADED_REASON_UNVERIFIED_EVALUATION in res.payload.degraded_reasons


def test_checking_agent_preserves_existing_resolved_ratio_threshold():
    agent = CheckingAgent(min_questions_ratio=0.9)
    spec, plans, questions = make_10_plans_and_questions()

    # 2 out of 10 blocked -> 8/10 = 0.8 < 0.9 min_ratio
    for i in (0, 1):
        questions[i].is_unresolved = True
        questions[i].unresolved_scope = "question"
        questions[i].unresolved_subparts = ()
        questions[i].failure_code = "BLOOM_MISMATCH"
        questions[i].failure_detail = "Failed question"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1", "module_1_Q2"],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert BLOCK_REASON_INSUFFICIENT_QUESTIONS in res.payload.blocked_reasons


def test_checking_agent_preserves_fatal_writing_failures():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=[],
        telemetry={"writing_failures": [{"slot_id": "module_1_Q1", "error": "timeout"}]},
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert BLOCK_REASON_WRITING_FAILURES in res.payload.blocked_reasons


# -----------------------------------------------------------------------------
# Tests 22 - 26: Contract Preservation & Precedence
# -----------------------------------------------------------------------------


def test_checking_agent_ordinary_blocked_slots_with_resolved_slots_preserves_partial():
    agent = CheckingAgent(min_questions_ratio=0.9)
    spec, plans, questions = make_10_plans_and_questions()

    # 9 resolved, 1 ordinary blocked slot (consistent with registry)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].unresolved_subparts = ()
    questions[0].failure_code = "TOPIC_OUT_OF_SCOPE"
    questions[0].failure_detail = "Topic not in syllabus"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    res = agent.run(ctx)

    # 9/10 = 0.9 ratio, 0 fatal defects -> STATUS_PARTIAL preserved
    assert res.payload.status == STATUS_PARTIAL
    assert res.payload.resolved_slots == 9
    assert res.payload.unresolved_slots == ["module_1_Q1"]


def test_checking_agent_zero_fully_resolved_slots_produces_blocked():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    unres_ids = []
    for q in questions:
        q.is_unresolved = True
        q.unresolved_scope = "question"
        q.unresolved_subparts = ()
        q.failure_code = "TOPIC_OUT_OF_SCOPE"
        q.failure_detail = "Nothing resolved"
        unres_ids.append(q.slot_id)

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=unres_ids,
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert BLOCK_REASON_UNRESOLVED_SLOTS in res.payload.blocked_reasons


def test_checking_agent_unresolved_question_absent_from_registry_is_classified_as_blocked():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Subquestion failure marked on question, but missing from registry
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = ("b",)
    questions[0].failure_code = "BLOOM_MISMATCH"
    questions[0].failure_detail = "Subpart b failure"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=[],  # Absent from registry!
    )
    (
        sorted_f,
        sorted_p,
        sorted_b,
        sorted_unexp_obs,
        sorted_unexp_reg,
        fatal_defects,
        all_defects,
        telemetry,
    ) = agent._account_slots(ctx)

    # Must be classified as blocked B (not P) because of registry contradiction
    assert "module_1_Q1" in sorted_b
    assert "module_1_Q1" not in sorted_p
    assert any("question_unresolved_but_absent_from_registry" in d for d in fatal_defects)

    res = agent.run(ctx)
    assert res.payload.status == STATUS_BLOCKED


def test_checking_agent_malformed_unresolved_subparts_handled_without_exception():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Non-string element in subparts
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = [123, None]  # malformed elements
    questions[0].failure_code = "BLOOM_MISMATCH"
    questions[0].failure_detail = "Malformed subparts"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    (
        sorted_f,
        sorted_p,
        sorted_b,
        sorted_unexp_obs,
        sorted_unexp_reg,
        fatal_defects,
        all_defects,
        telemetry,
    ) = agent._account_slots(ctx)

    assert "module_1_Q1" in sorted_b
    assert any("MALFORMED_UNRESOLVED_SUBPARTS:module_1_Q1:invalid_element" in d for d in fatal_defects)


def test_checking_agent_ordinary_blocked_with_fatal_defect_takes_precedence():
    agent = CheckingAgent(min_questions_ratio=0.9)
    spec, plans, questions = make_10_plans_and_questions()

    # 1 ordinary blocked slot (consistent with registry)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].unresolved_subparts = ()
    questions[0].failure_code = "TOPIC_OUT_OF_SCOPE"
    questions[0].failure_detail = "Topic not in syllabus"

    # Plus unexpected question
    extra_q = make_clean_question(slot_id="extra_unexpected_Q99")
    questions.append(extra_q)

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    res = agent.run(ctx)

    # Fatal defect takes strict precedence over STATUS_PARTIAL
    assert res.payload.status == STATUS_BLOCKED
    assert any("UNEXPECTED_SLOT_ID:extra_unexpected_Q99" in r for r in res.payload.blocked_reasons)


# -----------------------------------------------------------------------------
# Tests 27 - 29: Dimension Type-Safety, None Subparts, Question Scope with Subparts
# -----------------------------------------------------------------------------


def test_checking_agent_unresolved_subparts_none_classified_as_malformed_defect():
    agent = CheckingAgent()
    q = make_clean_question(slot_id="module_1_Q1")
    q.unresolved_subparts = None

    # Resolvability rejects None
    assert agent._is_resolvable(q) is False

    spec, plans, questions = make_10_plans_and_questions()
    questions[0].unresolved_subparts = None

    ctx = AgentContext(plans=plans, paper_spec=spec, questions=questions, unresolved_slots=[])
    (
        sorted_f,
        sorted_p,
        sorted_b,
        sorted_unexp_obs,
        sorted_unexp_reg,
        fatal_defects,
        all_defects,
        telemetry,
    ) = agent._account_slots(ctx)

    assert "module_1_Q1" in sorted_b
    assert any("MALFORMED_UNRESOLVED_SUBPARTS:module_1_Q1:invalid_type_NoneType" in d for d in fatal_defects)

    res = agent.run(ctx)
    assert res.payload.status == STATUS_BLOCKED


def test_checking_agent_paperspec_invalid_dimensions_fail_closed():
    agent = CheckingAgent()

    # Test helper _is_positive_int directly
    assert _is_positive_int(1) is True
    assert _is_positive_int(5) is True
    assert _is_positive_int(0) is False
    assert _is_positive_int(-1) is False
    assert _is_positive_int(1.5) is False
    assert _is_positive_int("5") is False
    assert _is_positive_int(None) is False
    assert _is_positive_int(True) is False   # Crucial: rejects boolean True
    assert _is_positive_int(False) is False  # Crucial: rejects boolean False

    invalid_dimensions = [0, -1, "5", None, 5.5, True, False]
    for inv in invalid_dimensions:
        spec = type("BadSpec", (), {"module_count": inv, "questions_per_module": 2})()
        ctx = AgentContext(plans=None, paper_spec=spec)
        with pytest.raises(ValueError, match="Invalid PaperSpec dimensions"):
            agent._get_validated_expected_slots(ctx)

        spec2 = type("BadSpec", (), {"module_count": 5, "questions_per_module": inv})()
        ctx2 = AgentContext(plans=None, paper_spec=spec2)
        with pytest.raises(ValueError, match="Invalid PaperSpec dimensions"):
            agent._get_validated_expected_slots(ctx2)


def test_checking_agent_question_scope_with_subparts_is_fatal_defect():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Contradiction: question-level scope with non-empty subparts
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].unresolved_subparts = ("b",)
    questions[0].failure_code = "BLOOM_MISMATCH"
    questions[0].failure_detail = "Contradictory question scope with subparts"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any("INCONSISTENT_UNRESOLVED_METADATA:module_1_Q1:question_scope_with_nonempty_subparts_b" in r for r in res.payload.blocked_reasons)


# -----------------------------------------------------------------------------
# Tests 30 - 32: Cross-Field Consistency
# -----------------------------------------------------------------------------


def test_checking_agent_is_unresolved_false_with_subquestion_markers_fails_closed():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Question claims clean (is_unresolved=False) but has active subquestion markers
    questions[0].is_unresolved = False
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = ("b",)
    questions[0].failure_code = "BLOOM_DEPTH_MISMATCH"
    questions[0].failure_detail = None

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any("INCONSISTENT_UNRESOLVED_METADATA:module_1_Q1:is_unresolved_false_with_active_markers" in r for r in res.payload.blocked_reasons)


def test_checking_agent_is_unresolved_false_with_active_failure_code_fails_closed():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Question claims clean (is_unresolved=False) but has active failure_code
    questions[0].is_unresolved = False
    questions[0].failure_code = "BLOOM_MISMATCH"
    questions[0].failure_detail = None
    questions[0].unresolved_scope = None
    questions[0].unresolved_subparts = ()

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=[],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any("INCONSISTENT_UNRESOLVED_METADATA:module_1_Q1:is_unresolved_false_with_active_markers" in r for r in res.payload.blocked_reasons)


def test_checking_agent_is_unresolved_true_with_missing_failure_code_fails_closed():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Question is unresolved, but failure_code is None
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].unresolved_subparts = ()
    questions[0].failure_code = None
    questions[0].failure_detail = "Some detail"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any("INCONSISTENT_UNRESOLVED_METADATA:module_1_Q1:unresolved_true_with_missing_failure_code" in r for r in res.payload.blocked_reasons)


# -----------------------------------------------------------------------------
# Tests 33 - 35: Detail Consistency & Empty Content Ordering
# -----------------------------------------------------------------------------


def test_checking_agent_resolved_question_with_failure_detail_fails_closed():
    agent = CheckingAgent()
    q = make_clean_question(slot_id="module_1_Q1")
    q.failure_detail = "repaired detail"

    # Resolvability rejects non-None detail on resolved question
    assert agent._is_resolvable(q) is False

    spec, plans, questions = make_10_plans_and_questions()
    questions[0].failure_detail = "repaired detail"

    ctx = AgentContext(plans=plans, paper_spec=spec, questions=questions, unresolved_slots=[])
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any("INCONSISTENT_UNRESOLVED_METADATA:module_1_Q1:is_unresolved_false_with_active_markers" in r for r in res.payload.blocked_reasons)


def test_checking_agent_unresolved_question_with_missing_failure_detail_fails_closed():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Question is unresolved, but failure_detail is None
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].unresolved_subparts = ()
    questions[0].failure_code = "BLOOM_MISMATCH"
    questions[0].failure_detail = None

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    res = agent.run(ctx)

    assert res.payload.status == STATUS_BLOCKED
    assert any("INCONSISTENT_UNRESOLVED_METADATA:module_1_Q1:unresolved_true_with_missing_failure_detail" in r for r in res.payload.blocked_reasons)


def test_checking_agent_empty_content_with_contradictory_metadata_retains_both_defects_and_blocks():
    agent = CheckingAgent(min_questions_ratio=0.9)
    spec, plans, questions = make_10_plans_and_questions()

    # Slot 1: empty text/solution AND contradictory metadata
    questions[0].question_text = ""
    questions[0].solution = ""
    questions[0].is_unresolved = False
    questions[0].failure_code = "BLOOM_MISMATCH"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=[],
    )
    (
        sorted_f,
        sorted_p,
        sorted_b,
        sorted_unexp_obs,
        sorted_unexp_reg,
        fatal_defects,
        all_defects,
        telemetry,
    ) = agent._account_slots(ctx)

    # Both defects must be retained
    assert "EMPTY_CONTENT:module_1_Q1" in all_defects
    assert any("INCONSISTENT_UNRESOLVED_METADATA:module_1_Q1" in d for d in all_defects)
    assert any("INCONSISTENT_UNRESOLVED_METADATA:module_1_Q1" in d for d in fatal_defects)

    res = agent.run(ctx)
    # Contradictory metadata blocks the paper even though 9/10 slots are clean
    assert res.payload.status == STATUS_BLOCKED


# -----------------------------------------------------------------------------
# Tests 36 - 39: Positive Controls
# -----------------------------------------------------------------------------


def test_checking_agent_positive_control_valid_question_level_unresolved_in_b_without_contradiction():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Valid Stage 2A question-level failure
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].unresolved_subparts = ()
    questions[0].failure_code = "TOPIC_OUT_OF_SCOPE"
    questions[0].failure_detail = "Topic not covered in syllabus"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    (
        sorted_f,
        sorted_p,
        sorted_b,
        sorted_unexp_obs,
        sorted_unexp_reg,
        fatal_defects,
        all_defects,
        telemetry,
    ) = agent._account_slots(ctx)

    assert len(fatal_defects) == 0
    assert sorted_b == ["module_1_Q1"]
    assert "QUESTION_UNRESOLVED:module_1_Q1:TOPIC_OUT_OF_SCOPE" in all_defects


def test_checking_agent_positive_control_valid_subquestion_level_unresolved_in_p_with_subparts():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Valid Stage 2A subquestion failure
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = ("b",)
    questions[0].failure_code = "BLOOM_DEPTH_MISMATCH"
    questions[0].failure_detail = "Subpart (b) Bloom level L1 instead of L3"

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    (
        sorted_f,
        sorted_p,
        sorted_b,
        sorted_unexp_obs,
        sorted_unexp_reg,
        fatal_defects,
        all_defects,
        telemetry,
    ) = agent._account_slots(ctx)

    assert len(fatal_defects) == 0
    assert sorted_p == ["module_1_Q1"]
    assert telemetry["subquestion_defects"]["module_1_Q1"] == ("b",)
    assert "SUBPART_UNRESOLVED:module_1_Q1:b" in all_defects


def test_checking_agent_positive_control_repaired_question_returns_to_f():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Clean repaired question
    repaired_q = make_clean_question(slot_id="module_1_Q1")
    repaired_q.is_unresolved = False
    repaired_q.unresolved_scope = None
    repaired_q.unresolved_subparts = ()
    repaired_q.failure_code = None
    repaired_q.failure_detail = None

    assert agent._is_resolvable(repaired_q) is True

    questions[0] = repaired_q
    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=[],
    )
    (
        sorted_f,
        sorted_p,
        sorted_b,
        sorted_unexp_obs,
        sorted_unexp_reg,
        fatal_defects,
        all_defects,
        telemetry,
    ) = agent._account_slots(ctx)

    assert len(fatal_defects) == 0
    assert len(sorted_f) == 10
    assert len(sorted_p) == 0
    assert len(sorted_b) == 0

    res = agent.run(ctx)
    assert res.payload.status == STATUS_OK


def test_checking_agent_unresolved_failure_detail_validation():
    agent = CheckingAgent()
    spec, plans, questions = make_10_plans_and_questions()

    # Non-string failure_detail
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].unresolved_subparts = ()
    questions[0].failure_code = "BLOOM_MISMATCH"
    questions[0].failure_detail = 12345

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1"],
    )
    res = agent.run(ctx)
    assert res.payload.status == STATUS_BLOCKED
    assert any("MALFORMED_QUESTION_METADATA:module_1_Q1:invalid_failure_detail" in r for r in res.payload.blocked_reasons)

    # Empty string failure_detail
    questions[0].failure_detail = ""
    res2 = agent.run(ctx)
    assert res2.payload.status == STATUS_BLOCKED
    assert any("MALFORMED_QUESTION_METADATA:module_1_Q1:invalid_failure_detail" in r for r in res2.payload.blocked_reasons)

    # Whitespace-only failure_detail
    questions[0].failure_detail = "   "
    res3 = agent.run(ctx)
    assert res3.payload.status == STATUS_BLOCKED
    assert any("MALFORMED_QUESTION_METADATA:module_1_Q1:invalid_failure_detail" in r for r in res3.payload.blocked_reasons)


# -----------------------------------------------------------------------------
# Test 40: Missing Question With Registry Entry Fails Closed as Fatal Defect
# -----------------------------------------------------------------------------


def test_checking_agent_missing_question_with_registry_entry_is_fatal_defect_blocking_status():
    """
    Slot absent from context.questions with ID present in context.unresolved_slots
    enters fatal_integrity_defects as UNRESOLVED_SLOT_WITHOUT_QUESTION, fails closed into B,
    and produces STATUS_BLOCKED even if all 9 other slots are clean.
    """
    agent = CheckingAgent(min_questions_ratio=0.9)
    spec, plans, questions = make_10_plans_and_questions()

    # Remove slot module_1_Q1 from questions entirely, but record it in unresolved_slots
    missing_sid = "module_1_Q1"
    questions_9 = [q for q in questions if q.slot_id != missing_sid]

    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions_9,
        unresolved_slots=[missing_sid],
    )

    (
        sorted_f,
        sorted_p,
        sorted_b,
        sorted_unexp_obs,
        sorted_unexp_reg,
        fatal_defects,
        all_defects,
        telemetry,
    ) = agent._account_slots(ctx)

    # Must be in blocked B
    assert missing_sid in sorted_b
    assert missing_sid not in sorted_p
    assert missing_sid not in sorted_f

    # Diagnostic fatal defect recorded
    expected_defect = f"UNRESOLVED_SLOT_WITHOUT_QUESTION:{missing_sid}"
    assert expected_defect in fatal_defects
    assert expected_defect in all_defects

    # Run must be BLOCKED (not PARTIAL, not OK)
    res = agent.run(ctx)
    assert res.payload.status == STATUS_BLOCKED
    assert f"INTEGRITY_DEFECT:{expected_defect}" in res.payload.blocked_reasons


# -----------------------------------------------------------------------------
# Test 41: Degraded Audit Low Ratio Exception Preserves Partial Status
# -----------------------------------------------------------------------------


def test_checking_agent_degraded_audit_low_ratio_exception_preserves_partial():
    """
    When audit_report.degraded_mode is True and at least one slot is resolved,
    a resolved ratio below min_questions_ratio (e.g. 8/10 < 0.9) enters degraded_reasons
    rather than blocked_reasons. Combined with at least one unresolved slot,
    status becomes STATUS_PARTIAL (not STATUS_BLOCKED).
    """
    agent = CheckingAgent(min_questions_ratio=0.9)
    spec, plans, questions = make_10_plans_and_questions()

    # 2 unresolved questions out of 10: 8 resolved / 10 expected = 0.8 < 0.9
    for i in (0, 1):
        questions[i].is_unresolved = True
        questions[i].unresolved_scope = "question"
        questions[i].unresolved_subparts = ()
        questions[i].failure_code = "TOPIC_OUT_OF_SCOPE"
        questions[i].failure_detail = "Failed topic"

    mock_audit = type("Audit", (), {"degraded_mode": True, "verdicts": []})()
    ctx = AgentContext(
        plans=plans,
        paper_spec=spec,
        questions=questions,
        unresolved_slots=["module_1_Q1", "module_1_Q2"],
        audit_report=mock_audit,
    )

    res = agent.run(ctx)

    # In degraded audit mode, low ratio does NOT hard-block if resolved > 0
    assert BLOCK_REASON_INSUFFICIENT_QUESTIONS in res.payload.degraded_reasons
    assert BLOCK_REASON_INSUFFICIENT_QUESTIONS not in res.payload.blocked_reasons
    assert len(res.payload.blocked_reasons) == 0

    # With 2 unresolved slots and 8 resolved slots, status is STATUS_PARTIAL
    assert res.payload.status == STATUS_PARTIAL
    assert res.payload.resolved_slots == 8
    assert res.payload.unresolved_slots == ["module_1_Q1", "module_1_Q2"]

