"""
Stage G3 Acceptance Test Suite — Downstream Multi-Part Structural Alignment.

Verifies Gates G3-T01 through G3-T10 and user-mandated implementation constraints:
- G3-T01: Evaluator subpart correspondence (PASS, FAIL, INDETERMINATE, LEGACY).
- G3-T02: Multi-part Bloom verb evaluation (individual SubPart evaluation against SubPartPlan).
- G3-T03: Length budget enforcement (6-mark <= 600 chars, 8-mark <= 800 chars, outer marker stripping).
- G3-T04: Deterministic case normalization & redundant leading label stripping.
- G3-T05: Deterministic Bloom verb repair (SubPart isolation, multiword modifier guard, cognitive demand re-validation).
- G3-T06: LLM isolated subpart repair (bit-for-bit sibling preservation).
- G3-T07: LLM repair criteria-total consistency & marks conservation.
- G3-T08: Strict projection consistency verification (rejection of corrupted projections).
- G3-T09: Evaluation telemetry preservation (subpart_failures in audit report).
- G3-T10: Full backward compatibility for legacy single-part questions.
"""

from __future__ import annotations

import copy
import pytest
from typing import Any, Dict, List, Optional

from core.generation.agents.base import AgentContext
from core.generation.agents.deterministic_repairs import (
    repair_bloom_verb,
    repair_figure_reference,
    repair_marking_scheme,
    repair_marking_scheme_rescale,
    repair_multi_part_structure,
    repair_short_question,
    repair_sibling_independence,
    try_deterministic_repair,
)
from core.generation.agents.evaluation_agent import EvaluationAgent
from core.generation.agents.evaluation_contracts import (
    BLOOM_MISMATCH,
    MARKING_SCHEME_MISMATCH,
    MISSING_FIGURE_REFERENCE,
    MULTI_PART_STRUCTURE_MISMATCH,
    QUESTION_TOO_LONG,
    QUESTION_TOO_SHORT,
    SIBLING_INTERCONNECTED,
    UNVERIFIED,
    AuditReport,
    SlotVerdict,
)
from core.generation.agents.generated_question import GeneratedQuestion, SubPart
from core.generation.agents.local_evaluators import (
    CorrespondenceOutcome,
    bloom_verb_evaluator,
    multi_part_structure_evaluator,
    question_length_evaluator,
    resolve_subpart_plan_correspondence,
    run_local_evaluators,
)
from core.generation.agents.planning_agent import QuestionPlan, SubPartPlan
from core.generation.agents.refinement_agent import RefinementAgent
from core.generation.agents.refinement_contracts import TIER_DETERMINISTIC, TIER_LLM
from core.generation.agents.refinement_prompts import build_repair_prompt


# =============================================================================
# Helper Fixtures & Builders
# =============================================================================

def make_sample_multipart_question(
    slot_id: str = "slot_1",
    marks: int = 10,
    partition: Optional[List[int]] = None,
    sp_a_text: str = "Explain the architecture of the operating system kernel.",
    sp_b_text: str = "Analyze the process scheduling algorithms and compare preemptive and non-preemptive strategies.",
    sp_a_marks: int = 6,
    sp_b_marks: int = 4,
    sp_a_label: str = "a",
    sp_b_label: str = "b",
) -> GeneratedQuestion:
    if partition is None:
        partition = [sp_a_marks, sp_b_marks]

    sp_a = SubPart(
        label=sp_a_label,
        marks=sp_a_marks,
        text=sp_a_text,
        topic="Kernel Architecture",
        solution="Detailed kernel explanation.",
        marking_scheme=[
            {"criterion": "Architecture overview", "marks": sp_a_marks // 2},
            {"criterion": "Core mechanisms", "marks": sp_a_marks - (sp_a_marks // 2)},
        ],
    )
    sp_b = SubPart(
        label=sp_b_label,
        marks=sp_b_marks,
        text=sp_b_text,
        topic="Process Scheduling",
        solution="Detailed scheduling analysis.",
        marking_scheme=[
            {"criterion": "Algorithm trade-offs", "marks": sp_b_marks // 2},
            {"criterion": "Comparative trace", "marks": sp_b_marks - (sp_b_marks // 2)},
        ],
    )
    q = GeneratedQuestion(
        slot_id=slot_id,
        module_id="mod_1",
        global_q_idx=1,
        question_text=f"({sp_a_label}) {sp_a_text}\n\n({sp_b_label}) {sp_b_text}",
        solution=f"({sp_a_label}) {sp_a.solution}\n\n({sp_b_label}) {sp_b.solution}",
        marking_scheme=[],
        marks=marks,
        partition=partition,
        bloom="L2",
        co="CO1",
        topic="Operating Systems",
        sub_parts=[sp_a, sp_b],
    )
    q.sync_materialized_projections()
    return q


def make_sample_plan(
    slot_id: str = "slot_1",
    sp_a_bloom: str = "L2",
    sp_b_bloom: str = "L4",
    sp_a_marks: int = 6,
    sp_b_marks: int = 4,
) -> QuestionPlan:
    plan = QuestionPlan(
        slot_id=slot_id,
        module_id="mod_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[sp_a_marks, sp_b_marks],
        total_marks=sp_a_marks + sp_b_marks,
        co="CO3" if sp_b_bloom == "L4" else "CO1",
        bloom="L4" if sp_b_bloom == "L4" else "L2",
        topic="Operating Systems",
    )
    plan.sub_part_plans = [
        SubPartPlan(
            label="a",
            marks=sp_a_marks,
            topic="Kernel Architecture",
            bloom=sp_a_bloom,
            co="CO1" if sp_a_bloom in ("L1", "L2") else ("CO3" if sp_a_bloom == "L4" else "CO2"),
        ),
        SubPartPlan(
            label="b",
            marks=sp_b_marks,
            topic="Process Scheduling",
            bloom=sp_b_bloom,
            co="CO3" if sp_b_bloom == "L4" else ("CO1" if sp_b_bloom in ("L1", "L2") else "CO2"),
        ),
    ]
    return plan


# =============================================================================
# Gate G3-T01: Evaluator SubPart Correspondence
# =============================================================================

class TestG3T01SubPartCorrespondence:
    """Verify correspondence resolution under all standard and adversarial conditions."""

    def test_clean_correspondence(self):
        q = make_sample_multipart_question()
        plan = make_sample_plan()
        outcome, mapping, err = resolve_subpart_plan_correspondence(q, plan)
        assert outcome == CorrespondenceOutcome.PASS
        assert len(mapping) == 2
        assert mapping["a"][0].label == "a" and mapping["a"][1].bloom == "L2"
        assert mapping["b"][0].label == "b" and mapping["b"][1].bloom == "L4"
        assert err is None

    def test_duplicate_labels_in_subparts_fails_safely(self):
        q = make_sample_multipart_question()
        # Adversarially set duplicate labels in question
        q.sub_parts[1].label = "a"
        plan = make_sample_plan()
        outcome, mapping, err = resolve_subpart_plan_correspondence(q, plan)
        assert outcome == CorrespondenceOutcome.FAIL
        assert "duplicate" in err.detail.lower()
        assert mapping == {}

    def test_missing_plan_entries(self):
        q = make_sample_multipart_question()
        plan = make_sample_plan()
        # Plan only has subpart a, question has a and b
        plan.sub_part_plans = plan.sub_part_plans[:1]
        outcome, mapping, err = resolve_subpart_plan_correspondence(q, plan)
        assert outcome == CorrespondenceOutcome.FAIL
        assert "extra" in err.detail.lower()

    def test_extra_generated_subparts(self):
        q = make_sample_multipart_question()
        # Add an extra subpart c
        sp_c = SubPart(label="c", marks=2, text="List three OS types.", solution="Sol", marking_scheme=[{"criterion": "Types", "marks": 2}])
        q.sub_parts.append(sp_c)
        plan = make_sample_plan()
        outcome, mapping, err = resolve_subpart_plan_correspondence(q, plan)
        assert outcome == CorrespondenceOutcome.FAIL
        assert "extra" in err.detail.lower()

    def test_case_normalization_in_correspondence(self):
        q = make_sample_multipart_question()
        q.sub_parts[0].label = "A"
        q.sub_parts[1].label = "B"
        plan = make_sample_plan()
        outcome, mapping, err = resolve_subpart_plan_correspondence(q, plan)
        assert outcome == CorrespondenceOutcome.PASS
        assert len(mapping) == 2

    def test_mismatched_slot_id(self):
        q = make_sample_multipart_question(slot_id="slot_1")
        plan = make_sample_plan(slot_id="slot_2")
        outcome, mapping, err = resolve_subpart_plan_correspondence(q, plan)
        assert outcome == CorrespondenceOutcome.INDETERMINATE
        assert "slot_id correspondence ambiguous" in err.detail.lower()

    def test_legacy_fallback_when_plan_is_none(self):
        q = make_sample_multipart_question()
        outcome, mapping, err = resolve_subpart_plan_correspondence(q, None)
        assert outcome == CorrespondenceOutcome.LEGACY
        assert mapping == {}
        assert err is None


# =============================================================================
# Gate G3-T02: Multi-Part Bloom Verb Evaluation
# =============================================================================

class TestG3T02MultiPartBloomEvaluation:
    """Verify evaluation audits each SubPart against individual SubPartPlan.bloom."""

    def test_multipart_bloom_pass(self):
        q = make_sample_multipart_question(
            sp_a_text="Explain the kernel subsystems.", # L2
            sp_b_text="Analyze the turnaround times.",    # L4
        )
        plan = make_sample_plan(sp_a_bloom="L2", sp_b_bloom="L4")
        verdict = bloom_verb_evaluator(q, plan=plan)
        assert verdict is None

    def test_multipart_bloom_subpart_failure(self):
        q = make_sample_multipart_question(
            sp_a_text="Explain the kernel subsystems.", # L2 (valid for L2)
            sp_b_text="Describe the turnaround times.",  # L2 verb for L4 plan (invalid!)
        )
        plan = make_sample_plan(sp_a_bloom="L2", sp_b_bloom="L4")
        verdict = bloom_verb_evaluator(q, plan=plan)
        assert verdict is not None
        assert verdict.verdict == "fail"
        assert verdict.reason_code == BLOOM_MISMATCH
        assert "SubPart (b)" in verdict.detail
        assert "describe" in verdict.detail
        assert verdict.affected_subparts == ("b",)

    def test_indeterminate_when_correspondence_fails(self):
        q = make_sample_multipart_question(slot_id="slot_1")
        plan = make_sample_plan(slot_id="slot_2") # Slot ID mismatch -> INDETERMINATE
        verdict = bloom_verb_evaluator(q, plan=plan)
        assert verdict is not None
        assert verdict.verdict == "unverified"
        assert verdict.reason_code == UNVERIFIED
        assert "ambiguous" in verdict.detail.lower()


# =============================================================================
# Gate G3-T03: Length Budget Enforcement
# =============================================================================

class TestG3T03LengthBudgetEnforcement:
    """Verify explicit VTU length policy per subpart marks and composite limits."""

    def test_six_mark_length_limits(self):
        # 6 marks: max 600 chars
        valid_text = "Explain the scheduling criteria: " + ("x" * 400)
        q = make_sample_multipart_question(sp_a_text=valid_text, sp_a_marks=6)
        plan = make_sample_plan(sp_a_marks=6)
        verdict = question_length_evaluator(q, plan=plan)
        assert verdict is None

        # 6 marks: 700 chars -> FAIL
        invalid_text = "Explain the scheduling criteria: " + ("x" * 650)
        q_long = make_sample_multipart_question(sp_a_text=invalid_text, sp_a_marks=6)
        verdict_fail = question_length_evaluator(q_long, plan=plan)
        assert verdict_fail is not None
        assert verdict_fail.verdict == "fail"
        assert verdict_fail.reason_code == QUESTION_TOO_LONG
        assert "SubPart (a)" in verdict_fail.detail
        assert verdict_fail.affected_subparts == ("a",)

    def test_eight_mark_length_limits(self):
        # 8 marks: max 800 chars
        valid_8m_text = "Analyze the paging mechanism: " + ("y" * 650)
        q = make_sample_multipart_question(sp_a_text=valid_8m_text, sp_a_marks=8, sp_b_marks=6, marks=14)
        plan = make_sample_plan(sp_a_marks=8, sp_b_marks=6)
        verdict = question_length_evaluator(q, plan=plan)
        assert verdict is None

    def test_stripping_outer_markers_before_measurement(self):
        # 6 marks: limit 600 chars. Inner text is 590 chars (<= 600).
        # With markers '(a) ' and ' [6 Marks]', raw length is 604 chars (> 600).
        inner = "Describe memory management: " + ("z" * 562)
        assert len(inner) == 590
        with_markers = f"(a) {inner} [6 Marks]"
        assert len(with_markers) == 604
        q = make_sample_multipart_question(sp_a_text=with_markers, sp_a_marks=6)
        plan = make_sample_plan(sp_a_marks=6)
        verdict = question_length_evaluator(q, plan=plan)
        assert verdict is None


# =============================================================================
# Gate G3-T04: Deterministic Case Normalization
# =============================================================================

class TestG3T04DeterministicCaseNormalization:
    """Verify repair_multi_part_structure normalizes uppercase labels and strips redundant markers."""

    def test_case_normalization_and_marker_stripping(self):
        q = make_sample_multipart_question()
        # Give uppercase labels and redundant markers in text
        q.sub_parts[0].label = "A"
        q.sub_parts[0].text = "(A) Explain the architecture of the operating system kernel."
        q.sub_parts[1].label = "B"
        q.sub_parts[1].text = "B. Analyze the process scheduling algorithms."

        repaired = repair_multi_part_structure(q, MULTI_PART_STRUCTURE_MISMATCH, "Uppercase labels")
        assert repaired is not None
        assert repaired.sub_parts[0].label == "a"
        assert repaired.sub_parts[1].label == "b"
        assert not repaired.sub_parts[0].text.startswith("(A)")
        assert not repaired.sub_parts[0].text.startswith("(a)")
        assert repaired.sub_parts[0].text.startswith("Explain")
        assert not repaired.sub_parts[1].text.startswith("B.")
        assert repaired.sub_parts[1].text.startswith("Analyze")

        # Invariant check
        consistent, err = repaired.verify_projection_consistency()
        assert consistent and err == ""
        assert repaired.question_text == f"(a) {repaired.sub_parts[0].text}\n\n(b) {repaired.sub_parts[1].text}"

    def test_duplicate_labels_rejected(self):
        q = make_sample_multipart_question()
        q.sub_parts[1].label = "a"
        repaired = repair_multi_part_structure(q, MULTI_PART_STRUCTURE_MISMATCH, "Duplicate")
        assert repaired is None

    def test_missing_subparts_rejected(self):
        q = make_sample_multipart_question()
        q.partition = [6, 4]
        q.sub_parts = q.sub_parts[:1] # Missing part b
        repaired = repair_multi_part_structure(q, MULTI_PART_STRUCTURE_MISMATCH, "Missing part")
        assert repaired is None


# =============================================================================
# Gate G3-T05: Deterministic Bloom Verb Repair
# =============================================================================

class TestG3T05DeterministicBloomRepair:
    """Verify deterministic Bloom verb repair operates on SubPart while preserving siblings."""

    def test_subpart_bloom_repair(self):
        q = make_sample_multipart_question(
            sp_a_text="Explain the operating system kernel structure in detail.",
            sp_b_text="Describe the scheduling algorithms and compare throughput.", # Needs L4 'Analyze'
        )
        detail = "SubPart (b) verb 'describe' not allowed for Bloom level L4 (allowed: ['analyze', 'examine'])"
        repaired = repair_bloom_verb(q, BLOOM_MISMATCH, detail)

        assert repaired is not None
        # Subpart a must be preserved bit-for-bit
        assert repaired.sub_parts[0].text == q.sub_parts[0].text
        assert repaired.sub_parts[0].marking_scheme == q.sub_parts[0].marking_scheme

        # Subpart b verb replaced with canonical L4 verb
        assert repaired.sub_parts[1].text.lower().startswith(("analyze", "analyse"))
        assert "scheduling algorithms" in repaired.sub_parts[1].text

        # Invariant check
        consistent, err = repaired.verify_projection_consistency()
        assert consistent and err == ""

    def test_multiword_modifier_guard(self):
        # "Illustrate with a neat sketch..." should not be replaced with "Analyze with a neat sketch"
        q = make_sample_multipart_question(
            sp_b_text="Illustrate with a neat sketch the process control block layout.",
        )
        detail = "SubPart (b) verb 'illustrate' not allowed for Bloom level L4"
        repaired = repair_bloom_verb(q, BLOOM_MISMATCH, detail)
        assert repaired is None # Safe fail-closed behavior


# =============================================================================
# Gate G3-T06: LLM Isolated SubPart Repair
# =============================================================================

class MockLLMCaller:
    def __init__(self, return_payload: Any):
        self.payload = return_payload
        self.calls = []

    def call(self, prompt: str, schema: Optional[dict] = None, image_path: Optional[str] = None, seed: Optional[int] = None) -> Any:
        self.calls.append(prompt)
        return self.payload


class TestG3T06LLMIsolatedSubPartRepair:
    """Verify LLM repair replaces only the targeted subpart and preserves siblings bit-for-bit."""

    def test_llm_isolated_repair_preserves_sibling(self):
        q = make_sample_multipart_question()
        orig_sp_a = copy.deepcopy(q.sub_parts[0])

        verdict = SlotVerdict(
            slot_id=q.slot_id,
            verdict="fail",
            source="local",
            reason_code=BLOOM_MISMATCH,
            detail="SubPart (b) verb 'describe' not allowed for Bloom level L4",
            affected_subparts=("b",),
        )

        mock_response = {
            "label": "b",
            "text": "Analyze the process scheduling algorithms and compare CPU utilization under RR and SJF.",
            "solution": "Comprehensive model answer for subpart (b).",
            "marking_scheme": [
                {"criterion": "Analytical comparison of algorithms", "marks": 2},
                {"criterion": "CPU utilization analysis under RR vs SJF", "marks": 2},
            ],
            "change_summary": "Elevated cognitive demand to L4 Analyze",
        }
        mock_llm = MockLLMCaller(mock_response)
        agent = RefinementAgent(llm_caller=mock_llm)

        context = AgentContext(questions=[q], audit_report=AuditReport(verdicts=[verdict]))
        result = agent.run(context)

        assert result.success
        assert len(context.questions) == 1
        repaired_q = context.questions[0]

        # Sibling SubPart (a) must be bit-for-bit identical to original
        assert repaired_q.sub_parts[0].label == orig_sp_a.label
        assert repaired_q.sub_parts[0].text == orig_sp_a.text
        assert repaired_q.sub_parts[0].solution == orig_sp_a.solution
        assert repaired_q.sub_parts[0].marks == orig_sp_a.marks
        assert repaired_q.sub_parts[0].marking_scheme == orig_sp_a.marking_scheme

        # Target SubPart (b) must be updated
        assert repaired_q.sub_parts[1].text == mock_response["text"]
        assert repaired_q.sub_parts[1].solution == mock_response["solution"]
        assert repaired_q.sub_parts[1].marks == 4

        # Invariant check: materialized projections must match authoritative subparts
        consistent, err = repaired_q.verify_projection_consistency()
        assert consistent and err == ""
        assert repaired_q.question_text == f"({orig_sp_a.label}) {orig_sp_a.text}\n\n(b) {mock_response['text']}"


# =============================================================================
# Gate G3-T07: Criteria-Total Consistency & Marks Conservation
# =============================================================================

class TestG3T07CriteriaTotalConsistency:
    """Verify criteria sum strictly equals subpart marks; accepts numerical normalization, rejects drift."""

    def test_criteria_numerical_normalization_accepted(self):
        q = make_sample_multipart_question()
        verdict = SlotVerdict(
            slot_id=q.slot_id,
            verdict="fail",
            source="local",
            reason_code=BLOOM_MISMATCH,
            detail="SubPart (b) verb issue",
            affected_subparts=("b",),
        )
        # LLM returns valid criteria marks represented as float and string integer representations
        mock_response = {
            "label": "b",
            "text": "Analyze process scheduling algorithms in detail.",
            "solution": "Model solution for part b.",
            "marking_scheme": [
                {"criterion": "Trade-off analysis", "marks": 2.0},
                {"criterion": "Comparative trace", "marks": "2"},
            ],
            "change_summary": "Fixed verb with normalized numbers",
        }
        mock_llm = MockLLMCaller(mock_response)
        agent = RefinementAgent(llm_caller=mock_llm)

        context = AgentContext(questions=[q], audit_report=AuditReport(verdicts=[verdict]))
        result = agent.run(context)

        assert result.success
        assert len(result.payload.unresolved_slots) == 0
        repaired_q = context.questions[0]
        # Sum must strictly equal 4, and each mark must be normalized to int
        assert sum(c["marks"] for c in repaired_q.sub_parts[1].marking_scheme) == 4
        for c in repaired_q.sub_parts[1].marking_scheme:
            assert type(c["marks"]) is int
        consistent, err = repaired_q.verify_projection_consistency()
        assert consistent and err == ""

    def test_criteria_substantive_mark_corruption_rejected(self):
        # sp_b starts with multiword modifier so deterministic repair fails closed and LLM repair is exercised
        q = make_sample_multipart_question(
            sp_b_text="Illustrate with a neat sketch the process scheduling algorithms in detail.",
        )
        orig_text = q.sub_parts[1].text
        verdict = SlotVerdict(
            slot_id=q.slot_id,
            verdict="fail",
            source="local",
            reason_code=BLOOM_MISMATCH,
            detail="SubPart (b) verb issue for Bloom level L4",
            affected_subparts=("b",),
        )
        # LLM attempts substantive mark drift: criteria sum to 3 instead of 4 marks
        mock_response = {
            "label": "b",
            "text": "Analyze process scheduling algorithms in detail.",
            "solution": "Model solution for part b.",
            "marking_scheme": [
                {"criterion": "Part 1", "marks": 2},
                {"criterion": "Part 2", "marks": 1},  # Sum is 3, target is 4
            ],
            "change_summary": "Fixed verb",
        }
        mock_llm = MockLLMCaller(mock_response)
        agent = RefinementAgent(llm_caller=mock_llm)

        context = AgentContext(questions=[q], audit_report=AuditReport(verdicts=[verdict]))
        result = agent.run(context)

        assert result.success
        assert q.slot_id in result.payload.unresolved_slots
        # Question must be marked unresolved, candidate rejected, original text preserved (no rescaling)
        unresolved_q = context.questions[0]
        assert unresolved_q.is_unresolved is True
        assert unresolved_q.sub_parts[1].text == orig_text


# =============================================================================
# Gate G3-T08: Projection Consistency Verification
# =============================================================================

class TestG3T08ProjectionConsistency:
    """Verify that corrupt or desynchronized projections are rejected."""

    def test_rejection_of_corrupted_projections(self):
        q = make_sample_multipart_question()
        q.question_text = "Desynchronized text that does not match subparts!"
        consistent, err = q.verify_projection_consistency()
        assert not consistent
        assert "question_text does not match" in err


# =============================================================================
# Gate G3-T09: Evaluation Telemetry Preservation
# =============================================================================

class TestG3T09EvaluationTelemetry:
    """Verify subpart_failures count is recorded in audit report telemetry."""

    def test_evaluation_telemetry_records_subpart_failures(self):
        q = make_sample_multipart_question(
            sp_a_text="Explain the kernel.",
            sp_b_text="Describe the scheduling.",  # Fails L4
        )
        plan = make_sample_plan(sp_a_bloom="L2", sp_b_bloom="L4")

        agent = EvaluationAgent()
        context = type("MockContext", (), {"questions": [q], "plans": [plan], "audit_report": None})()
        result = agent.run(context)

        assert result.success
        report = result.payload
        assert "subpart_failures" in report.telemetry
        assert report.telemetry["subpart_failures"] >= 1


# =============================================================================
# Gate G3-T10: Legacy Single-Part Compatibility
# =============================================================================

class TestG3T10LegacySinglePartCompatibility:
    """Verify 100% backward compatibility for single-part questions without subparts."""

    def test_legacy_single_part_bloom_and_repair(self):
        q = GeneratedQuestion(
            slot_id="slot_legacy",
            module_id="mod_1",
            global_q_idx=1,
            question_text="Explain the operating system kernel subsystems in full detail.",
            solution="Comprehensive explanation.",
            marking_scheme=[{"criterion": "Full explanation", "marks": 10}],
            marks=10,
            partition=[10],
            bloom="L4",  # Needs L4
            co="CO3",
            topic="Operating Systems",
        )
        # Evaluator audit
        verdict = bloom_verb_evaluator(q, plan=None)
        assert verdict is not None
        assert verdict.verdict == "fail"
        assert verdict.reason_code == BLOOM_MISMATCH

        # Deterministic repair
        repaired = repair_bloom_verb(q, BLOOM_MISMATCH, verdict.detail)
        assert repaired is not None
        assert repaired.question_text.lower().startswith(("analyze", "analyse"))
        assert repaired.marks == 10
        assert len(repaired.sub_parts) == 0


# =============================================================================
# Non-Authorized Repairs Fail Closed on Multi-Part Questions (Issue A)
# =============================================================================

class TestG3NonAuthorizedRepairsFailClosed:
    """Verify non-authorized secondary deterministic repairs fail closed on multi-part questions."""

    def test_secondary_deterministic_repairs_fail_closed_on_subparts(self):
        q = make_sample_multipart_question(
            sp_a_text="Based on the evidence, explain the kernel architecture.",
            sp_b_text="With reference to part (a), describe scheduling.",
        )
        q.sub_parts[1].image_path = "figures/test.png"
        q.image_path = "figures/test.png"

        assert repair_sibling_independence(q, SIBLING_INTERCONNECTED, "detail") is None
        assert repair_figure_reference(q, MISSING_FIGURE_REFERENCE, "detail") is None
        assert repair_marking_scheme(q, MARKING_SCHEME_MISMATCH, "detail") is None
        assert repair_marking_scheme_rescale(q, MARKING_SCHEME_MISMATCH, "detail") is None
        assert repair_short_question(q, QUESTION_TOO_SHORT, "detail") is None


# =============================================================================
# Issue C: Failure-Path Test Suite (8 Mandatory Failure Paths)
# =============================================================================

class TestG3FailurePaths:
    """Comprehensive failure-path verification for Stage G3 downstream alignment."""

    def test_01_malformed_llm_replacement_rejected(self):
        """Malformed LLM replacement rejected; original question unchanged."""
        q = make_sample_multipart_question(
            sp_b_text="Illustrate with a neat sketch the process scheduling algorithms in detail.",
        )
        orig_text = q.question_text
        verdict = SlotVerdict(
            slot_id=q.slot_id,
            verdict="fail",
            source="local",
            reason_code=BLOOM_MISMATCH,
            detail="SubPart (b) verb issue for Bloom level L4",
            affected_subparts=("b",),
        )
        mock_llm = MockLLMCaller("INVALID JSON {not parseable")
        agent = RefinementAgent(llm_caller=mock_llm)
        context = AgentContext(questions=[q], audit_report=AuditReport(verdicts=[verdict]))
        result = agent.run(context)

        assert result.success
        assert q.slot_id in result.payload.unresolved_slots
        assert context.questions[0].is_unresolved is True
        assert context.questions[0].question_text == orig_text

    def test_02_llm_sibling_metadata_modification_prevented(self):
        """LLM repair targeted at SubPart (b) preserves sibling SubPart (a) metadata bit-for-bit."""
        q = make_sample_multipart_question(
            sp_b_text="Illustrate with a neat sketch the process scheduling algorithms in detail.",
        )
        q.sub_parts[0].figure_id = "fig_kernel_01"
        q.sub_parts[0].image_path = "figures/kernel.png"
        q.sub_parts[0].image_sha256 = "deadbeef12345678"
        q.sync_materialized_projections()

        orig_sp_a = copy.deepcopy(q.sub_parts[0])

        verdict = SlotVerdict(
            slot_id=q.slot_id,
            verdict="fail",
            source="local",
            reason_code=BLOOM_MISMATCH,
            detail="SubPart (b) verb issue for Bloom level L4",
            affected_subparts=("b",),
        )
        mock_response = {
            "label": "b",
            "text": "Analyze the process scheduling algorithms in detail.",
            "solution": "Repaired solution for part b.",
            "marking_scheme": [{"criterion": "Analysis", "marks": 4}],
            "change_summary": "Upgraded verb",
        }
        mock_llm = MockLLMCaller(mock_response)
        agent = RefinementAgent(llm_caller=mock_llm)
        context = AgentContext(questions=[q], audit_report=AuditReport(verdicts=[verdict]))
        agent.run(context)

        repaired_q = context.questions[0]
        sp_a = repaired_q.sub_parts[0]
        assert sp_a.figure_id == orig_sp_a.figure_id == "fig_kernel_01"
        assert sp_a.image_path == orig_sp_a.image_path == "figures/kernel.png"
        assert sp_a.image_sha256 == orig_sp_a.image_sha256 == "deadbeef12345678"
        assert sp_a.text == orig_sp_a.text
        assert sp_a.marks == orig_sp_a.marks
        assert sp_a.solution == orig_sp_a.solution

    def test_03_invalid_target_subpart_identity_rejected(self):
        """Invalid target subpart identity (e.g. label 'z') cannot match; candidate rejected."""
        q = make_sample_multipart_question(
            sp_b_text="Illustrate with a neat sketch the process scheduling algorithms in detail.",
        )
        orig_text = q.question_text
        verdict = SlotVerdict(
            slot_id=q.slot_id,
            verdict="fail",
            source="local",
            reason_code=BLOOM_MISMATCH,
            detail="SubPart (z) does not exist",
            affected_subparts=("z",),
        )
        mock_response = {
            "label": "z",
            "text": "Analyze something.",
            "solution": "Solution.",
            "marking_scheme": [{"criterion": "Criteria", "marks": 4}],
        }
        mock_llm = MockLLMCaller(mock_response)
        agent = RefinementAgent(llm_caller=mock_llm)
        context = AgentContext(questions=[q], audit_report=AuditReport(verdicts=[verdict]))
        agent.run(context)

        assert q.slot_id in context.unresolved_slots
        assert context.questions[0].question_text == orig_text

    def test_04_mark_corruption_attempt_rejected(self):
        """LLM attempts substantive mark corruption ('marks': 7 instead of 4) -> rejected."""
        q = make_sample_multipart_question(
            sp_b_text="Illustrate with a neat sketch the process scheduling algorithms in detail.",
        )
        orig_text = q.question_text
        verdict = SlotVerdict(
            slot_id=q.slot_id,
            verdict="fail",
            source="local",
            reason_code=BLOOM_MISMATCH,
            detail="SubPart (b) verb issue for Bloom level L4",
            affected_subparts=("b",),
        )
        mock_response = {
            "label": "b",
            "marks": 7,  # Subpart b has 4 marks, attempting to inflate to 7
            "text": "Analyze process scheduling.",
            "solution": "Solution.",
            "marking_scheme": [{"criterion": "Analysis", "marks": 7}],
        }
        mock_llm = MockLLMCaller(mock_response)
        agent = RefinementAgent(llm_caller=mock_llm)
        context = AgentContext(questions=[q], audit_report=AuditReport(verdicts=[verdict]))
        agent.run(context)

        assert q.slot_id in context.unresolved_slots
        assert context.questions[0].question_text == orig_text

    def test_05_projection_sync_failure_rejected(self):
        """SubPart validation failure (e.g. empty text) causes candidate rejection."""
        q = make_sample_multipart_question(
            sp_b_text="Illustrate with a neat sketch the process scheduling algorithms in detail.",
        )
        orig_text = q.question_text
        verdict = SlotVerdict(
            slot_id=q.slot_id,
            verdict="fail",
            source="local",
            reason_code=BLOOM_MISMATCH,
            detail="SubPart (b) verb issue for Bloom level L4",
            affected_subparts=("b",),
        )
        mock_response = {
            "label": "b",
            "text": "   ",  # Empty text violates SubPart.validate()
            "solution": "Solution.",
            "marking_scheme": [{"criterion": "Analysis", "marks": 4}],
        }
        mock_llm = MockLLMCaller(mock_response)
        agent = RefinementAgent(llm_caller=mock_llm)
        context = AgentContext(questions=[q], audit_report=AuditReport(verdicts=[verdict]))
        agent.run(context)

        assert q.slot_id in context.unresolved_slots
        assert context.questions[0].question_text == orig_text

    def test_06_retry_exhaustion_unresolved(self):
        """LLM retry exhaustion produces explicit unresolved slot with deterministic telemetry."""
        q = make_sample_multipart_question(
            sp_b_text="Illustrate with a neat sketch the process scheduling algorithms in detail.",
        )
        verdict = SlotVerdict(
            slot_id=q.slot_id,
            verdict="fail",
            source="local",
            reason_code=BLOOM_MISMATCH,
            detail="SubPart (b) verb issue for Bloom level L4",
            affected_subparts=("b",),
        )
        # LLM returns missing required keys repeatedly
        mock_llm = MockLLMCaller({"invalid_key": "some value"})
        agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=2)
        context = AgentContext(questions=[q], audit_report=AuditReport(verdicts=[verdict]))
        result = agent.run(context)

        assert q.slot_id in result.payload.unresolved_slots
        assert result.payload.outcomes[0].reason_out == "LLM_REPAIR_FAILED"
        unres_q = context.questions[0]
        assert unres_q.is_unresolved is True
        assert unres_q.unresolved_scope == "subquestion"
        assert unres_q.unresolved_subparts == ("b",)
        assert unres_q.failure_code == "BLOOM_MISMATCH"

    def test_07_missing_ambiguous_plan_correspondence_indeterminate(self):
        """Ambiguous or missing plan correspondence yields INDETERMINATE or LEGACY outcome."""
        # Case 1: missing plan for multi-part question falls back to LEGACY safely
        q = make_sample_multipart_question()
        status, _, _ = resolve_subpart_plan_correspondence(q, None)
        assert status == CorrespondenceOutcome.LEGACY

        # Case 2: slot_id mismatch yields INDETERMINATE
        q_mismatch = make_sample_multipart_question(slot_id="slot_A")
        plan = make_sample_plan(slot_id="slot_B")
        status, _, err_v = resolve_subpart_plan_correspondence(q_mismatch, plan)
        assert status == CorrespondenceOutcome.INDETERMINATE
        assert err_v is not None and err_v.reason_code == UNVERIFIED

        # Case 3: duplicate labels in plan yields INDETERMINATE
        q_norm = make_sample_multipart_question(slot_id="slot_1")
        plan_dup = make_sample_plan(slot_id="slot_1")
        plan_dup.sub_part_plans[1].label = "a"
        status, _, err_v = resolve_subpart_plan_correspondence(q_norm, plan_dup)
        assert status == CorrespondenceOutcome.INDETERMINATE
        assert err_v is not None and err_v.reason_code == UNVERIFIED

        # Case 4: duplicate labels in question yields FAIL (confirmed contract violation)
        q_dup = make_sample_multipart_question(slot_id="slot_1", sp_a_label="a", sp_b_label="a")
        plan_norm = make_sample_plan(slot_id="slot_1")
        status, _, err_v = resolve_subpart_plan_correspondence(q_dup, plan_norm)
        assert status == CorrespondenceOutcome.FAIL
        assert err_v is not None and err_v.reason_code == MULTI_PART_STRUCTURE_MISMATCH

    def test_08_unverified_evaluation_not_treated_as_successful_repair(self):
        """An unresolved repair is not silently recorded as resolved."""
        q = make_sample_multipart_question(
            sp_b_text="Describe the scheduling mechanism in detail."
        )
        verdict = SlotVerdict(
            slot_id=q.slot_id,
            verdict="fail",
            source="local",
            reason_code=BLOOM_MISMATCH,
            detail="SubPart (b) requires L4",
            affected_subparts=("b",),
        )
        agent = RefinementAgent(llm_caller=None)  # No LLM caller, cannot repair
        context = AgentContext(questions=[q], audit_report=AuditReport(verdicts=[verdict]))
        result = agent.run(context)

        assert result.payload.telemetry["resolved"] == 0
        assert result.payload.telemetry["unresolved"] == 1
        assert q.slot_id in result.payload.unresolved_slots
        assert context.questions[0].is_unresolved is True


# =============================================================================
# Issue D: E2E Refinement Path Diagnostic Scenario
# =============================================================================

class TestG3E2EDiagnosticScenario:
    """Explicit multi-part diagnostic flow: SubPart A passes, SubPart B fails, only B is repaired."""

    def test_e2e_refinement_path_diagnostic_scenario(self):
        # 1. Setup question: SubPart A passes (L2 'Explain'), SubPart B fails (requires L4 but starts with 'Describe')
        q = make_sample_multipart_question(
            sp_a_text="Explain the operating system kernel subsystems and system call interface.",
            sp_b_text="Describe the process scheduling algorithms and analyze scheduling delays.",
            sp_a_marks=6,
            sp_b_marks=4,
        )
        q.sub_parts[0].figure_id = "fig_os_kernel"
        q.sub_parts[0].image_path = "figures/os_kernel.png"
        q.sub_parts[0].image_sha256 = "c0ffee1234567890abcdef"
        q.sync_materialized_projections()

        orig_sp_a = copy.deepcopy(q.sub_parts[0])
        orig_sp_b = copy.deepcopy(q.sub_parts[1])

        plan = make_sample_plan(sp_a_bloom="L2", sp_b_bloom="L4", sp_a_marks=6, sp_b_marks=4)

        # 2. Run EvaluationAgent: should detect failure strictly on SubPart B
        eval_agent = EvaluationAgent()
        eval_ctx = type("MockContext", (), {"questions": [q], "plans": [plan], "audit_report": None})()
        eval_result = eval_agent.run(eval_ctx)

        assert eval_result.success
        report: AuditReport = eval_result.payload
        failed_verdicts = [v for v in report.verdicts if v.failed]
        assert len(failed_verdicts) >= 1
        b_verdict = next(v for v in failed_verdicts if "b" in getattr(v, "affected_subparts", ()))
        assert b_verdict.reason_code == BLOOM_MISMATCH

        # 3. Run RefinementAgent: provides L4 replacement for SubPart B
        l4_replacement = {
            "label": "b",
            "text": "Analyze the process scheduling algorithms and evaluate scheduling delays under load.",
            "solution": "Analytical solution with queue models.",
            "marking_scheme": [
                {"criterion": "Algorithm analysis", "marks": 2},
                {"criterion": "Delay evaluation", "marks": 2},
            ],
            "change_summary": "Upgraded subpart b verb to Analyze (L4)",
        }
        mock_llm = MockLLMCaller(l4_replacement)
        refine_agent = RefinementAgent(llm_caller=mock_llm)
        refine_ctx = AgentContext(questions=[q], audit_report=report)
        refine_result = refine_agent.run(refine_ctx)

        assert refine_result.success
        assert len(refine_result.payload.unresolved_slots) == 0

        repaired_q: GeneratedQuestion = refine_ctx.questions[0]
        assert repaired_q.is_unresolved is False

        # 4. Strict assertions:
        # A. Sibling SubPart A must be strictly BIT-FOR-BIT IDENTICAL to original
        sp_a = repaired_q.sub_parts[0]
        assert sp_a.label == orig_sp_a.label == "a"
        assert sp_a.marks == orig_sp_a.marks == 6
        assert sp_a.text == orig_sp_a.text
        assert sp_a.topic == orig_sp_a.topic
        assert sp_a.solution == orig_sp_a.solution
        assert sp_a.marking_scheme == orig_sp_a.marking_scheme
        assert sp_a.figure_id == orig_sp_a.figure_id == "fig_os_kernel"
        assert sp_a.image_path == orig_sp_a.image_path == "figures/os_kernel.png"
        assert sp_a.image_sha256 == orig_sp_a.image_sha256 == "c0ffee1234567890abcdef"

        # B. Target SubPart B is updated to the repaired text
        sp_b = repaired_q.sub_parts[1]
        assert sp_b.label == "b"
        assert sp_b.marks == 4
        assert sp_b.text == l4_replacement["text"]
        assert sp_b.solution == l4_replacement["solution"]
        assert sum(c["marks"] for c in sp_b.marking_scheme) == 4

        # C. Materialized parent projections synchronize
        assert repaired_q.question_text == f"(a) {orig_sp_a.text}\n\n(b) {l4_replacement['text']}"
        consistent, err = repaired_q.verify_projection_consistency()
        assert consistent and err == ""

        # D. Re-evaluation through EvaluationAgent passes cleanly
        re_eval_ctx = type("MockContext", (), {"questions": [repaired_q], "plans": [plan], "audit_report": None})()
        re_eval_result = eval_agent.run(re_eval_ctx)
        re_report: AuditReport = re_eval_result.payload
        re_failed = [v for v in re_report.verdicts if v.failed]
        assert len(re_failed) == 0

