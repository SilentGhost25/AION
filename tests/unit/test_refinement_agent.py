"""
Unit tests for the Refinement Agent.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from core.generation.agents import (
    AgentContext,
    AuditReport,
    BLOOM_MISMATCH,
    CIRCULAR,
    FACTUAL_ERROR,
    GeneratedQuestion,
    LLMCaller,
    MARKING_SCHEME_MISMATCH,
    MISSING_FIGURE_REFERENCE,
    MULTI_PART_STRUCTURE_MISMATCH,
    QUESTION_TOO_SHORT,
    RefinementAgent,
    RefinementOutcome,
    RefinementReport,
    SlotVerdict,
    TIER_DETERMINISTIC,
    TIER_LLM,
    TIER_NONE,
    UNVERIFIED,
    UNGROUNDED,
)


# -----------------------------------------------------------------------------
# Mocks
# -----------------------------------------------------------------------------


class SequenceLLM:
    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def call(self, prompt, schema=None, image_path=None, seed=None):
        self.calls.append({"prompt": prompt, "seed": seed})
        if not self._responses:
            raise RuntimeError("SequenceLLM exhausted")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class AlwaysTrueSympy:
    def __init__(self):
        self.calls = []

    def verify(self, question_text, solution_text):
        self.calls.append((question_text, solution_text))
        return True


class AlwaysFalseSympy:
    def __init__(self):
        self.calls = []

    def verify(self, question_text, solution_text):
        self.calls.append((question_text, solution_text))
        return False


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def make_question(
    slot_id="module_1_Q1",
    question_text="Explain how Docker manages isolation.",
    solution="Docker uses namespaces and cgroups.",
    marking_scheme=None,
    marks=10,
    partition=None,
    bloom="L2",
    co="CO1",
    topic="Docker",
    image_path=None,
    references_image=False,
):
    return GeneratedQuestion(
        slot_id=slot_id,
        module_id="module_1",
        global_q_idx=1,
        question_text=question_text,
        solution=solution,
        marking_scheme=marking_scheme or [{"criterion": "x", "marks": marks}],
        marks=marks,
        partition=partition or [marks],
        bloom=bloom,
        co=co,
        topic=topic,
        image_path=image_path,
        references_image=references_image,
    )


def fail_verdict(slot_id, reason_code, detail="issue", suggested_fix=None):
    return SlotVerdict(
        slot_id=slot_id,
        verdict="fail",
        source="api",
        reason_code=reason_code,
        detail=detail,
        suggested_fix=suggested_fix,
    )


def pass_verdict(slot_id):
    return SlotVerdict(slot_id=slot_id, verdict="pass", source="api")


def make_context(questions, verdicts):
    return AgentContext(
        questions=list(questions),
        audit_report=AuditReport(verdicts=list(verdicts)),
    )


# -----------------------------------------------------------------------------
# No-op cases
# -----------------------------------------------------------------------------


def test_no_audit_report_returns_empty():
    agent = RefinementAgent()
    ctx = AgentContext(questions=[make_question()], audit_report=None)
    result = agent.run(ctx)
    assert result.success is True
    assert isinstance(result.payload, RefinementReport)
    assert result.payload.outcomes == []


def test_all_pass_leaves_questions_unchanged():
    q = make_question()
    agent = RefinementAgent()
    ctx = make_context([q], [pass_verdict("module_1_Q1")])
    result = agent.run(ctx)
    report = result.payload
    assert report.unresolved_slots == []
    assert len(report.repaired_questions) == 1
    assert report.repaired_questions[0] is q


# -----------------------------------------------------------------------------
# Deterministic repairs
# -----------------------------------------------------------------------------


def test_deterministic_bloom_verb_repair():
    # L2 requires "explain"/"describe"/etc. Question starts with "Analyze".
    q = make_question(bloom="L2", question_text="Analyze how Docker isolates containers.")
    agent = RefinementAgent()
    ctx = make_context([q], [fail_verdict("module_1_Q1", BLOOM_MISMATCH)])
    result = agent.run(ctx)
    report = result.payload
    assert report.deterministic_repairs == 1
    assert report.unresolved_slots == []
    outcome = report.outcome_for("module_1_Q1")
    assert outcome.resolved is True
    assert outcome.tier_used == TIER_DETERMINISTIC
    repaired = report.repaired_questions[0]
    assert repaired.question_text.lower().startswith("explain")


def test_deterministic_figure_reference_repair():
    q = make_question(
        image_path="/tmp/fig.png",
        references_image=False,
        question_text="Explain the orbit's eccentricity.",
    )
    agent = RefinementAgent()
    ctx = make_context([q], [fail_verdict("module_1_Q1", MISSING_FIGURE_REFERENCE)])
    result = agent.run(ctx)
    report = result.payload
    assert report.deterministic_repairs == 1
    repaired = report.repaired_questions[0]
    assert "given figure" in repaired.question_text.lower()
    assert repaired.references_image is True


def test_deterministic_marking_scheme_single_entry():
    q = make_question(
        marks=10,
        marking_scheme=[{"criterion": "answer", "marks": 4}],
    )
    agent = RefinementAgent()
    ctx = make_context([q], [fail_verdict("module_1_Q1", MARKING_SCHEME_MISMATCH)])
    result = agent.run(ctx)
    report = result.payload
    assert report.deterministic_repairs == 1
    repaired = report.repaired_questions[0]
    assert repaired.marking_scheme[0]["marks"] == 10


def test_deterministic_short_question_repair():
    q = make_question(question_text="Explain.", topic="Docker networking")
    agent = RefinementAgent()
    ctx = make_context([q], [fail_verdict("module_1_Q1", QUESTION_TOO_SHORT)])
    result = agent.run(ctx)
    report = result.payload
    repaired = report.repaired_questions[0]
    assert "Docker networking" in repaired.question_text


# -----------------------------------------------------------------------------
# LLM repairs
# -----------------------------------------------------------------------------


def test_llm_repair_fires_when_deterministic_cannot_help():
    q = make_question(
        question_text="Explain how Docker manages isolation.",
        solution="Docker uses namespaces.",
    )
    llm = SequenceLLM([{
        "question_text": "Describe how Docker uses namespaces and cgroups.",
        "solution": "Docker uses Linux namespaces for isolation and cgroups for resource limits.",
        "marking_scheme": [{"criterion": "complete answer", "marks": 10}],
        "diagram_request": None,
        "references_image": False,
        "change_summary": "Replaced circular definition with application question.",
    }])
    agent = RefinementAgent(llm_caller=llm)
    ctx = make_context([q], [fail_verdict("module_1_Q1", CIRCULAR)])
    result = agent.run(ctx)
    report = result.payload
    assert report.llm_repairs == 1
    assert report.unresolved_slots == []
    repaired = report.repaired_questions[0]
    assert "namespaces" in repaired.question_text


def test_llm_repair_budget_enforced():
    # 3 failing slots, budget = 1 → 1 repaired, 2 unresolved
    q1 = make_question(slot_id="Q1", question_text="Explain A.")
    q2 = make_question(slot_id="Q2", question_text="Explain B.")
    q3 = make_question(slot_id="Q3", question_text="Explain C.")
    llm = SequenceLLM([{
        "question_text": "Describe A in detail.",
        "solution": "A detailed answer.",
        "marking_scheme": [{"criterion": "x", "marks": 10}],
    }])
    agent = RefinementAgent(llm_caller=llm, max_llm_repairs_per_paper=1)
    ctx = make_context(
        [q1, q2, q3],
        [
            fail_verdict("Q1", CIRCULAR),
            fail_verdict("Q2", CIRCULAR),
            fail_verdict("Q3", CIRCULAR),
        ],
    )
    result = agent.run(ctx)
    report = result.payload
    assert report.llm_repairs == 1
    assert len(report.unresolved_slots) == 2
    # Remaining two should be marked LLM_BUDGET_EXHAUSTED
    assert sum(1 for o in report.outcomes if o.reason_out == "LLM_BUDGET_EXHAUSTED") == 2


def test_no_llm_caller_marks_unresolved():
    q = make_question()
    agent = RefinementAgent(llm_caller=None)
    ctx = make_context([q], [fail_verdict("module_1_Q1", CIRCULAR)])
    result = agent.run(ctx)
    report = result.payload
    assert len(report.unresolved_slots) == 1
    outcome = report.outcome_for("module_1_Q1")
    assert outcome.reason_out == "NO_LLM_CALLER"


def test_llm_repair_retries_on_bad_json():
    q = make_question()
    llm = SequenceLLM([
        "not json",
        {
            "question_text": "Describe A in detail.",
            "solution": "Detailed answer.",
            "marking_scheme": [{"criterion": "x", "marks": 10}],
        },
    ])
    agent = RefinementAgent(llm_caller=llm)
    ctx = make_context([q], [fail_verdict("module_1_Q1", CIRCULAR)])
    result = agent.run(ctx)
    assert result.payload.llm_repairs == 1


def test_llm_repair_gives_up_after_max_attempts():
    q = make_question()
    llm = SequenceLLM(["bad", "bad"])
    agent = RefinementAgent(llm_caller=llm, max_attempts_per_slot=2)
    ctx = make_context([q], [fail_verdict("module_1_Q1", CIRCULAR)])
    result = agent.run(ctx)
    report = result.payload
    assert report.llm_repairs == 0
    assert report.outcome_for("module_1_Q1").reason_out == "LLM_REPAIR_FAILED"


# -----------------------------------------------------------------------------
# SymPy verification
# -----------------------------------------------------------------------------


def test_sympy_verification_passes():
    q = make_question(
        question_text="Calculate the orbital velocity given r = 7000 km.",
        solution="Using v = sqrt(GM/r), we get 7.5 km/s.",
        marking_scheme=[{"criterion": "correct answer", "marks": 10}],
    )
    llm = SequenceLLM([{
        "question_text": "Calculate the orbital velocity for a satellite at r = 7000 km.",
        "solution": "v = sqrt(3.986e5 / 7000) ≈ 7.55 km/s.",
        "marking_scheme": [{"criterion": "correct answer", "marks": 10}],
    }])
    sympy = AlwaysTrueSympy()
    agent = RefinementAgent(llm_caller=llm, sympy_verifier=sympy)
    ctx = make_context([q], [fail_verdict("module_1_Q1", FACTUAL_ERROR)])
    result = agent.run(ctx)
    assert result.payload.unresolved_slots == []
    assert len(sympy.calls) == 1


def test_sympy_verification_fails():
    q = make_question(
        question_text="Calculate the orbital velocity.",
        solution="v = 100 km/s.",
        marking_scheme=[{"criterion": "x", "marks": 10}],
    )
    llm = SequenceLLM([{
        "question_text": "Calculate orbital velocity at r = 7000 km.",
        "solution": "v = 999 km/s (wrong).",
        "marking_scheme": [{"criterion": "x", "marks": 10}],
    }])
    sympy = AlwaysFalseSympy()
    agent = RefinementAgent(llm_caller=llm, sympy_verifier=sympy)
    ctx = make_context([q], [fail_verdict("module_1_Q1", FACTUAL_ERROR)])
    result = agent.run(ctx)
    report = result.payload
    assert len(report.unresolved_slots) == 1
    assert report.sympy_rejections == 1
    assert report.outcome_for("module_1_Q1").reason_out == "SYMPY_VERIFICATION_FAILED"


def test_sympy_skipped_for_non_numerical():
    q = make_question(
        question_text="Explain how Docker manages isolation.",
        solution="Docker uses namespaces and cgroups.",
    )
    llm = SequenceLLM([{
        "question_text": "Describe Docker isolation mechanisms.",
        "solution": "Docker uses Linux namespaces and cgroups.",
        "marking_scheme": [{"criterion": "x", "marks": 10}],
    }])
    sympy = AlwaysTrueSympy()
    agent = RefinementAgent(llm_caller=llm, sympy_verifier=sympy)
    ctx = make_context([q], [fail_verdict("module_1_Q1", CIRCULAR)])
    result = agent.run(ctx)
    assert len(sympy.calls) == 0  # not numerical


# -----------------------------------------------------------------------------
# Multi-part structure — should go to LLM, not deterministic
# -----------------------------------------------------------------------------


def test_multi_part_structure_goes_to_llm():
    q = make_question(
        partition=[6, 4],
        marks=10,
        question_text="Explain Docker isolation and resource limits without sub-parts.",
        marking_scheme=[
            {"criterion": "isolation", "marks": 6},
            {"criterion": "limits", "marks": 4},
        ],
    )
    llm = SequenceLLM([{
        "question_text": "(i) Explain Docker isolation. (ii) Discuss resource limits.",
        "solution": "Full answer covering both.",
        "marking_scheme": [
            {"criterion": "isolation", "marks": 6},
            {"criterion": "limits", "marks": 4},
        ],
    }])
    agent = RefinementAgent(llm_caller=llm)
    ctx = make_context([q], [fail_verdict("module_1_Q1", MULTI_PART_STRUCTURE_MISMATCH)])
    result = agent.run(ctx)
    assert result.payload.llm_repairs == 1


# -----------------------------------------------------------------------------
# Mixed batch
# -----------------------------------------------------------------------------


def test_mixed_pass_and_fail_batch():
    q1 = make_question(slot_id="Q1")                       # passes
    q2 = make_question(slot_id="Q2", bloom="L2",
                       question_text="Analyze Docker isolation.")  # deterministic
    q3 = make_question(slot_id="Q3", question_text="Explain A.")    # llm
    llm = SequenceLLM([{
        "question_text": "Describe Q3 in detail.",
        "solution": "Detailed.",
        "marking_scheme": [{"criterion": "x", "marks": 10}],
    }])
    agent = RefinementAgent(llm_caller=llm)
    ctx = make_context(
        [q1, q2, q3],
        [
            pass_verdict("Q1"),
            fail_verdict("Q2", BLOOM_MISMATCH),
            fail_verdict("Q3", CIRCULAR),
        ],
    )
    result = agent.run(ctx)
    report = result.payload
    assert report.deterministic_repairs == 1
    assert report.llm_repairs == 1
    assert report.unresolved_slots == []
    # Order preserved
    assert [q.slot_id for q in report.repaired_questions] == ["Q1", "Q2", "Q3"]


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        RefinementAgent(max_llm_repairs_per_paper=-1)


def test_invalid_attempts_rejected():
    with pytest.raises(ValueError):
        RefinementAgent(max_attempts_per_slot=0)


# -----------------------------------------------------------------------------
# Contract checks
# -----------------------------------------------------------------------------


def test_refinement_outcome_requires_reason_when_unresolved():
    with pytest.raises(ValueError):
        RefinementOutcome(
            slot_id="Q1", resolved=False,
            tier_used=TIER_NONE, reason_in=CIRCULAR,
            # missing reason_out
        )


def test_refinement_outcome_rejects_unknown_tier():
    with pytest.raises(ValueError):
        RefinementOutcome(
            slot_id="Q1", resolved=True,
            tier_used="invalid", reason_in=CIRCULAR,
        )


# -----------------------------------------------------------------------------
# Context mutation
# -----------------------------------------------------------------------------


def test_context_updated_with_repaired_questions():
    q = make_question(bloom="L2",
                      question_text="Analyze Docker isolation.")
    agent = RefinementAgent()
    ctx = make_context([q], [fail_verdict("module_1_Q1", BLOOM_MISMATCH)])
    agent.run(ctx)
    assert ctx.questions[0].question_text.lower().startswith("explain")
    assert ctx.unresolved_slots == []


def test_context_records_unresolved_slots():
    q = make_question()
    agent = RefinementAgent(llm_caller=None)
    ctx = make_context([q], [fail_verdict("module_1_Q1", CIRCULAR)])
    agent.run(ctx)
    assert ctx.unresolved_slots == ["module_1_Q1"]
