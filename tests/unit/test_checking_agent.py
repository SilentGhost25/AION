"""
Unit tests for the Checking Agent.
"""

import os
from dataclasses import dataclass
from typing import Any, List

import pytest

from core.generation.agents import (
    AgentContext,
    AuditReport,
    BLOCK_REASON_INSUFFICIENT_QUESTIONS,
    BLOCK_REASON_UNRESOLVED_SLOTS,
    BLOCK_REASON_WRITING_FAILURES,
    CheckingAgent,
    GeneratedQuestion,
    PaperBlockedError,
    QAReport,
    STATUS_BLOCKED,
    STATUS_DEGRADED,
    STATUS_PASS,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("AION_ENABLE_UNRESOLVED_HARD_BLOCK", raising=False)


@dataclass
class StubSpec:
    total_questions: int = 10


def make_question(
    slot_id="module_1_Q1",
    question_text="Explain how Docker manages isolation.",
    solution="Docker uses namespaces and cgroups.",
):
    return GeneratedQuestion(
        slot_id=slot_id,
        module_id="module_1",
        global_q_idx=1,
        question_text=question_text,
        solution=solution,
        marking_scheme=[{"criterion": "x", "marks": 10}],
        marks=10,
        partition=[10],
        bloom="L2",
        co="CO1",
    )


def make_context(questions=None, unresolved=None, expected=10, telemetry=None):
    ctx = AgentContext(
        paper_spec=StubSpec(total_questions=expected),
        questions=list(questions or []),
        unresolved_slots=list(unresolved or []),
        telemetry=telemetry or {},
    )
    return ctx


# -----------------------------------------------------------------------------
# Happy path
# -----------------------------------------------------------------------------


def test_all_resolved_returns_pass():
    questions = [make_question(slot_id=f"Q{i}") for i in range(10)]
    ctx = make_context(questions=questions, expected=10)
    result = CheckingAgent().run(ctx)
    assert result.success is True
    report = result.payload
    assert isinstance(report, QAReport)
    assert report.status == STATUS_PASS
    assert report.passes is True
    assert report.blocked is False
    assert report.resolved_slots == 10
    assert report.unresolved_slots == []


# -----------------------------------------------------------------------------
# Fail-closed behavior (hard block enabled by default)
# -----------------------------------------------------------------------------


def test_unresolved_slots_block_by_default():
    questions = [make_question(slot_id=f"Q{i}") for i in range(9)]
    ctx = make_context(questions=questions, unresolved=["Q10"], expected=10)
    result = CheckingAgent().run(ctx)
    report = result.payload
    assert report.status == STATUS_BLOCKED
    assert report.blocked is True
    assert BLOCK_REASON_UNRESOLVED_SLOTS in report.blocked_reasons
    assert report.unresolved_slots == ["Q10"]


def test_hard_block_disabled_degrades_instead(monkeypatch):
    monkeypatch.setenv("AION_ENABLE_UNRESOLVED_HARD_BLOCK", "false")
    questions = [make_question(slot_id=f"Q{i}") for i in range(9)]
    ctx = make_context(questions=questions, unresolved=["Q10"], expected=10)
    report = CheckingAgent().run(ctx).payload
    assert report.status == STATUS_DEGRADED
    assert report.degraded is True
    assert BLOCK_REASON_UNRESOLVED_SLOTS in report.degraded_reasons


def test_hard_block_explicit_true(monkeypatch):
    monkeypatch.setenv("AION_ENABLE_UNRESOLVED_HARD_BLOCK", "true")
    questions = [make_question(slot_id=f"Q{i}") for i in range(9)]
    ctx = make_context(questions=questions, unresolved=["Q10"], expected=10)
    report = CheckingAgent().run(ctx).payload
    assert report.status == STATUS_BLOCKED


# -----------------------------------------------------------------------------
# Insufficient questions
# -----------------------------------------------------------------------------


def test_insufficient_questions_blocks():
    # Only 3 of 10 expected slots resolved
    questions = [make_question(slot_id=f"Q{i}") for i in range(3)]
    ctx = make_context(questions=questions, expected=10)
    report = CheckingAgent().run(ctx).payload
    assert report.status == STATUS_BLOCKED
    assert BLOCK_REASON_INSUFFICIENT_QUESTIONS in report.blocked_reasons


def test_boundary_ratio_accepted(monkeypatch):
    # 9 of 10 = 0.9 exactly at default threshold
    questions = [make_question(slot_id=f"Q{i}") for i in range(9)]
    ctx = make_context(questions=questions, expected=10)
    report = CheckingAgent().run(ctx).payload
    # No unresolved slots, ratio == 0.9, should PASS
    assert report.status == STATUS_PASS


def test_custom_min_ratio():
    agent = CheckingAgent(min_questions_ratio=0.5)
    questions = [make_question(slot_id=f"Q{i}") for i in range(5)]
    ctx = make_context(questions=questions, expected=10)
    report = agent.run(ctx).payload
    assert report.status == STATUS_PASS


# -----------------------------------------------------------------------------
# Empty / degenerate cases
# -----------------------------------------------------------------------------


def test_no_questions_blocks():
    ctx = make_context(questions=[], expected=10)
    report = CheckingAgent().run(ctx).payload
    assert report.status == STATUS_BLOCKED
    assert BLOCK_REASON_INSUFFICIENT_QUESTIONS in report.blocked_reasons


def test_no_spec_expected_is_zero():
    ctx = AgentContext(questions=[make_question()])
    report = CheckingAgent().run(ctx).payload
    # With no spec, expected = 0; ratio check skipped
    assert report.slot_count == 1
    assert report.status == STATUS_PASS


def test_empty_slot_text_is_not_resolved():
    # Question with empty text
    bad_q = GeneratedQuestion(
        slot_id="Q1", module_id="m1", global_q_idx=1,
        question_text="", solution="some answer",
        marking_scheme=[{"criterion": "x", "marks": 10}],
        marks=10, partition=[10],
    )
    ctx = make_context(questions=[bad_q], expected=1)
    report = CheckingAgent().run(ctx).payload
    assert report.resolved_slots == 0
    assert report.status == STATUS_BLOCKED


def test_empty_solution_is_not_resolved():
    bad_q = GeneratedQuestion(
        slot_id="Q1", module_id="m1", global_q_idx=1,
        question_text="Explain Docker.", solution="",
        marking_scheme=[{"criterion": "x", "marks": 10}],
        marks=10, partition=[10],
    )
    ctx = make_context(questions=[bad_q], expected=1)
    report = CheckingAgent().run(ctx).payload
    assert report.resolved_slots == 0


# -----------------------------------------------------------------------------
# Writing failures
# -----------------------------------------------------------------------------


def test_writing_failures_in_telemetry_block_when_hard_block():
    telemetry = {
        "writing": {"questions_generated": 8, "failures": 2},
        "writing_failures": [{"slot_id": "Q9"}, {"slot_id": "Q10"}],
    }
    ctx = make_context(
        questions=[make_question(slot_id=f"Q{i}") for i in range(8)],
        expected=10,
        telemetry=telemetry,
    )
    report = CheckingAgent().run(ctx).payload
    assert report.status == STATUS_BLOCKED
    # Either insufficient OR writing failures may fire — both are valid
    assert (
        BLOCK_REASON_WRITING_FAILURES in report.blocked_reasons
        or BLOCK_REASON_INSUFFICIENT_QUESTIONS in report.blocked_reasons
    )


# -----------------------------------------------------------------------------
# Telemetry aggregation
# -----------------------------------------------------------------------------


def test_telemetry_is_aggregated_into_report():
    telemetry = {
        "planning": {"plans_count": 10, "visual_count": 4},
        "writing": {"questions_generated": 10, "sources": {"local_llm": 6, "local_vlm": 4}},
        "evaluation": {"api_evaluated": 10},
        "refinement": {"deterministic_repairs": 2, "llm_repairs": 1},
    }
    ctx = make_context(
        questions=[make_question(slot_id=f"Q{i}") for i in range(10)],
        expected=10,
        telemetry=telemetry,
    )
    report = CheckingAgent().run(ctx).payload
    assert "planning" in report.telemetry
    assert "writing" in report.telemetry
    assert "evaluation" in report.telemetry
    assert "refinement" in report.telemetry
    assert "checking" in report.telemetry
    assert report.telemetry["checking"]["expected_slots"] == 10


def test_cost_telemetry_extracted():
    telemetry = {"cost": {"api_calls": 3, "cost_usd": 0.045}}
    ctx = make_context(
        questions=[make_question(slot_id=f"Q{i}") for i in range(10)],
        expected=10,
        telemetry=telemetry,
    )
    report = CheckingAgent().run(ctx).payload
    assert report.cost_telemetry == {"api_calls": 3, "cost_usd": 0.045}


def test_cache_telemetry_extracted():
    telemetry = {"cache": {"hits": 5, "misses": 5, "hit_rate": 0.5}}
    ctx = make_context(
        questions=[make_question(slot_id=f"Q{i}") for i in range(10)],
        expected=10,
        telemetry=telemetry,
    )
    report = CheckingAgent().run(ctx).payload
    assert report.cache_telemetry == {"hits": 5, "misses": 5, "hit_rate": 0.5}


def test_non_serializable_telemetry_coerced():
    class Custom:
        def __repr__(self):
            return "<custom>"
    telemetry = {"weird": {"obj": Custom()}}
    ctx = make_context(
        questions=[make_question(slot_id=f"Q{i}") for i in range(10)],
        expected=10,
        telemetry=telemetry,
    )
    # Should not raise
    report = CheckingAgent().run(ctx).payload
    # Non-serializable values coerced to strings
    assert "weird" in report.telemetry


# -----------------------------------------------------------------------------
# QAReport contract
# -----------------------------------------------------------------------------


def test_qa_report_to_dict():
    report = QAReport(
        status=STATUS_BLOCKED,
        slot_count=10,
        resolved_slots=8,
        unresolved_slots=["Q9", "Q10"],
        blocked_reasons=[BLOCK_REASON_UNRESOLVED_SLOTS],
        telemetry={"checking": {"expected_slots": 10}},
    )
    d = report.to_dict()
    assert d["status"] == STATUS_BLOCKED
    assert d["unresolved_slots"] == ["Q9", "Q10"]
    assert d["blocked_reasons"] == [BLOCK_REASON_UNRESOLVED_SLOTS]
    assert d["slot_count"] == 10


def test_blocked_report_requires_reason():
    with pytest.raises(ValueError):
        QAReport(status=STATUS_BLOCKED)


def test_invalid_status_rejected():
    with pytest.raises(ValueError):
        QAReport(status="WHATEVER")


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------


def test_invalid_min_ratio_rejected():
    with pytest.raises(ValueError):
        CheckingAgent(min_questions_ratio=1.5)
    with pytest.raises(ValueError):
        CheckingAgent(min_questions_ratio=-0.1)


# -----------------------------------------------------------------------------
# PaperBlockedError
# -----------------------------------------------------------------------------


def test_paper_blocked_error_carries_report():
    report = QAReport(
        status=STATUS_BLOCKED,
        blocked_reasons=[BLOCK_REASON_UNRESOLVED_SLOTS],
        unresolved_slots=["Q1", "Q2"],
    )
    err = PaperBlockedError(report)
    assert err.report is report
    assert "UNRESOLVED_SLOTS" in str(err)
    assert "2" in str(err)


# -----------------------------------------------------------------------------
# Degraded mode with multiple reasons
# -----------------------------------------------------------------------------


def test_degraded_can_carry_multiple_reasons(monkeypatch):
    monkeypatch.setenv("AION_ENABLE_UNRESOLVED_HARD_BLOCK", "false")
    telemetry = {
        "writing_failures": [{"slot_id": "Q10"}],
    }
    ctx = make_context(
        questions=[make_question(slot_id=f"Q{i}") for i in range(9)],
        unresolved=["Q10"],
        expected=10,
        telemetry=telemetry,
    )
    report = CheckingAgent().run(ctx).payload
    assert report.status == STATUS_DEGRADED
    # At least one degraded reason recorded
    assert len(report.degraded_reasons) >= 1


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_determinism_same_input_same_report():
    questions = [make_question(slot_id=f"Q{i}") for i in range(10)]
    def run_once():
        ctx = make_context(questions=questions, expected=10)
        return CheckingAgent().run(ctx).payload
    r1 = run_once()
    r2 = run_once()
    assert r1.status == r2.status
    assert r1.resolved_slots == r2.resolved_slots
    assert r1.unresolved_slots == r2.unresolved_slots
