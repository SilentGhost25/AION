"""
Unit tests for the Agent Orchestrator.

Uses stub agents so each stage can be controlled independently.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from core.generation.agents import (
    AgentContext,
    AgentOrchestrator,
    AgentResult,
    AuditReport,
    BLOCK_REASON_UNRESOLVED_SLOTS,
    GeneratedQuestion,
    OrchestratorResult,
    PaperBlockedError,
    QAReport,
    QuestionPlan,
    STATUS_BLOCKED,
    STATUS_DEGRADED,
    STATUS_PASS,
    SlotVerdict,
)


# -----------------------------------------------------------------------------
# Stub agents
# -----------------------------------------------------------------------------


class StubPlanningAgent:
    name = "planning"

    def __init__(self, plans=None, fail=False):
        self._plans = plans or []
        self._fail = fail

    def run(self, context):
        if self._fail:
            return AgentResult(success=False, failure_code="PLAN_FAIL",
                               failure_detail="stub planning failure")
        context.plans = self._plans
        return AgentResult(
            success=True,
            payload=self._plans,
            telemetry={"plans_count": len(self._plans)},
        )


class StubWritingAgent:
    name = "writing"

    def __init__(self, questions=None, failures=None, fail=False):
        self._questions = questions or []
        self._failures = failures or []
        self._fail = fail

    def run(self, context):
        if self._fail:
            return AgentResult(success=False, failure_code="WRITE_FAIL",
                               failure_detail="stub writing failure")
        # Simulate the WritingAgentOutput shape
        class Out:
            def __init__(self, q, f):
                self.questions = q
                self.failures = f
        context.questions = list(self._questions)
        return AgentResult(
            success=True,
            payload=Out(self._questions, self._failures),
            telemetry={"questions_generated": len(self._questions),
                       "failures": len(self._failures)},
        )


class StubEvaluationAgent:
    name = "evaluation"

    def __init__(self, verdicts=None, fail=False):
        self._verdicts = verdicts or []
        self._fail = fail

    def run(self, context):
        if self._fail:
            return AgentResult(success=False, failure_code="EVAL_FAIL",
                               failure_detail="stub evaluation failure")
        report = AuditReport(verdicts=self._verdicts)
        context.audit_report = report
        return AgentResult(
            success=True,
            payload=report,
            telemetry={"questions_evaluated": len(self._verdicts)},
        )


class StubRefinementAgent:
    name = "refinement"

    def __init__(self, fail=False):
        self._fail = fail

    def run(self, context):
        if self._fail:
            return AgentResult(success=False, failure_code="REFINE_FAIL",
                               failure_detail="stub refinement failure")
        # No-op refinement
        from core.generation.agents.refinement_contracts import RefinementReport
        report = RefinementReport()
        return AgentResult(success=True, payload=report,
                           telemetry={"resolved": 0, "unresolved": 0})


class StubCheckingAgent:
    name = "checking"

    def __init__(self, status=STATUS_PASS, block_reasons=None, fail=False):
        self._status = status
        self._block_reasons = block_reasons or []
        self._fail = fail

    def run(self, context):
        if self._fail:
            return AgentResult(success=False, failure_code="CHECK_FAIL",
                               failure_detail="stub checking failure")
        # Derive unresolved from context
        unresolved = list(context.unresolved_slots or [])
        report = QAReport(
            status=self._status,
            slot_count=len(context.questions) + len(unresolved),
            resolved_slots=len(context.questions),
            unresolved_slots=unresolved,
            blocked_reasons=list(self._block_reasons) if self._status == STATUS_BLOCKED else [],
            degraded_reasons=["UNRESOLVED_SLOTS"] if self._status == STATUS_DEGRADED else [],
            telemetry={"checking": {"expected_slots": 10}},
        )
        context.qa_report = report
        return AgentResult(success=True, payload=report,
                           telemetry={"status": self._status})


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def make_question(slot_id="Q1"):
    return GeneratedQuestion(
        slot_id=slot_id,
        module_id="module_1",
        global_q_idx=1,
        question_text="Explain Docker.",
        solution="Docker is a platform.",
        marking_scheme=[{"criterion": "x", "marks": 10}],
        marks=10,
        partition=[10],
    )


def make_context():
    return AgentContext(paper_spec=None, artifact=None, knowledge_graph=None,
                        marks_split=[], telemetry={})


def make_orchestrator(
    plans=None, questions=None, writing_failures=None,
    verdicts=None, status=STATUS_PASS, block_reasons=None,
    plan_fail=False, write_fail=False, eval_fail=False,
    refine_fail=False, check_fail=False,
    include_eval=True, include_refine=True,
):
    return AgentOrchestrator(
        planning_agent=StubPlanningAgent(plans=plans, fail=plan_fail),
        writing_agent=StubWritingAgent(
            questions=questions, failures=writing_failures, fail=write_fail
        ),
        evaluation_agent=StubEvaluationAgent(verdicts=verdicts, fail=eval_fail)
            if include_eval else None,
        refinement_agent=StubRefinementAgent(fail=refine_fail)
            if include_refine else None,
        checking_agent=StubCheckingAgent(
            status=status, block_reasons=block_reasons, fail=check_fail
        ),
    )


# -----------------------------------------------------------------------------
# Happy path
# -----------------------------------------------------------------------------


def test_full_pipeline_success():
    orch = make_orchestrator(
        plans=[QuestionPlan(slot_id="Q1", module_id="module_1", module_idx=1,
                            slot_idx=0, global_q_idx=1, partition=[10], total_marks=10)],
        questions=[make_question("Q1")],
        verdicts=[SlotVerdict(slot_id="Q1", verdict="pass", source="api")],
        status=STATUS_PASS,
    )
    result = orch.run(make_context())
    assert isinstance(result, OrchestratorResult)
    assert result.success is True
    assert result.qa_report is not None
    assert result.qa_report.status == STATUS_PASS
    assert result.paper_passed is True
    assert len(result.plans) == 1
    assert len(result.questions) == 1


# -----------------------------------------------------------------------------
# Agent crash handling
# -----------------------------------------------------------------------------


def test_planning_failure_returns_early():
    orch = make_orchestrator(plan_fail=True)
    result = orch.run(make_context())
    assert result.success is False
    assert result.failure_code == "PLANNING_FAILED"


def test_writing_failure_returns_early():
    orch = make_orchestrator(write_fail=True)
    result = orch.run(make_context())
    assert result.success is False
    assert result.failure_code == "WRITING_FAILED"


def test_evaluation_failure_returns_early():
    orch = make_orchestrator(
        questions=[make_question()],
        eval_fail=True,
    )
    result = orch.run(make_context())
    assert result.success is False
    assert result.failure_code == "EVALUATION_FAILED"


def test_refinement_failure_returns_early():
    orch = make_orchestrator(
        questions=[make_question()],
        verdicts=[SlotVerdict(slot_id="Q1", verdict="fail", source="api",
                              reason_code="CIRCULAR")],
        refine_fail=True,
    )
    result = orch.run(make_context())
    assert result.success is False
    assert result.failure_code == "REFINEMENT_FAILED"


def test_checking_failure_returns_early():
    orch = make_orchestrator(
        questions=[make_question()],
        check_fail=True,
    )
    result = orch.run(make_context())
    assert result.success is False
    assert result.failure_code == "CHECKING_FAILED"


# -----------------------------------------------------------------------------
# Writing failures become unresolved slots
# -----------------------------------------------------------------------------


def test_writing_failures_propagate_to_unresolved():
    class F:
        def __init__(self, slot_id):
            self.slot_id = slot_id
            self.failure_code = "GENERATION_FAILED"
            self.failure_detail = "..."

    orch = make_orchestrator(
        questions=[make_question("Q1")],
        writing_failures=[F("Q2"), F("Q3")],
        status=STATUS_BLOCKED,
        block_reasons=[BLOCK_REASON_UNRESOLVED_SLOTS],
    )
    ctx = make_context()
    result = orch.run(ctx)
    assert "Q2" in ctx.unresolved_slots
    assert "Q3" in ctx.unresolved_slots
    assert result.qa_report.status == STATUS_BLOCKED


# -----------------------------------------------------------------------------
# Optional agents
# -----------------------------------------------------------------------------


def test_no_evaluation_agent_skips_evaluation():
    orch = make_orchestrator(
        questions=[make_question()],
        include_eval=False,
        include_refine=False,
    )
    ctx = make_context()
    result = orch.run(ctx)
    assert result.success is True
    assert ctx.audit_report is None
    assert result.audit_report is None


def test_no_refinement_agent_skips_refinement():
    orch = make_orchestrator(
        questions=[make_question()],
        verdicts=[SlotVerdict(slot_id="Q1", verdict="pass", source="api")],
        include_refine=False,
    )
    result = orch.run(make_context())
    assert result.success is True


# -----------------------------------------------------------------------------
# Blocked paper handling
# -----------------------------------------------------------------------------


def test_blocked_paper_returns_blocked_status():
    orch = make_orchestrator(
        questions=[make_question("Q1")],
        status=STATUS_BLOCKED,
        block_reasons=[BLOCK_REASON_UNRESOLVED_SLOTS],
    )
    result = orch.run(make_context())
    assert result.success is True
    assert result.paper_blocked is True
    assert result.paper_passed is False


def test_run_or_raise_raises_on_blocked(monkeypatch):
    monkeypatch.setenv("AION_ENABLE_UNRESOLVED_HARD_BLOCK", "true")
    orch = make_orchestrator(
        questions=[make_question("Q1")],
        status=STATUS_BLOCKED,
        block_reasons=[BLOCK_REASON_UNRESOLVED_SLOTS],
    )
    with pytest.raises(PaperBlockedError) as exc_info:
        orch.run_or_raise(make_context())
    assert exc_info.value.report.status == STATUS_BLOCKED


def test_run_or_raise_does_not_raise_when_hard_block_disabled(monkeypatch):
    monkeypatch.setenv("AION_ENABLE_UNRESOLVED_HARD_BLOCK", "false")
    orch = make_orchestrator(
        questions=[make_question("Q1")],
        status=STATUS_BLOCKED,
        block_reasons=[BLOCK_REASON_UNRESOLVED_SLOTS],
    )
    result = orch.run_or_raise(make_context())
    assert result.success is True
    assert result.paper_blocked is True


def test_run_or_raise_does_not_raise_on_pass():
    orch = make_orchestrator(
        questions=[make_question("Q1")],
        status=STATUS_PASS,
    )
    result = orch.run_or_raise(make_context())
    assert result.paper_passed is True


# -----------------------------------------------------------------------------
# Telemetry aggregation
# -----------------------------------------------------------------------------


def test_telemetry_from_all_agents_aggregated():
    orch = make_orchestrator(
        plans=[QuestionPlan(slot_id="Q1", module_id="module_1", module_idx=1,
                            slot_idx=0, global_q_idx=1, partition=[10], total_marks=10)],
        questions=[make_question("Q1")],
        verdicts=[SlotVerdict(slot_id="Q1", verdict="pass", source="api")],
        status=STATUS_PASS,
    )
    result = orch.run(make_context())
    telem = result.telemetry
    assert "planning" in telem
    assert "writing" in telem
    assert "evaluation" in telem
    assert "refinement" in telem
    assert "checking" in telem
    assert "orchestrator" in telem
    assert telem["orchestrator"]["stages_completed"] == 5


# -----------------------------------------------------------------------------
# OrchestratorResult contract
# -----------------------------------------------------------------------------


def test_orchestrator_result_requires_failure_code():
    with pytest.raises(ValueError):
        OrchestratorResult(success=False)


def test_orchestrator_result_requires_qa_report_on_success():
    with pytest.raises(ValueError):
        OrchestratorResult(success=True)


def test_orchestrator_result_helpers():
    report = QAReport(status=STATUS_PASS)
    r = OrchestratorResult(success=True, qa_report=report)
    assert r.paper_passed is True
    assert r.paper_blocked is False
    assert r.paper_degraded is False


def test_orchestrator_result_degraded():
    report = QAReport(status=STATUS_DEGRADED, degraded_reasons=["UNRESOLVED_SLOTS"])
    r = OrchestratorResult(success=True, qa_report=report)
    assert r.paper_degraded is True
    assert r.paper_passed is False


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------


def test_missing_planning_agent_rejected():
    with pytest.raises(ValueError):
        AgentOrchestrator(
            planning_agent=None,
            writing_agent=StubWritingAgent(),
            checking_agent=StubCheckingAgent(),
        )


def test_missing_writing_agent_rejected():
    with pytest.raises(ValueError):
        AgentOrchestrator(
            planning_agent=StubPlanningAgent(),
            writing_agent=None,
            checking_agent=StubCheckingAgent(),
        )


def test_missing_checking_agent_rejected():
    with pytest.raises(ValueError):
        AgentOrchestrator(
            planning_agent=StubPlanningAgent(),
            writing_agent=StubWritingAgent(),
            checking_agent=None,
        )


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_same_inputs_same_result():
    def run_once():
        orch = make_orchestrator(
            questions=[make_question("Q1"), make_question("Q2")],
            verdicts=[
                SlotVerdict(slot_id="Q1", verdict="pass", source="api"),
                SlotVerdict(slot_id="Q2", verdict="pass", source="api"),
            ],
        )
        return orch.run(make_context())

    r1 = run_once()
    r2 = run_once()
    assert r1.success == r2.success
    assert r1.qa_report.status == r2.qa_report.status
    assert len(r1.questions) == len(r2.questions)
