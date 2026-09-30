"""
Unit tests for the v3 pipeline bridge.

Uses a stub DependencyFactory and stub legacy_runner to avoid real LLMs.
"""

import os
from dataclasses import dataclass
from typing import Any, List

import pytest

from core.generation.agents import (
    AgentContext,
    AgentResult,
    AgentOrchestrator,
    GeneratedQuestion,
    OrchestratorResult,
    PaperBlockedError,
    QAReport,
    QuestionPlan,
    STATUS_BLOCKED,
    STATUS_PASS,
)
from core.generation.agents.pipeline_bridge import (
    V3_FLAG_ENV,
    run_pipeline_v3_or_legacy,
    run_v3_pipeline,
    v3_enabled,
)


# -----------------------------------------------------------------------------
# Stub agents
# -----------------------------------------------------------------------------


class StubPlan:
    name = "planning"
    def __init__(self, plans):
        self._plans = plans
    def run(self, ctx):
        return AgentResult(success=True, payload=self._plans, telemetry={"n": len(self._plans)})


class StubWrite:
    name = "writing"
    def __init__(self, questions):
        self._questions = questions
    def run(self, ctx):
        class Out:
            def __init__(self, q): self.questions = q; self.failures = []
        return AgentResult(success=True, payload=Out(self._questions), telemetry={})


class StubCheck:
    name = "checking"
    def __init__(self, status=STATUS_PASS):
        self._status = status
    def run(self, ctx):
        blocked = ["UNRESOLVED_SLOTS"] if self._status == STATUS_BLOCKED else []
        r = QAReport(status=self._status,
                     slot_count=len(ctx.questions),
                     resolved_slots=len(ctx.questions),
                     blocked_reasons=blocked)
        return AgentResult(success=True, payload=r, telemetry={"status": self._status})


class StubFactory:
    def __init__(self, questions, status=STATUS_PASS):
        self._questions = questions
        self._status = status
    def build(self):
        from core.generation.agents.dependency_factory import FactoryResult
        return FactoryResult(
            success=True,
            planning=StubPlan([]),
            writing=StubWrite(self._questions),
            evaluation=None,
            refinement=None,
            checking=StubCheck(self._status),
            notes=["stub factory"],
        )


class FailingFactory:
    def build(self):
        from core.generation.agents.dependency_factory import FactoryResult
        return FactoryResult(success=False,
                             failure_code="LLM_CALLER_UNAVAILABLE",
                             failure_detail="no model")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def make_question(slot_id="module_1_Q1", module_id="module_1"):
    return GeneratedQuestion(
        slot_id=slot_id,
        module_id=module_id,
        global_q_idx=1,
        question_text="Explain Docker.",
        solution="Docker is a platform.",
        marking_scheme=[{"criterion": "x", "marks": 10}],
        marks=10, partition=[10], bloom="L2", co="CO1", topic="Docker",
    )


@dataclass
class StubSpec:
    module_count: int = 1
    total_questions: int = 1
    marks_per_question: int = 10


# -----------------------------------------------------------------------------
# Flag detection
# -----------------------------------------------------------------------------


def test_v3_enabled_default_false(monkeypatch):
    monkeypatch.delenv(V3_FLAG_ENV, raising=False)
    assert v3_enabled() is False


@pytest.mark.parametrize("val,expected", [
    ("true", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("0", False), ("", False), ("off", False),
])
def test_v3_enabled_parses_env(monkeypatch, val, expected):
    monkeypatch.setenv(V3_FLAG_ENV, val)
    assert v3_enabled() is expected


# -----------------------------------------------------------------------------
# Direct v3 run
# -----------------------------------------------------------------------------


def test_v3_pipeline_returns_legacy_shape():
    q = make_question()
    parts, full, meta = run_v3_pipeline(
        paper_spec=StubSpec(),
        artifact=None,
        marks_split=[[10]],
        dependency_factory=StubFactory([q]),
    )
    assert meta["success"] is True
    assert meta["paper_status"] == STATUS_PASS
    assert len(full) == 1
    assert full[0]["slot_id"] == "module_1_Q1"
    assert full[0]["question_text"] == "Explain Docker."
    assert len(parts) == 1
    assert parts[0]["module_id"] == "module_1"
    assert parts[0]["questions"][0]["slot_id"] == "module_1_Q1"


def test_v3_pipeline_meta_contains_telemetry():
    q = make_question()
    _, _, meta = run_v3_pipeline(
        paper_spec=StubSpec(),
        artifact=None,
        marks_split=[[10]],
        dependency_factory=StubFactory([q]),
    )
    assert "telemetry" in meta
    assert "qa_report" in meta
    assert meta["qa_report"]["status"] == STATUS_PASS


def test_v3_pipeline_factory_failure_returns_empty():
    parts, full, meta = run_v3_pipeline(
        paper_spec=StubSpec(),
        artifact=None,
        marks_split=[[10]],
        dependency_factory=FailingFactory(),
    )
    assert parts == []
    assert full == []
    assert meta["success"] is False
    assert meta["failure_code"] == "LLM_CALLER_UNAVAILABLE"


def test_v3_pipeline_blocked_status_preserved():
    q = make_question()
    parts, full, meta = run_v3_pipeline(
        paper_spec=StubSpec(),
        artifact=None,
        marks_split=[[10]],
        dependency_factory=StubFactory([q], status=STATUS_BLOCKED),
    )
    # Blocked still returns questions but meta says BLOCKED
    assert meta["success"] is True
    assert meta["paper_status"] == STATUS_BLOCKED


def test_v3_pipeline_attaches_qa_report_to_parts():
    q = make_question()
    parts, _, _ = run_v3_pipeline(
        paper_spec=StubSpec(),
        artifact=None,
        marks_split=[[10]],
        dependency_factory=StubFactory([q]),
    )
    assert "qa_report" in parts[0]
    assert parts[0]["qa_report"]["status"] == STATUS_PASS


# -----------------------------------------------------------------------------
# Flag-driven dispatch
# -----------------------------------------------------------------------------


def test_flag_off_uses_legacy(monkeypatch):
    monkeypatch.setenv(V3_FLAG_ENV, "false")
    legacy_calls = []

    def legacy_runner(**kwargs):
        legacy_calls.append(kwargs)
        return ([{"legacy": True}], [{"legacy": True}])

    parts, full = run_pipeline_v3_or_legacy(
        legacy_runner=legacy_runner,
        v3_kwargs={},
        legacy_kwargs={"exam_type": "ia"},
    )
    assert len(legacy_calls) == 1
    assert parts == [{"legacy": True}]


def test_flag_on_uses_v3(monkeypatch):
    monkeypatch.setenv(V3_FLAG_ENV, "true")
    q = make_question()

    def legacy_runner(**kwargs):
        raise AssertionError("legacy should not be called when flag is on")

    parts, full = run_pipeline_v3_or_legacy(
        legacy_runner=legacy_runner,
        v3_kwargs={
            "paper_spec": StubSpec(),
            "artifact": None,
            "marks_split": [[10]],
            "dependency_factory": StubFactory([q]),
        },
        legacy_kwargs={},
    )
    assert len(full) == 1
    assert full[0]["slot_id"] == "module_1_Q1"


def test_flag_on_v3_failure_returns_empty(monkeypatch):
    monkeypatch.setenv(V3_FLAG_ENV, "true")
    parts, full = run_pipeline_v3_or_legacy(
        legacy_runner=lambda **kw: ([{"legacy": 1}], [{"legacy": 1}]),
        v3_kwargs={
            "paper_spec": StubSpec(),
            "artifact": None,
            "marks_split": [[10]],
            "dependency_factory": FailingFactory(),
        },
        legacy_kwargs={},
    )
    # v3 failed; do not fall back to legacy
    assert parts == []
    assert full == []


def test_flag_on_attaches_meta_to_first_part(monkeypatch):
    monkeypatch.setenv(V3_FLAG_ENV, "true")
    q = make_question()
    parts, _ = run_pipeline_v3_or_legacy(
        legacy_runner=lambda **kw: ([], []),
        v3_kwargs={
            "paper_spec": StubSpec(),
            "artifact": None,
            "marks_split": [[10]],
            "dependency_factory": StubFactory([q]),
        },
        legacy_kwargs={},
    )
    assert "_orchestrator_meta" in parts[0]
    assert parts[0]["_orchestrator_meta"]["success"] is True
