"""
Phase 5 — Multi-subject acceptance with v3 enabled.

Runs the v3 pipeline end-to-end on four subjects using mock callers.
No network, no LLM, no GPU. Every assertion is deterministic.
"""

import os
from typing import Any

import pytest

from core.generation.agents import (
    AgentOrchestrator,
    GeneratedQuestion,
    STATUS_BLOCKED,
    STATUS_DEGRADED,
    STATUS_PASS,
)
from core.generation.agents.base import AgentContext
from core.generation.agents.dependency_factory import DependencyFactory, FactoryResult
from core.generation.agents.pipeline_bridge import run_v3_pipeline
from tests.fixtures.v3_mock_callers import (
    MockAPICaller,
    MockLLMCaller,
    MockSympyVerifier,
    MockVLMCaller,
)
from tests.fixtures.v3_paper_specs import (
    TestSubject,
    build_test_catalog,
)


# -----------------------------------------------------------------------------
# Fixture: build a stubbed DependencyFactory per test
# -----------------------------------------------------------------------------


def build_stub_factory(
    llm_mode="normal",
    vlm_mode="normal",
    api_mode="all_pass",
    fail_on_nth=None,
) -> DependencyFactory:
    """Build a DependencyFactory that returns mock callers."""
    def make_llm():
        return MockLLMCaller(mode=llm_mode, fail_on_nth=fail_on_nth)

    def make_vlm():
        return MockVLMCaller(mode=vlm_mode)

    def make_api():
        return MockAPICaller(mode=api_mode)

    def make_sympy():
        return MockSympyVerifier(result=True)

    return DependencyFactory(
        llm_caller_factory=make_llm,
        vlm_caller_factory=make_vlm,
        api_caller_factory=make_api,
        sympy_verifier_factory=make_sympy,
    )


# -----------------------------------------------------------------------------
# Shared test: every subject
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("subject", build_test_catalog(), ids=lambda s: s.name)
def test_v3_pipeline_completes(subject: TestSubject):
    """The v3 pipeline runs to completion on every subject."""
    parts, full, meta = run_v3_pipeline(
        paper_spec=subject.spec,
        artifact=subject.artifact,
        marks_split=subject.marks_split,
        dependency_factory=build_stub_factory(),
    )

    assert meta["success"] is True, f"pipeline failed: {meta}"
    assert len(full) == subject.spec.total_questions
    assert len(parts) == subject.spec.module_count


@pytest.mark.parametrize("subject", build_test_catalog(), ids=lambda s: s.name)
def test_v3_paper_status_not_blocked(subject: TestSubject):
    """The Checking Agent must not block a fully-successful run."""
    _, _, meta = run_v3_pipeline(
        paper_spec=subject.spec,
        artifact=subject.artifact,
        marks_split=subject.marks_split,
        dependency_factory=build_stub_factory(),
    )
    assert meta["paper_status"] in (STATUS_PASS, STATUS_DEGRADED)
    assert meta["paper_status"] != STATUS_BLOCKED


@pytest.mark.parametrize("subject", build_test_catalog(), ids=lambda s: s.name)
def test_v3_marks_distribution_matches_split(subject: TestSubject):
    """Every question's marks equal its assigned partition."""
    _, full, _ = run_v3_pipeline(
        paper_spec=subject.spec,
        artifact=subject.artifact,
        marks_split=subject.marks_split,
        dependency_factory=build_stub_factory(),
    )
    for idx, question in enumerate(full):
        expected_total = sum(subject.marks_split[idx])
        assert question["marks"] == expected_total, (
            f"slot {question['slot_id']} has marks={question['marks']}, "
            f"expected {expected_total}"
        )


@pytest.mark.parametrize("subject", build_test_catalog(), ids=lambda s: s.name)
def test_v3_co_distribution_matches_module(subject: TestSubject):
    """Each question's CO matches its module's CO mapping."""
    _, full, _ = run_v3_pipeline(
        paper_spec=subject.spec,
        artifact=subject.artifact,
        marks_split=subject.marks_split,
        dependency_factory=build_stub_factory(),
    )
    for question in full:
        module_idx = int(question["module_id"].split("_")[1])
        # Default mapping: module M → CO(min(M, co_count))
        expected_co = f"CO{min(module_idx, subject.spec.co_count)}"
        assert question["co"] == expected_co, (
            f"slot {question['slot_id']} has co={question['co']}, "
            f"expected {expected_co}"
        )


@pytest.mark.parametrize("subject", build_test_catalog(), ids=lambda s: s.name)
def test_v3_questions_have_required_fields(subject: TestSubject):
    """Every question has the fields the DOCX export expects."""
    _, full, _ = run_v3_pipeline(
        paper_spec=subject.spec,
        artifact=subject.artifact,
        marks_split=subject.marks_split,
        dependency_factory=build_stub_factory(),
    )
    required = [
        "slot_id", "module_id", "question_text", "solution",
        "marking_scheme", "marks", "bloom", "co",
    ]
    for question in full:
        for key in required:
            assert key in question, f"slot {question.get('slot_id')} missing {key}"
            assert question[key] is not None, (
                f"slot {question.get('slot_id')} has None for {key}"
            )


@pytest.mark.parametrize("subject", build_test_catalog(), ids=lambda s: s.name)
def test_v3_marking_schemes_sum_correctly(subject: TestSubject):
    """Each question's marking scheme sums to its total marks."""
    _, full, _ = run_v3_pipeline(
        paper_spec=subject.spec,
        artifact=subject.artifact,
        marks_split=subject.marks_split,
        dependency_factory=build_stub_factory(),
    )
    for question in full:
        total = sum(e.get("marks", 0) for e in question["marking_scheme"])
        assert total == question["marks"], (
            f"slot {question['slot_id']}: marking sums to {total}, "
            f"question marks = {question['marks']}"
        )


# -----------------------------------------------------------------------------
# Visual binding tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("subject", build_test_catalog(), ids=lambda s: s.name)
def test_v3_visual_count_within_quota(subject: TestSubject):
    """
    The Planning Agent caps figures at 1 per module.
    Verify no module has more than 1 visual question.
    """
    _, full, _ = run_v3_pipeline(
        paper_spec=subject.spec,
        artifact=subject.artifact,
        marks_split=subject.marks_split,
        dependency_factory=build_stub_factory(),
    )
    per_module: dict = {}
    for q in full:
        if q.get("image_path"):
            per_module.setdefault(q["module_id"], 0)
            per_module[q["module_id"]] += 1
    for module_id, count in per_module.items():
        assert count <= 1, f"{module_id} has {count} visual questions, max is 1"


# -----------------------------------------------------------------------------
# Failure-path tests
# -----------------------------------------------------------------------------


def test_v3_llm_failure_blocks_paper():
    """If the LLM caller consistently fails, the paper is blocked."""
    subject = build_test_catalog()[0]  # Sat Com
    _, full, meta = run_v3_pipeline(
        paper_spec=subject.spec,
        artifact=subject.artifact,
        marks_split=subject.marks_split,
        dependency_factory=build_stub_factory(llm_mode="fail"),
    )
    # Writing agent produces no questions → checking blocks
    assert meta["success"] is True  # orchestrator didn't crash
    assert meta["paper_status"] == STATUS_BLOCKED
    assert len(full) < subject.spec.total_questions


def test_v3_api_unavailable_degrades():
    """If the API is unavailable, the paper degrades but still ships."""
    subject = build_test_catalog()[0]
    _, full, meta = run_v3_pipeline(
        paper_spec=subject.spec,
        artifact=subject.artifact,
        marks_split=subject.marks_split,
        dependency_factory=build_stub_factory(api_mode="unavailable"),
    )
    assert meta["success"] is True
    assert meta["paper_status"] == STATUS_DEGRADED
    assert len(full) == subject.spec.total_questions


def test_v3_factory_failure_returns_empty():
    """A factory that fails to build agents returns an empty paper."""
    from tests.fixtures.v3_paper_specs import IAT1_SPEC, IAT1_MARKS_SPLIT, make_fake_artifact

    class BrokenFactory:
        def build(self):
            return FactoryResult(
                success=False,
                failure_code="LLM_CALLER_UNAVAILABLE",
                failure_detail="no model",
            )

    parts, full, meta = run_v3_pipeline(
        paper_spec=IAT1_SPEC,
        artifact=make_fake_artifact(5),
        marks_split=IAT1_MARKS_SPLIT,
        dependency_factory=BrokenFactory(),
    )
    assert parts == []
    assert full == []
    assert meta["success"] is False
