"""
Phase 5 — Determinism verification.

Proves that the v3 pipeline produces byte-identical output for the
same inputs and same seed. Uses mock callers so no external factors
introduce non-determinism.
"""

import copy
import pytest

from core.generation.agents.pipeline_bridge import run_v3_pipeline
from tests.fixtures.v3_mock_callers import (
    MockAPICaller,
    MockLLMCaller,
    MockSympyVerifier,
    MockVLMCaller,
)
from tests.fixtures.v3_paper_specs import (
    build_test_catalog,
    IAT1_SPEC,
    IAT1_MARKS_SPLIT,
    make_fake_artifact,
)


def build_stub_factory():
    from core.generation.agents.dependency_factory import DependencyFactory
    return DependencyFactory(
        llm_caller_factory=lambda: MockLLMCaller(mode="normal"),
        vlm_caller_factory=lambda: MockVLMCaller(mode="normal"),
        api_caller_factory=lambda: MockAPICaller(mode="all_pass"),
        sympy_verifier_factory=lambda: MockSympyVerifier(result=True),
    )


def _question_signature(q: dict) -> tuple:
    """Extract a canonical signature that ignores dict ordering."""
    return (
        q["slot_id"],
        q["module_id"],
        q["question_text"],
        q["solution"],
        tuple((e.get("criterion"), e.get("marks")) for e in q["marking_scheme"]),
        q["marks"],
        q["bloom"],
        q["co"],
        q.get("image_path"),
    )


# -----------------------------------------------------------------------------
# Determinism tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("subject", build_test_catalog(), ids=lambda s: s.name)
def test_v3_deterministic_full_paper(subject):
    """Same inputs → same questions, twice."""
    def run_once():
        _, full, _ = run_v3_pipeline(
            paper_spec=subject.spec,
            artifact=subject.artifact,
            marks_split=subject.marks_split,
            dependency_factory=build_stub_factory(),
        )
        return full

    run_a = run_once()
    run_b = run_once()

    assert len(run_a) == len(run_b)
    for qa, qb in zip(run_a, run_b):
        assert _question_signature(qa) == _question_signature(qb), (
            f"slot {qa['slot_id']} differs between runs"
        )


def test_v3_deterministic_paper_status():
    """Same inputs → same paper status."""
    subject = build_test_catalog()[0]

    def status_once():
        _, _, meta = run_v3_pipeline(
            paper_spec=subject.spec,
            artifact=subject.artifact,
            marks_split=subject.marks_split,
            dependency_factory=build_stub_factory(),
        )
        return meta["paper_status"]

    assert status_once() == status_once()


def test_v3_deterministic_telemetry_shape():
    """Telemetry keys are identical between runs (values may include timing)."""
    subject = build_test_catalog()[0]

    def keys_once():
        _, _, meta = run_v3_pipeline(
            paper_spec=subject.spec,
            artifact=subject.artifact,
            marks_split=subject.marks_split,
            dependency_factory=build_stub_factory(),
        )
        return sorted(meta["telemetry"].keys())

    assert keys_once() == keys_once()


def test_v3_mock_llm_is_deterministic():
    """Sanity check: the mock itself is deterministic."""
    llm1 = MockLLMCaller(mode="normal")
    llm2 = MockLLMCaller(mode="normal")
    prompt = "[SLOT CONTRACT]\nMarks: 10\nBloom level: L2\napproved L2 verb: explain"
    r1 = llm1.call(prompt)
    r2 = llm2.call(prompt)
    assert r1 == r2


# -----------------------------------------------------------------------------
# Ordering guarantees
# -----------------------------------------------------------------------------


def test_v3_question_order_stable():
    """Questions are produced in slot order across runs."""
    subject = build_test_catalog()[0]

    def slot_order():
        _, full, _ = run_v3_pipeline(
            paper_spec=subject.spec,
            artifact=subject.artifact,
            marks_split=subject.marks_split,
            dependency_factory=build_stub_factory(),
        )
        return [q["slot_id"] for q in full]

    order_1 = slot_order()
    order_2 = slot_order()
    assert order_1 == order_2


def test_v3_modules_in_order():
    """Paper parts are ordered by module index."""
    subject = build_test_catalog()[0]
    parts, _, _ = run_v3_pipeline(
        paper_spec=subject.spec,
        artifact=subject.artifact,
        marks_split=subject.marks_split,
        dependency_factory=build_stub_factory(),
    )
    indices = [p["module_index"] for p in parts]
    assert indices == sorted(indices)


# -----------------------------------------------------------------------------
# Isolation between runs
# -----------------------------------------------------------------------------


def test_v3_inputs_not_mutated():
    """The pipeline does not mutate its inputs."""
    subject = build_test_catalog()[0]
    spec_before = copy.deepcopy(subject.spec)
    split_before = copy.deepcopy(subject.marks_split)
    artifact_blocks_before = len(subject.artifact.text_blocks)

    run_v3_pipeline(
        paper_spec=subject.spec,
        artifact=subject.artifact,
        marks_split=subject.marks_split,
        dependency_factory=build_stub_factory(),
    )

    assert subject.spec == spec_before
    assert subject.marks_split == split_before
    assert len(subject.artifact.text_blocks) == artifact_blocks_before
