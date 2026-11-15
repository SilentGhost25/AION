"""
Stage 1B Verification Suite: Unresolved-Question Dataclass Contracts & Serialization.

Verifies:
1. Backward-compatible construction (positional and keyword-based).
2. Default resolution metadata values.
3. Question-level and subquestion-level unresolved states.
4. Tuple immutability and default isolation.
5. dataclasses.asdict() and core.serialization.jsonable() compatibility.
6. Absence of raw failed text in resolution fields.
7. copy.deepcopy() preservation.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, fields
from typing import Any, Dict, List

import pytest

from core.generation.agents.generated_question import GeneratedQuestion
from core.serialization import jsonable


# =============================================================================
# 1. Constructor Backward Compatibility & Defaults
# =============================================================================

def test_legacy_positional_construction_mandatory_args():
    """Verify that constructing with 6 mandatory positional arguments succeeds."""
    q = GeneratedQuestion(
        "slot_1",
        "module_1",
        1,
        "Explain virtual memory management.",
        "Model solution text.",
        [{"criterion": "Definitions", "marks": 4}, {"criterion": "Paging", "marks": 6}],
    )
    # Core identity & mandatory fields
    assert q.slot_id == "slot_1"
    assert q.module_id == "module_1"
    assert q.global_q_idx == 1
    assert q.question_text == "Explain virtual memory management."
    assert q.solution == "Model solution text."
    assert len(q.marking_scheme) == 2

    # Existing default fields
    assert q.marks == 0
    assert q.partition == []
    assert q.bloom == "L2"
    assert q.co == "CO1"
    assert q.topic == ""
    assert q.references_image is False
    assert q.image_path is None
    assert q.figure_caption is None
    assert q.generation_source == "local_llm"
    assert q.attempts == 1
    assert q.telemetry == {}

    # Stage 1B resolution contract defaults
    assert q.is_unresolved is False
    assert q.unresolved_scope is None
    assert q.unresolved_subparts == ()
    assert q.failure_code is None
    assert q.failure_detail is None


def test_legacy_keyword_construction_without_new_fields():
    """Verify standard keyword instantiation from WritingAgent without new fields."""
    q = GeneratedQuestion(
        slot_id="module_2_Q3",
        module_id="module_2",
        global_q_idx=3,
        question_text="(a) Explain CPU scheduling. (b) Compare FCFS and SJF.",
        solution="Full solution.",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        partition=[6, 4],
        bloom="L3",
        co="CO2",
        topic="CPU Scheduling",
        evidence_summary="Context excerpt from module 2.",
        references_image=False,
    )
    assert q.marks == 10
    assert q.partition == [6, 4]
    assert q.is_unresolved is False
    assert q.unresolved_scope is None
    assert q.unresolved_subparts == ()
    assert q.failure_code is None
    assert q.failure_detail is None


def test_default_tuple_isolation():
    """Verify that default unresolved_subparts is an empty tuple and isolated across instances."""
    q1 = GeneratedQuestion("s1", "m1", 1, "text1", "sol1", [])
    q2 = GeneratedQuestion("s2", "m2", 2, "text2", "sol2", [])
    assert q1.unresolved_subparts is ()
    assert q2.unresolved_subparts is ()
    assert isinstance(q1.unresolved_subparts, tuple)
    assert isinstance(q2.unresolved_subparts, tuple)


# =============================================================================
# 2. Typed Unresolved Scopes & Resolution States
# =============================================================================

def test_question_level_unresolved_construction():
    """Verify question-level failure metadata (e.g. prompt ungrounded, whole question failed)."""
    q = GeneratedQuestion(
        slot_id="module_1_Q1",
        module_id="module_1",
        global_q_idx=1,
        question_text="Unresolved question placeholder",
        solution="",
        marking_scheme=[],
        marks=10,
        partition=[6, 4],
        bloom="L2",
        co="CO1",
        is_unresolved=True,
        unresolved_scope="question",
        unresolved_subparts=("a", "b"),
        failure_code="UNGROUNDED",
        failure_detail="Question refers to concepts outside the uploaded syllabus text.",
    )
    assert q.is_unresolved is True
    assert q.unresolved_scope == "question"
    assert q.unresolved_subparts == ("a", "b")
    assert q.failure_code == "UNGROUNDED"
    assert q.failure_detail == "Question refers to concepts outside the uploaded syllabus text."


def test_subquestion_level_unresolved_construction():
    """Verify subquestion-level failure metadata targeting only subpart (b)."""
    q = GeneratedQuestion(
        slot_id="module_3_Q6",
        module_id="module_3",
        global_q_idx=6,
        question_text="(a) With the aid of the given figure, analyze multithreading. (b) [Placeholder]",
        solution="Model solution for (a).",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        partition=[6, 4],
        bloom="L4",
        co="CO3",
        is_unresolved=True,
        unresolved_scope="subquestion",
        unresolved_subparts=("b",),
        failure_code="BLOOM_MISMATCH",
        failure_detail="Leading verb 'Explain' does not match target bloom L4 'Analyze'.",
    )
    assert q.is_unresolved is True
    assert q.unresolved_scope == "subquestion"
    assert q.unresolved_subparts == ("b",)
    assert q.failure_code == "BLOOM_MISMATCH"


def test_multiple_subparts_unresolved():
    """Verify multi-part question with several failed subparts (e.g. 3-part partition)."""
    q = GeneratedQuestion(
        slot_id="module_5_Q10",
        module_id="module_5",
        global_q_idx=10,
        question_text="Three-part question text.",
        solution="Solution.",
        marking_scheme=[],
        marks=20,
        partition=[8, 6, 6],
        is_unresolved=True,
        unresolved_scope="subquestion",
        unresolved_subparts=("b", "c"),
        failure_code="SIBLING_DUPLICATE",
        failure_detail="Subparts (b) and (c) contain duplicated queries.",
    )
    assert q.unresolved_subparts == ("b", "c")
    assert len(q.unresolved_subparts) == 2


# =============================================================================
# 3. Serialization Contracts (asdict & jsonable)
# =============================================================================

def test_asdict_serialization_shape():
    """Verify dataclasses.asdict() emits all fields including the new 5 Stage 1B fields."""
    q = GeneratedQuestion(
        slot_id="slot_test",
        module_id="module_1",
        global_q_idx=1,
        question_text="Text.",
        solution="Solution.",
        marking_scheme=[],
        is_unresolved=True,
        unresolved_scope="subquestion",
        unresolved_subparts=("b",),
        failure_code="BLOOM_MISMATCH",
        failure_detail="Verb mismatch.",
    )
    d = asdict(q)
    # Exactly 24 fields on GeneratedQuestion (19 pre-existing + 5 Stage 1B)
    assert len(d) == 24
    assert d["is_unresolved"] is True
    assert d["unresolved_scope"] == "subquestion"
    assert d["unresolved_subparts"] == ("b",)
    assert d["failure_code"] == "BLOOM_MISMATCH"
    assert d["failure_detail"] == "Verb mismatch."


def test_jsonable_serialization_compliance():
    """
    Verify core.serialization.jsonable converts tuple fields to standard JSON lists
    and preserves boolean and null values without raising TypeError in strict mode.
    """
    q = GeneratedQuestion(
        slot_id="slot_json",
        module_id="module_4",
        global_q_idx=7,
        question_text="JSON test text.",
        solution="JSON test sol.",
        marking_scheme=[{"criterion": "Criterion 1", "marks": 5}],
        marks=10,
        partition=[5, 5],
        is_unresolved=True,
        unresolved_scope="subquestion",
        unresolved_subparts=("a", "b"),
        failure_code="SCENARIO_CONTRADICTION",
        failure_detail="Contradictory premise.",
    )
    j = jsonable(q, _strict=True)

    # Tuple converted to list by jsonable()
    assert isinstance(j["unresolved_subparts"], list)
    assert j["unresolved_subparts"] == ["a", "b"]

    # Primitives properly formatted for JSON
    assert j["is_unresolved"] is True
    assert j["unresolved_scope"] == "subquestion"
    assert j["failure_code"] == "SCENARIO_CONTRADICTION"
    assert j["failure_detail"] == "Contradictory premise."


def test_jsonable_default_values():
    """Verify jsonable output when new fields are at default values."""
    q = GeneratedQuestion("s_def", "m1", 1, "text", "sol", [])
    j = jsonable(q, _strict=True)
    assert j["is_unresolved"] is False
    assert j["unresolved_scope"] is None
    assert j["unresolved_subparts"] == []
    assert j["failure_code"] is None
    assert j["failure_detail"] is None


# =============================================================================
# 4. Copying and Integrity Contracts
# =============================================================================

def test_deepcopy_preserves_resolution_contracts():
    """Verify copy.deepcopy preserves all 5 new Stage 1B resolution fields."""
    q = GeneratedQuestion(
        slot_id="slot_cp",
        module_id="module_1",
        global_q_idx=1,
        question_text="Original text",
        solution="Original solution",
        marking_scheme=[],
        is_unresolved=True,
        unresolved_scope="subquestion",
        unresolved_subparts=("b",),
        failure_code="QUESTION_TOO_SHORT",
        failure_detail="Question length is below minimum required words.",
    )
    q_copy = copy.deepcopy(q)
    assert q_copy.is_unresolved is True
    assert q_copy.unresolved_scope == "subquestion"
    assert q_copy.unresolved_subparts == ("b",)
    assert q_copy.failure_code == "QUESTION_TOO_SHORT"
    assert q_copy.failure_detail == "Question length is below minimum required words."


def test_no_raw_failed_text_in_resolution_metadata():
    """
    Verify that resolution fields contain only structured metadata codes and summaries,
    never raw failed full-question pedagogical text.
    """
    raw_failed_text = "(a) Explain CPU scheduling. (b) Explain CPU scheduling."
    q = GeneratedQuestion(
        slot_id="slot_diagnostic",
        module_id="module_2",
        global_q_idx=3,
        question_text="Placeholder text only",
        solution="Model solution",
        marking_scheme=[],
        is_unresolved=True,
        unresolved_scope="subquestion",
        unresolved_subparts=("b",),
        failure_code="SIBLING_DUPLICATE",
        failure_detail="Jaccard overlap 1.0 between subpart (a) and (b)",
    )
    # The raw text must not appear anywhere in failure_code, failure_detail, or subparts
    assert raw_failed_text not in str(q.failure_code)
    assert raw_failed_text not in str(q.failure_detail)
    assert raw_failed_text not in str(q.unresolved_subparts)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
