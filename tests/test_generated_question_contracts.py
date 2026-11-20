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

from core.generation.agents.generated_question import GeneratedQuestion, SubPart
from core.generation.agents.legacy_adapter import _extract_subpart_structure
from core.generation.agents.local_evaluators import _PART_LABEL
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
    # Exactly 25 fields on GeneratedQuestion (24 pre-existing + 1 sub_parts)
    assert len(d) == 25
    assert "sub_parts" in d
    assert d["sub_parts"] == []
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




# =============================================================================
# 5. Stage G1-A Acceptance Contract Tests (22 Tests)
# =============================================================================

def test_subpart_dataclass_defaults_and_validation():
    """12. Verify valid construction of SubPart and its defaults."""
    sp = SubPart(label="a", marks=6, text="Explain process scheduling algorithms.")
    assert sp.label == "a"
    assert sp.marks == 6
    assert sp.text == "Explain process scheduling algorithms."
    assert sp.topic == ""
    assert sp.solution == ""
    assert sp.marking_scheme == []
    valid, err = sp.validate()
    assert valid is True
    assert err == ""


def test_subpart_rejects_non_string_types():
    """13. SubPart must reject non-string label, text, topic, and solution."""
    with pytest.raises(TypeError, match="SubPart.label must be exact str"):
        SubPart(label=1, marks=6, text="Some text")  # type: ignore

    with pytest.raises(TypeError, match="SubPart.text must be exact str"):
        SubPart(label="a", marks=6, text=123)  # type: ignore

    with pytest.raises(TypeError, match="SubPart.topic must be exact str"):
        SubPart(label="a", marks=6, text="Text", topic=None)  # type: ignore

    with pytest.raises(TypeError, match="SubPart.solution must be exact str"):
        SubPart(label="a", marks=6, text="Text", solution=123)  # type: ignore

    with pytest.raises(TypeError, match="SubPart.marking_scheme must be exact list"):
        SubPart(label="a", marks=6, text="Text", marking_scheme="not_a_list")  # type: ignore


def test_subpart_rejects_boolean_and_float_marks():
    """14. SubPart marks must be an exact integer, rejecting bool and float."""
    with pytest.raises(TypeError, match="SubPart.marks must be an exact integer"):
        SubPart(label="a", marks=True, text="Text")  # type: ignore

    with pytest.raises(TypeError, match="SubPart.marks must be an exact integer"):
        SubPart(label="a", marks=6.5, text="Text")  # type: ignore


def test_subpart_rejects_zero_or_negative_marks():
    """15. SubPart marks must be strictly positive integer (> 0)."""
    with pytest.raises(ValueError, match="SubPart.marks must be strictly positive"):
        SubPart(label="a", marks=0, text="Text")

    with pytest.raises(ValueError, match="SubPart.marks must be strictly positive"):
        SubPart(label="a", marks=-4, text="Text")


def test_subpart_rejects_uppercase_padded_and_invalid_labels():
    """16. SubPart label must be single lowercase letter 'a'-'z'."""
    for invalid_label in ["A", " a ", "aa", "1", "α", "", " "]:
        with pytest.raises(ValueError, match="SubPart label must be a single lowercase letter"):
            SubPart(label=invalid_label, marks=6, text="Text")


def test_subpart_marking_scheme_criteria_sum_consistency():
    """17. SubPart marking scheme criteria must sum to subpart marks if provided."""
    # Valid criteria breakdown
    sp = SubPart(
        label="a",
        marks=6,
        text="Explain paging.",
        marking_scheme=[{"criterion": "Definition", "marks": 2}, {"criterion": "Diagram", "marks": 4}],
    )
    valid, err = sp.validate()
    assert valid is True

    # Criteria sum mismatch (sum is 5, marks is 6)
    with pytest.raises(ValueError, match=r"criteria sum \(5\) != subpart marks \(6\)"):
        SubPart(
            label="a",
            marks=6,
            text="Explain paging.",
            marking_scheme=[{"criterion": "Def", "marks": 2}, {"criterion": "Diag", "marks": 3}],
        )

    # Non-dict entry
    with pytest.raises(TypeError, match="must be exact dict"):
        SubPart(label="a", marks=6, text="Text", marking_scheme=["not_dict"])  # type: ignore

    # Non-string key inside entry
    with pytest.raises(TypeError, match="key must be exact str"):
        SubPart(label="a", marks=6, text="Text", marking_scheme=[{1: "bad key", "criterion": "c", "marks": 6}])  # type: ignore

    # Missing criterion text
    with pytest.raises(ValueError, match="missing non-empty 'criterion' string"):
        SubPart(label="a", marks=6, text="Text", marking_scheme=[{"criterion": "  ", "marks": 6}])

    # Invalid marks in criterion
    with pytest.raises(TypeError, match=r"\.marks must be exact int"):
        SubPart(label="a", marks=6, text="Text", marking_scheme=[{"criterion": "c", "marks": True}])  # type: ignore

    with pytest.raises(ValueError, match=r"\.marks must be strictly positive"):
        SubPart(label="a", marks=6, text="Text", marking_scheme=[{"criterion": "c", "marks": 0}])


def test_subpart_post_construction_mutation_and_non_mutating_validation():
    """18. validate() detects mutations without mutating unstripped text."""
    sp = SubPart(label="a", marks=6, text="Explain scheduling.")
    assert sp.validate() == (True, "")

    # Mutate to invalid types
    sp.marks = True  # type: ignore
    valid, err = sp.validate()
    assert valid is False
    assert "marks must be an exact integer" in err

    # Restore marks, mutate text to unstripped whitespace
    sp.marks = 6
    sp.text = "  Explain process scheduling.  "
    valid, err = sp.validate()
    assert valid is True
    # Non-mutating contract: validate() must not rewrite sp.text
    assert sp.text == "  Explain process scheduling.  "

    # Mutate text to whitespace-only
    sp.text = "   "
    valid, err = sp.validate()
    assert valid is False
    assert "SubPart text must not be empty" in err


def test_generated_question_rejects_boolean_and_negative_total_marks():
    """19. GeneratedQuestion marks must be strictly positive integer in validate_structure()."""
    q = GeneratedQuestion("s1", "m1", 1, "text", "sol", [], marks=10, partition=[10])
    assert q.validate_structure()[0] is True

    q.marks = True  # type: ignore
    valid, err = q.validate_structure()
    assert valid is False
    assert "marks must be an exact integer" in err

    q.marks = -10
    valid, err = q.validate_structure()
    assert valid is False
    assert "marks must be strictly positive" in err


def test_generated_question_rejects_invalid_partition_containers():
    """20. Partition must be an exact list of positive integers."""
    q = GeneratedQuestion("s1", "m1", 1, "text", "sol", [], marks=10, partition=[10])

    q.partition = "10"  # type: ignore
    valid, err = q.validate_structure()
    assert valid is False
    assert "partition must be an exact list" in err

    q.partition = [6, True]  # type: ignore
    valid, err = q.validate_structure()
    assert valid is False
    assert "partition[1] must be an integer" in err

    q.partition = [6, -4]
    valid, err = q.validate_structure()
    assert valid is False
    assert "partition[1] must be strictly positive" in err


def test_generated_question_rejects_invalid_subpart_containers_and_elements():
    """21. sub_parts must be an exact list of SubPart instances."""
    with pytest.raises(TypeError, match="sub_parts must be exact list"):
        GeneratedQuestion("s1", "m1", 1, "text", "sol", [], sub_parts={"a": 6})  # type: ignore

    with pytest.raises(TypeError, match=r"sub_parts\[0\] must be exact SubPart"):
        GeneratedQuestion("s1", "m1", 1, "text", "sol", [], sub_parts=["not_a_subpart"])  # type: ignore


def test_asdict_and_jsonable_with_subparts():
    """22. asdict() and jsonable() handle SubPart instances properly."""
    sp_a = SubPart(label="a", marks=6, text="Explain scheduling.", topic="CPU", solution="Sol A")
    sp_b = SubPart(label="b", marks=4, text="Compare algorithms.", topic="CPU", solution="Sol B")
    q = GeneratedQuestion(
        slot_id="s1",
        module_id="m1",
        global_q_idx=1,
        question_text="",
        solution="",
        marking_scheme=[],
        marks=10,
        partition=[6, 4],
        sub_parts=[sp_a, sp_b],
    )
    d = asdict(q)
    assert len(d) == 25
    assert len(d["sub_parts"]) == 2
    assert d["sub_parts"][0]["label"] == "a"
    assert d["sub_parts"][0]["marks"] == 6
    assert d["sub_parts"][1]["label"] == "b"

    j = jsonable(q, _strict=True)
    assert len(j["sub_parts"]) == 2
    assert j["sub_parts"][0]["label"] == "a"


def test_from_dict_strict_type_validation():
    """23. from_dict() enforces exact types and rejects bools for ints and non-string keys."""
    with pytest.raises(TypeError, match="from_dict argument must be exact dict"):
        GeneratedQuestion.from_dict("not_a_dict")  # type: ignore

    with pytest.raises(TypeError, match="All dictionary keys must be strings"):
        GeneratedQuestion.from_dict({1: "invalid key", "slot_id": "s1"})  # type: ignore

    with pytest.raises(TypeError, match="marks must be an integer"):
        GeneratedQuestion.from_dict({
            "slot_id": "s1", "module_id": "m1", "global_q_idx": 1,
            "question_text": "text", "solution": "sol", "marking_scheme": [],
            "marks": True,
        })


def test_from_dict_rejects_unknown_top_level_and_subpart_fields():
    """24. from_dict() rejects unknown top-level and subpart fields."""
    valid_base = {
        "slot_id": "s1", "module_id": "m1", "global_q_idx": 1,
        "question_text": "text", "solution": "sol", "marking_scheme": [],
        "marks": 10, "partition": [10],
    }

    bad_top = dict(valid_base)
    bad_top["unrecognized_top_field"] = "bad"
    with pytest.raises(ValueError, match="received unknown top-level fields"):
        GeneratedQuestion.from_dict(bad_top)

    bad_sp = dict(valid_base)
    bad_sp["sub_parts"] = [{
        "label": "a", "marks": 10, "text": "text", "unknown_subpart_field": 123
    }]
    with pytest.raises(ValueError, match="contains unknown fields"):
        GeneratedQuestion.from_dict(bad_sp)


def test_from_dict_allow_legacy_flag_type_validation():
    """25. allow_legacy must be an exact bool."""
    d = {
        "slot_id": "s1", "module_id": "m1", "global_q_idx": 1,
        "question_text": "text", "solution": "sol", "marking_scheme": [],
        "marks": 10, "partition": [10],
    }
    with pytest.raises(TypeError, match="allow_legacy must be exact bool"):
        GeneratedQuestion.from_dict(d, allow_legacy="true")  # type: ignore

    with pytest.raises(TypeError, match="allow_legacy must be exact bool"):
        GeneratedQuestion.from_dict(d, allow_legacy=1)  # type: ignore


def test_from_dict_legacy_key_distinction():
    """26. Absent sub_parts key vs explicit [] legacy distinction."""
    legacy_data = {
        "slot_id": "s1", "module_id": "m1", "global_q_idx": 1,
        "question_text": "(a) Part a. (b) Part b.", "solution": "sol", "marking_scheme": [],
        "marks": 10, "partition": [6, 4],
    }

    # Absent key with allow_legacy=True -> succeeds
    q = GeneratedQuestion.from_dict(legacy_data, allow_legacy=True)
    assert q.sub_parts == []

    # Absent key with allow_legacy=False -> raises ValueError
    with pytest.raises(ValueError, match="Missing 'sub_parts' key for multi-part question"):
        GeneratedQuestion.from_dict(legacy_data, allow_legacy=False)

    # Explicit sub_parts: [] with allow_legacy=True -> strictly raises ValueError!
    explicit_empty_data = dict(legacy_data)
    explicit_empty_data["sub_parts"] = []
    with pytest.raises(ValueError, match=r"Explicit 'sub_parts': \[\] provided for multi-part question"):
        GeneratedQuestion.from_dict(explicit_empty_data, allow_legacy=True)

    # Absent key with allow_legacy=True but bad partition sum -> raises ValueError!
    bad_sum_legacy = dict(legacy_data)
    bad_sum_legacy["marks"] = 20
    with pytest.raises(ValueError, match="Sum of partition"):
        GeneratedQuestion.from_dict(bad_sum_legacy, allow_legacy=True)


def test_from_dict_defensive_copying_and_roundtrip():
    """27. Round-trip serialization and defensive copying."""
    sp_a = SubPart(label="a", marks=6, text="Explain scheduling.", topic="CPU", solution="Sol A")
    sp_b = SubPart(label="b", marks=4, text="Compare algorithms.", topic="CPU", solution="Sol B")
    q = GeneratedQuestion(
        slot_id="s1", module_id="m1", global_q_idx=1,
        question_text="", solution="", marking_scheme=[],
        marks=10, partition=[6, 4], sub_parts=[sp_a, sp_b],
    )
    d = q.to_dict()
    q_hydrated = GeneratedQuestion.from_dict(d)

    assert q_hydrated.slot_id == q.slot_id
    assert len(q_hydrated.sub_parts) == 2
    assert q_hydrated.sub_parts[0].label == "a"
    assert q_hydrated.sub_parts[0].marks == 6

    # Mutate source dict, confirm hydrated question is unaffected (defensive copy)
    d["sub_parts"][0]["text"] = "mutated text"
    assert q_hydrated.sub_parts[0].text == "Explain scheduling."


def test_to_dict_sync_projections_flag():
    """28. to_dict(sync_projections) flag preserves or synchronizes projections."""
    sp_a = SubPart(label="a", marks=6, text="Part A text")
    sp_b = SubPart(label="b", marks=4, text="Part B text")
    q = GeneratedQuestion(
        slot_id="s1", module_id="m1", global_q_idx=1,
        question_text="stale text", solution="stale sol", marking_scheme=[],
        marks=10, partition=[6, 4], sub_parts=[sp_a, sp_b],
    )

    d_no_sync = q.to_dict(sync_projections=False)
    assert d_no_sync["question_text"] == "stale text"

    d_sync = q.to_dict(sync_projections=True)
    assert d_sync["question_text"] == "(a) Part A text\n\n(b) Part B text"


def test_deterministic_materialization_and_downstream_compatibility():
    """29. Deterministic projection format cleanly satisfies frozen legacy adapter and evaluators."""
    sp_a = SubPart(label="a", marks=6, text="Explain process scheduling algorithms in operating systems.")
    sp_b = SubPart(label="b", marks=4, text="Compare preemptive and non-preemptive scheduling.")
    q = GeneratedQuestion(
        slot_id="s1", module_id="m1", global_q_idx=1,
        question_text="", solution="", marking_scheme=[],
        marks=10, partition=[6, 4], sub_parts=[sp_a, sp_b],
    )
    q.sync_materialized_projections()

    # Downstream compatibility check against frozen _extract_subpart_structure in legacy_adapter.py
    extracted = _extract_subpart_structure(q.question_text, len(q.partition))
    assert extracted.success is True
    assert len(extracted.clauses) == 2
    assert extracted.clauses[0] == sp_a.text
    assert extracted.clauses[1] == sp_b.text

    # Downstream compatibility check against frozen _PART_LABEL regex in local_evaluators.py
    labels = _PART_LABEL.findall(q.question_text)
    assert len(labels) == 2
    assert labels == ["a", "b"]


def test_solution_projection_does_not_leak_stale_parent():
    """30. Materializing solution when subparts have no solution does not leak stale parent solution."""
    sp_a = SubPart(label="a", marks=6, text="Text A", solution="")
    sp_b = SubPart(label="b", marks=4, text="Text B", solution="")
    q = GeneratedQuestion(
        slot_id="s1", module_id="m1", global_q_idx=1,
        question_text="", solution="Stale parent solution text", marking_scheme=[],
        marks=10, partition=[6, 4], sub_parts=[sp_a, sp_b],
    )
    mat_sol = q.materialize_solution()
    assert mat_sol == ""


def test_marking_scheme_projection_preserves_metadata_and_blocks_stale_parent():
    """31. Materializing marking scheme formats subpart labels and preserves criteria metadata."""
    sp_a = SubPart(
        label="a", marks=6, text="Text A",
        marking_scheme=[{"criterion": "Def", "marks": 4, "rubric": "accurate", "bloom": "L2"}, {"criterion": "Ex", "marks": 2}],
    )
    sp_b = SubPart(label="b", marks=4, text="Text B", marking_scheme=[{"criterion": "Comp", "marks": 4}])
    q = GeneratedQuestion(
        slot_id="s1", module_id="m1", global_q_idx=1,
        question_text="", solution="", marking_scheme=[{"criterion": "Stale", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=[sp_a, sp_b],
    )
    mat_scheme = q.materialize_marking_scheme()
    assert len(mat_scheme) == 3
    assert mat_scheme[0]["criterion"] == "(a) Def"
    assert mat_scheme[0]["subpart"] == "a"
    assert mat_scheme[0]["rubric"] == "accurate"
    assert mat_scheme[0]["bloom"] == "L2"
    assert mat_scheme[2]["criterion"] == "(b) Comp"
    assert mat_scheme[2]["subpart"] == "b"


def test_structure_validation_comprehensive():
    """32. Comprehensive validation across 5 states, single-part generic marks, and bounds."""
    # Valid multi-part [6, 4]
    sp_a = SubPart(label="a", marks=6, text="Text A")
    sp_b = SubPart(label="b", marks=4, text="Text B")
    q = GeneratedQuestion("s1", "m1", 1, "(a) Text A\n\n(b) Text B", "", [], marks=10, partition=[6, 4], sub_parts=[sp_a, sp_b])
    assert q.validate_structure()[0] is True

    # Generic single-part [M] with subpart: marks=5, partition=[5]
    sp_single = SubPart(label="a", marks=5, text="Text")
    q_single = GeneratedQuestion("s1", "m1", 1, "Text", "", [], marks=5, partition=[5], sub_parts=[sp_single])
    assert q_single.validate_structure()[0] is True

    # Generic single-part [M] legacy: marks=5, partition=[5], sub_parts=[]
    q_legacy_single = GeneratedQuestion("s1", "m1", 1, "Text", "", [], marks=5, partition=[5], sub_parts=[])
    assert q_legacy_single.validate_structure()[0] is True

    # Single-part legacy with mismatching marks (marks=20, partition=[10])
    q_bad_single = GeneratedQuestion("s1", "m1", 1, "Text", "", [], marks=20, partition=[10], sub_parts=[])
    valid, err = q_bad_single.validate_structure()
    assert valid is False
    assert "Sum of partition" in err

    # partition=[] with populated sub_parts is strictly invalid
    q_empty_part = GeneratedQuestion("s1", "m1", 1, "Text", "", [], marks=6, partition=[], sub_parts=[sp_a])
    valid, err = q_empty_part.validate_structure()
    assert valid is False
    assert "partition is empty" in err

    # Partition > 26 subparts
    q_huge = GeneratedQuestion("s1", "m1", 1, "Text", "", [], marks=27, partition=[1]*27)
    valid, err = q_huge.validate_structure()
    assert valid is False
    assert "exceeds maximum 26 subparts" in err

    # allow_legacy=True does not exit early if parent fields are corrupted
    q_corrupt_parent = GeneratedQuestion("s1", "m1", 1, 12345, "", [], marks=10, partition=[6, 4], sub_parts=[])  # type: ignore
    valid, err = q_corrupt_parent.validate_structure(allow_legacy=True)
    assert valid is False
    assert "question_text must be exact str" in err

    # Legacy parent marking scheme entry does not impose criteria sum rule
    q_legacy_scheme = GeneratedQuestion(
        "s1", "m1", 1, "Text", "", [{"custom_key": "val", "notes": "freeform"}], marks=10, partition=[10], sub_parts=[]
    )
    assert q_legacy_scheme.validate_structure()[0] is True


def test_serialization_does_not_falsely_establish_pre_serialization_consistency():
    """33. Calling verify_projection_consistency detects divergence before serialization."""
    sp_a = SubPart(label="a", marks=6, text="Authoritative part A")
    sp_b = SubPart(label="b", marks=4, text="Authoritative part B")
    q = GeneratedQuestion(
        slot_id="s1", module_id="m1", global_q_idx=1,
        question_text="Divergent manual question text", solution="", marking_scheme=[],
        marks=10, partition=[6, 4], sub_parts=[sp_a, sp_b],
    )
    # Before synchronization, divergence is detected
    consistent, reason = q.verify_projection_consistency()
    assert consistent is False
    assert "question_text does not match" in reason

    # validate_structure with check_projections=True catches divergence
    valid, err = q.validate_structure(check_projections=True)
    assert valid is False
    assert "Projection inconsistency" in err

    # After explicit sync, consistency passes
    q.sync_materialized_projections()
    consistent, reason = q.verify_projection_consistency()
    assert consistent is True
    assert reason == ""


if __name__ == "__main__":
    pytest.main(["-v", __file__])
