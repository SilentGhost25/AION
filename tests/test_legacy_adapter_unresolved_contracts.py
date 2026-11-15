"""
Characterization and contract tests for Stage 2B: Legacy Adaptation Integrity.

Covers:
1. Safe stem-clause separation eliminating false overlap under frozen thresholds.
2. Detection of genuine overlap on duplicate clauses with shared stem.
3. Fail-closed rejection of sentence splitting without explicit structural markers.
4. Strict 3-way marker classification:
   - Rejection of unexpected alphabetic markers (e.g. (c), (a), (b)).
   - Rejection of nested markers preceding the first outer marker (e.g. (i), (a), (b)).
   - Preservation of nested Roman numerals in first and final outer clauses.
5. VTU partition bound enforcement (N <= 5) preventing outer/nested namespace collision.
6. Preservation of Stage 2A failure metadata on extraction failure and bound violations.
7. Fail-closed aggregate fallback on unmapped Stage 2A subpart labels.
8. Immutability of GeneratedQuestion.telemetry["warnings"].
9. Preservation of marks, partition, CO, Bloom, and images.
10. Actual DOCX export regression test inspecting docx.Document table cells.
11. Bijective base-26 generator complete range verification.
12. Alias identity (sub_questions and subQuestions) and strict jsonable serialization.
"""

import io
import re
import pytest
import docx

from core.generation.agents.generated_question import GeneratedQuestion
from core.generation.agents.legacy_adapter import (
    SubpartExtractionResult,
    _extract_subpart_structure,
    _is_nested_marker,
    _to_question_dict,
    get_subpart_label,
    to_legacy_shape,
    _OUTER_MARKER_RE,
)
from core.generation.agents.orchestrator_contracts import OrchestratorResult
from core.serialization import jsonable
from v0_1.docx_export import generate_docx_from_paper


def _make_sample_question(**kwargs) -> GeneratedQuestion:
    """Helper to construct a valid GeneratedQuestion with defaults."""
    defaults = {
        "slot_id": "slot_1_1",
        "module_id": "module_1",
        "global_q_idx": 1,
        "question_text": "(a) Explain paging. (b) Explain segmentation.",
        "solution": "Solution text here.",
        "marking_scheme": ["Part a: 5 marks", "Part b: 5 marks"],
        "marks": 10,
        "partition": [5, 5],
        "bloom": "L3",
        "co": "CO1",
        "topic": "Virtual Memory",
        "image_path": None,
        "figure_caption": None,
        "references_image": False,
        "diagram_request": None,
        "generation_source": "primary",
        "attempts": 1,
        "telemetry": {},
        "is_unresolved": False,
        "unresolved_scope": None,
        "unresolved_subparts": (),
        "failure_code": None,
        "failure_detail": None,
    }
    defaults.update(kwargs)
    return GeneratedQuestion(**defaults)


# -----------------------------------------------------------------------------
# 1. Stem-Clause Separation & Overlap Detection Under Frozen Thresholds
# -----------------------------------------------------------------------------


def test_shared_stem_distinct_clauses_no_false_overlap():
    """
    Verifies that a multi-part question with a long shared stem and distinct clauses
    does NOT trigger subquestion overlap under frozen baseline thresholds.
    """
    long_stem = "Given a demand-paged virtual memory system with a 32-bit address space, 4 KB page size, and an inverted page table structure:"
    text = f"{long_stem} (a) Calculate the total number of page table entries. (b) Explain the role and operation of the Translation Lookaside Buffer."
    q = _make_sample_question(
        question_text=text,
        partition=[6, 4],
    )
    res = _to_question_dict(q)

    assert res["subquestion_overlap"] is False
    assert res["overlap_details"] == []
    assert res["is_unresolved"] is False
    assert len(res["sub_questions"]) == 2
    # Stem is rendered once with subpart (a)
    assert res["sub_questions"][0]["text"].startswith(long_stem)
    assert "Calculate the total number" in res["sub_questions"][0]["text"]
    # Subpart (b) receives clean clause without redundant stem
    assert not res["sub_questions"][1]["text"].startswith(long_stem)
    assert res["sub_questions"][1]["text"] == "Explain the role and operation of the Translation Lookaside Buffer."


def test_shared_stem_duplicate_clauses_triggers_overlap():
    """
    Verifies that duplicate asks with a shared stem trigger overlap_detected=True.
    """
    stem = "In a multi-threaded operating system environment:"
    clause_a = "calculate the turnaround time and waiting time for processes using round robin scheduling algorithm."
    clause_b = "compute the turnaround time and waiting time for processes using round robin scheduling algorithm."
    text = f"{stem} (a) {clause_a} (b) {clause_b}"
    q = _make_sample_question(
        question_text=text,
        partition=[5, 5],
    )
    res = _to_question_dict(q)

    assert res["subquestion_overlap"] is True
    assert len(res["overlap_details"]) > 0
    assert res["telemetry"]["subquestion_overlap"] is True


# -----------------------------------------------------------------------------
# 2. Structural Parsing & Strict Marker State Machine
# -----------------------------------------------------------------------------


def test_sentence_splitting_rejected_without_markers():
    """
    Verifies that plain multi-sentence text without explicit structural markers
    fails closed (success=False), producing 1 aggregate entry with MULTI_PART_STRUCTURE_MISMATCH
    and NEVER duplicating text across partition slots.
    """
    text = "Explain the concept of paging. Discuss its advantages and limitations."
    q = _make_sample_question(
        question_text=text,
        partition=[6, 4],
        is_unresolved=False,
    )
    res = _to_question_dict(q)

    # Fail closed: extraction failed
    assert res["is_unresolved"] is True
    assert res["failure_code"] == "MULTI_PART_STRUCTURE_MISMATCH"
    assert res["adaptation_failure_code"] == "MULTI_PART_STRUCTURE_MISMATCH"
    # Exactly one aggregate entry, not duplicated across slots
    assert len(res["sub_questions"]) == 1
    assert res["sub_questions"][0]["letter"] == "a"
    assert res["sub_questions"][0]["text"] == text
    assert res["sub_questions"][0]["marks"] == 10
    assert res["sub_questions"][0]["is_unresolved"] is True
    assert res["sub_questions"][0]["adaptation_failure_code"] == "MULTI_PART_STRUCTURE_MISMATCH"


def test_unexpected_marker_c_a_b_fails_closed():
    """
    Verifies that an unexpected alphabetic marker (e.g. '(c)') appearing before '(a)'
    is NOT silently ignored, but immediately triggers fail-closed extraction failure.
    """
    text = "(c) Unexpected clause. (a) Explain paging. (b) Explain segmentation."
    extraction = _extract_subpart_structure(text, 2)
    assert extraction.success is False

    q = _make_sample_question(question_text=text, partition=[5, 5])
    res = _to_question_dict(q)
    assert res["is_unresolved"] is True
    assert res["adaptation_failure_code"] == "MULTI_PART_STRUCTURE_MISMATCH"
    assert len(res["sub_questions"]) == 1


def test_nested_marker_before_first_outer_marker_fails_closed():
    """
    Verifies that a nested marker appearing before any outer marker is opened
    (e.g. '(i)' before '(a)') fails closed because there is no parent clause.
    """
    text = "(i) Describe prerequisite. (a) Explain paging. (b) Explain segmentation."
    extraction = _extract_subpart_structure(text, 2)
    assert extraction.success is False

    q = _make_sample_question(question_text=text, partition=[5, 5])
    res = _to_question_dict(q)
    assert res["is_unresolved"] is True
    assert res["adaptation_failure_code"] == "MULTI_PART_STRUCTURE_MISMATCH"
    assert len(res["sub_questions"]) == 1


def test_nested_roman_numerals_first_outer_clause():
    """
    Verifies that nested Roman numerals (i), (ii) inside the first clause (a)
    are preserved as child elements and do not corrupt the outer partition structure.
    """
    text = (
        "(a) Explain memory management techniques:\n"
        "    (i) Describe contiguous allocation.\n"
        "    (ii) Explain non-contiguous paging.\n"
        "(b) Compare paging and segmentation with suitable diagrams."
    )
    extraction = _extract_subpart_structure(text, 2)
    assert extraction.success is True
    assert len(extraction.clauses) == 2
    assert "(i) Describe contiguous allocation." in extraction.clauses[0]
    assert "(ii) Explain non-contiguous paging." in extraction.clauses[0]
    assert extraction.clauses[1] == "Compare paging and segmentation with suitable diagrams."


def test_nested_roman_numerals_final_outer_clause():
    """
    Verifies that nested Roman numerals (i), (ii) inside the final clause (b)
    are preserved as child elements and are NOT treated as excess outer markers.
    """
    text = (
        "(a) Explain memory management.\n"
        "(b) Compare paging and segmentation:\n"
        "    (i) Describe page table structure.\n"
        "    (ii) Explain segment table structure."
    )
    extraction = _extract_subpart_structure(text, 2)
    assert extraction.success is True
    assert len(extraction.clauses) == 2
    assert extraction.clauses[0] == "Explain memory management."
    assert "(i) Describe page table structure." in extraction.clauses[1]
    assert "(ii) Explain segment table structure." in extraction.clauses[1]


# -----------------------------------------------------------------------------
# 3. VTU Partition Bound (N <= 5) & Namespace Overlap Elimination
# -----------------------------------------------------------------------------


def test_vtu_partition_bound_exceeded_fails_closed_and_prevents_i_collision():
    """
    Verifies that partitions with N > 5 (such as N=6 and N=9 where outer label is 'i')
    fail closed immediately, emitting an aggregate fallback with PARTITION_EXCEEDS_VTU_MAX
    and preventing any outer 'i' vs Roman '(i)' namespace collision.
    """
    # 9-part question reaching outer label 'i'
    text = "(a) P1. (b) P2. (c) P3. (d) P4. (e) P5. (f) P6. (g) P7. (h) P8. (i) P9."
    extraction = _extract_subpart_structure(text, 9)
    assert extraction.success is False

    q = _make_sample_question(
        question_text=text,
        partition=[1, 1, 1, 1, 1, 1, 1, 1, 2],
        is_unresolved=False,
    )
    res = _to_question_dict(q)

    assert res["is_unresolved"] is True
    assert res["failure_code"] == "PARTITION_EXCEEDS_VTU_MAX"
    assert res["adaptation_failure_code"] == "PARTITION_EXCEEDS_VTU_MAX"
    assert "partition_exceeds_vtu_max:9" in res["telemetry"]["warnings"]
    assert len(res["sub_questions"]) == 1
    assert res["sub_questions"][0]["marks"] == 10


def test_vtu_partition_bound_exceeded_preserves_stage2a_metadata():
    """
    Verifies that N > 5 on a question with pre-existing Stage 2A failure
    preserves the canonical Stage 2A failure code and records adaptation failure.
    """
    text = "(a) P1. (b) P2. (c) P3. (d) P4. (e) P5. (f) P6."
    q = _make_sample_question(
        question_text=text,
        partition=[2, 2, 2, 2, 1, 1],
        is_unresolved=True,
        unresolved_scope="question",
        unresolved_subparts=(),
        failure_code="BLOOM_DEPTH_MISMATCH",
        failure_detail="Question Bloom level too low.",
    )
    res = _to_question_dict(q)

    assert res["is_unresolved"] is True
    # Preserves Stage 2A failure code
    assert res["failure_code"] == "BLOOM_DEPTH_MISMATCH"
    assert res["failure_detail"] == "Question Bloom level too low."
    # Records adaptation failure code separately
    assert res["adaptation_failure_code"] == "PARTITION_EXCEEDS_VTU_MAX"
    assert len(res["sub_questions"]) == 1
    assert res["sub_questions"][0]["failure_code"] == "BLOOM_DEPTH_MISMATCH"
    assert res["sub_questions"][0]["adaptation_failure_code"] == "PARTITION_EXCEEDS_VTU_MAX"


# -----------------------------------------------------------------------------
# 4. Stage 2A Metadata Preservation & Fail-Closed Fallbacks
# -----------------------------------------------------------------------------


def test_unmapped_stage2a_subpart_labels_fails_closed():
    """
    Verifies that when Stage 2A reports an unresolved subpart label (e.g. 'c')
    that does not exist in the extracted outer labels ('a', 'b'), the adapter
    fails closed to 1 aggregate entry and records UNRESOLVED_SUBPART_LABEL_MISMATCH.
    """
    text = "(a) Define paging. (b) Explain segmentation."
    q = _make_sample_question(
        question_text=text,
        partition=[5, 5],
        is_unresolved=True,
        unresolved_scope="subquestion",
        unresolved_subparts=("c",),  # Label 'c' does not exist in 2-part partition!
        failure_code="BLOOM_DEPTH_MISMATCH",
        failure_detail="Subpart (c) Bloom mismatch.",
    )
    res = _to_question_dict(q)

    # Fail closed: never falsely mark 'a' or 'b' resolved
    assert res["is_unresolved"] is True
    assert res["failure_code"] == "BLOOM_DEPTH_MISMATCH"
    assert res["adaptation_failure_code"] == "UNRESOLVED_SUBPART_LABEL_MISMATCH"
    assert len(res["sub_questions"]) == 1
    assert res["sub_questions"][0]["is_unresolved"] is True
    assert res["sub_questions"][0]["failure_code"] == "BLOOM_DEPTH_MISMATCH"
    assert res["sub_questions"][0]["adaptation_failure_code"] == "UNRESOLVED_SUBPART_LABEL_MISMATCH"


def test_extraction_failure_preserves_stage2a_metadata():
    """
    Verifies that extraction failure on a question with pre-existing Stage 2A
    subpart failure (e.g. 'b') marks the aggregate fallback entry unresolved,
    preserves the Stage 2A failure code, and records MULTI_PART_STRUCTURE_MISMATCH.
    """
    text = "Explain paging mechanisms and compare with segmentation."  # Missing markers
    q = _make_sample_question(
        question_text=text,
        partition=[6, 4],
        is_unresolved=True,
        unresolved_scope="subquestion",
        unresolved_subparts=("b",),
        failure_code="BLOOM_DEPTH_MISMATCH",
        failure_detail="Subpart (b) Bloom level (L2) < required (L3)",
    )
    res = _to_question_dict(q)

    # Aggregate entry must be unresolved! Never marked resolved because 'a' not in ('b',)
    assert res["is_unresolved"] is True
    assert res["failure_code"] == "BLOOM_DEPTH_MISMATCH"
    assert res["failure_detail"] == "Subpart (b) Bloom level (L2) < required (L3)"
    assert res["adaptation_failure_code"] == "MULTI_PART_STRUCTURE_MISMATCH"

    assert len(res["sub_questions"]) == 1
    sq = res["sub_questions"][0]
    assert sq["is_unresolved"] is True
    assert sq["failure_code"] == "BLOOM_DEPTH_MISMATCH"
    assert sq["failure_detail"] == "Subpart (b) Bloom level (L2) < required (L3)"
    assert sq["adaptation_failure_code"] == "MULTI_PART_STRUCTURE_MISMATCH"


def test_telemetry_warnings_immutability():
    """
    Verifies that appending a partition warning to the adapted dictionary
    does NOT mutate the original GeneratedQuestion.telemetry["warnings"].
    """
    original_warnings = ["pre_existing_warning_1"]
    q = _make_sample_question(
        partition=[2, 2, 2, 2, 1, 1],  # 6 parts -> triggers high partition warning
        telemetry={"warnings": original_warnings},
    )

    res = _to_question_dict(q)

    assert "partition_exceeds_vtu_max:6" in res["telemetry"]["warnings"]
    # Original list in q must be unmodified!
    assert "partition_exceeds_vtu_max:6" not in original_warnings
    assert q.telemetry["warnings"] == ["pre_existing_warning_1"]


def test_marks_numbering_co_bloom_preserved():
    """
    Verifies that all question metadata (marks, partition, total_marks, global_q_idx,
    module_id, co, bloom, topic, solution, images) is 100% preserved.
    """
    q = _make_sample_question(
        slot_id="slot_3_2",
        module_id="module_3",
        global_q_idx=6,
        marks=20,
        partition=[8, 6, 6],
        co="CO3",
        bloom="L4",
        topic="Disk Scheduling",
        image_path="figures/disk.png",
        figure_caption="Figure 3.1: SSTF Scheduling",
        references_image=True,
    )
    res = _to_question_dict(q)

    assert res["slot_id"] == "slot_3_2"
    assert res["module_id"] == "module_3"
    assert res["global_q_idx"] == 6
    assert res["marks"] == 20
    assert res["total_marks"] == 20
    assert res["partition"] == [8, 6, 6]
    assert res["co"] == "CO3"
    assert res["bloom"] == "L4"
    assert res["topic"] == "Disk Scheduling"
    assert res["image_path"] == "figures/disk.png"
    assert res["figure_caption"] == "Figure 3.1: SSTF Scheduling"
    assert res["references_image"] is True


# -----------------------------------------------------------------------------
# 5. Deep DOCX Export Table Cell Regression
# -----------------------------------------------------------------------------


def test_actual_docx_export_table_cell_regression():
    """
    Executes actual generate_docx_from_paper() on representative adapted dictionaries
    and parses the generated docx document to verify:
    1. A valid binary DOCX is produced.
    2. Fallback unextractable questions produce exactly 1 question row in the table (NO text duplication).
    3. Multi-part questions render the shared stem once with subpart (a), and subpart (b) renders clean.
    """
    q_fallback = _make_sample_question(
        slot_id="slot_1_1",
        module_id="module_1",
        global_q_idx=1,
        question_text="Unextractable question text without markers.",
        partition=[6, 4],
    )
    adapted_fallback = _to_question_dict(q_fallback)

    stem = "Given an operating system memory management unit:"
    q_multipart = _make_sample_question(
        slot_id="slot_1_2",
        module_id="module_1",
        global_q_idx=2,
        question_text=f"{stem} (a) Describe page tables. (b) Explain segment tables.",
        partition=[5, 5],
    )
    adapted_multipart = _to_question_dict(q_multipart)

    paper_data = {
        "subject_name": "Operating Systems",
        "subject_code": "21CS54",
        "exam_type": "IAT1",
        "department": "Computer Science & Engineering",
        "semester": "V Semester",
        "max_marks": 50,
        "questions": [adapted_fallback, adapted_multipart],
    }

    docx_bytes = generate_docx_from_paper(paper_data)
    assert isinstance(docx_bytes, io.BytesIO)
    assert docx_bytes.getvalue().startswith(b"PK")  # Valid zip / docx magic header

    doc = docx.Document(docx_bytes)
    assert len(doc.tables) >= 1

    # Locate the master question table (the 6-column table with 'Question Description' header)
    table = None
    for t in doc.tables:
        if len(t.rows) > 0 and len(t.rows[0].cells) >= 6:
            header_text = [c.text.strip() for c in t.rows[0].cells]
            if "Question Description" in header_text:
                table = t
                break
    assert table is not None, "Master question table not found in generated DOCX"
    description_cells = [row.cells[2].text.strip() for row in table.rows if len(row.cells) >= 6]

    # Verify fallback question text appears exactly ONCE in the table (zero duplication!)
    fallback_occurrences = [text for text in description_cells if "Unextractable question text" in text]
    assert len(fallback_occurrences) == 1
    assert fallback_occurrences[0] == "Unextractable question text without markers."

    # Verify multi-part question: subpart (a) has stem + clause, subpart (b) has clause only
    multipart_part_a = [text for text in description_cells if "Describe page tables" in text]
    multipart_part_b = [text for text in description_cells if "Explain segment tables" in text]
    assert len(multipart_part_a) == 1
    assert stem in multipart_part_a[0]
    assert len(multipart_part_b) == 1
    assert stem not in multipart_part_b[0]


# -----------------------------------------------------------------------------
# 6. Additional Guardrails & Utility Verification
# -----------------------------------------------------------------------------


def test_regex_compiles_without_lookbehind_error():
    """Explicitly verifies that _OUTER_MARKER_RE compiles without lookbehind errors."""
    pattern = _OUTER_MARKER_RE.pattern
    compiled = re.compile(pattern, re.MULTILINE)
    assert compiled is not None

    # Test match at start of string
    m1 = compiled.search("(a) First question.")
    assert m1 is not None
    assert m1.group("p1") == "a"
    assert m1.start("marker") == 0

    # Test match after newline
    m2 = compiled.search("Prefix\n(b) Second question.")
    assert m2 is not None
    assert m2.group("p1") == "b"

    # Test match after period and space
    m3 = compiled.search("Sentence one. a) Third question.")
    assert m3 is not None
    assert m3.group("p2") == "a"


def test_bijective_base26_label_generator_complete_range():
    """Verifies bijective base-26 generator across 0..702."""
    assert get_subpart_label(0) == "a"
    assert get_subpart_label(25) == "z"
    assert get_subpart_label(26) == "aa"
    assert get_subpart_label(27) == "ab"
    assert get_subpart_label(51) == "az"
    assert get_subpart_label(52) == "ba"
    assert get_subpart_label(701) == "zz"
    assert get_subpart_label(702) == "aaa"

    with pytest.raises(ValueError):
        get_subpart_label(-1)


def test_subpart_presentation_layout():
    """Verifies stem is attached to part (a) once and part (b) is clean."""
    q = _make_sample_question(
        question_text="Consider a system: (a) Calculate throughput. (b) Calculate latency.",
        partition=[5, 5],
    )
    res = _to_question_dict(q)
    assert res["sub_questions"][0]["text"] == "Consider a system: Calculate throughput."
    assert res["sub_questions"][1]["text"] == "Calculate latency."
    assert res["stem"] == "Consider a system:"


def test_excess_outer_markers_fails_closed():
    """Verifies 3 outer markers for 2-part partition returns success=False without silent concatenation."""
    text = "(a) Explain paging. (b) Explain segmentation. (c) Compare both."
    extraction = _extract_subpart_structure(text, 2)
    assert extraction.success is False

    q = _make_sample_question(question_text=text, partition=[5, 5])
    res = _to_question_dict(q)
    assert res["is_unresolved"] is True
    assert res["adaptation_failure_code"] == "MULTI_PART_STRUCTURE_MISMATCH"
    assert len(res["sub_questions"]) == 1


def test_ambiguous_extraction_skips_pairwise_overlap():
    """Verifies detect_subquestion_overlap is not invoked on unextractable text."""
    q = _make_sample_question(
        question_text="Plain question without markers.",
        partition=[5, 5],
    )
    res = _to_question_dict(q)
    assert res["subquestion_overlap"] is False
    assert res["overlap_details"] == []


def test_stage2a_metadata_propagation_subquestion_scope_clean_extraction():
    """Verifies granular resolution state when extraction and label mapping succeed."""
    q = _make_sample_question(
        question_text="(a) Define paging. (b) Explain segmentation.",
        partition=[5, 5],
        is_unresolved=True,
        unresolved_scope="subquestion",
        unresolved_subparts=("b",),
        failure_code="BLOOM_DEPTH_MISMATCH",
        failure_detail="Subpart (b) Bloom mismatch.",
    )
    res = _to_question_dict(q)

    assert res["is_unresolved"] is True
    assert res["unresolved_scope"] == "subquestion"
    assert res["unresolved_subparts"] == ["b"]
    assert res["failure_code"] == "BLOOM_DEPTH_MISMATCH"

    # Part (a) is resolved
    assert res["sub_questions"][0]["letter"] == "a"
    assert res["sub_questions"][0]["is_unresolved"] is False
    assert res["sub_questions"][0]["failure_code"] is None

    # Part (b) is unresolved
    assert res["sub_questions"][1]["letter"] == "b"
    assert res["sub_questions"][1]["is_unresolved"] is True
    assert res["sub_questions"][1]["failure_code"] == "BLOOM_DEPTH_MISMATCH"


def test_stage2a_metadata_propagation_question_scope():
    """Verifies whole-question failure marks all subparts unresolved."""
    q = _make_sample_question(
        question_text="(a) Define paging. (b) Explain segmentation.",
        partition=[5, 5],
        is_unresolved=True,
        unresolved_scope="question",
        unresolved_subparts=(),
        failure_code="UNGROUNDED",
        failure_detail="Question content ungrounded in syllabus.",
    )
    res = _to_question_dict(q)

    assert res["is_unresolved"] is True
    assert res["unresolved_scope"] == "question"
    assert res["sub_questions"][0]["is_unresolved"] is True
    assert res["sub_questions"][0]["failure_code"] == "UNGROUNDED"
    assert res["sub_questions"][1]["is_unresolved"] is True
    assert res["sub_questions"][1]["failure_code"] == "UNGROUNDED"


def test_sub_questions_and_subQuestions_alias_and_jsonable():
    """Verifies alias identity and strict jsonable serialization."""
    q = _make_sample_question(
        question_text="(a) Define paging. (b) Explain segmentation.",
        partition=[5, 5],
    )
    res = _to_question_dict(q)

    assert res["sub_questions"] is res["subQuestions"]
    serialized = jsonable(res, _strict=True)
    assert isinstance(serialized, dict)
    assert serialized["sub_questions"] == serialized["subQuestions"]
