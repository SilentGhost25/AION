"""
Stage G4 Acceptance Test Suite: Downstream Export & Adapter Alignment

Covers:
- Track G4-A: Direct Authoritative SubPart Adaptation (legacy_adapter.py)
  - Mathematical and symbolic syntax resilience (no regex false-positives)
  - Proof that authoritative structured inputs bypass _extract_subpart_structure
  - Exact ordered marks partition matching (e.g. [5,5] vs [6,4] fails closed)
  - Fail-closed guardrails for empty, count-mismatch, type-mismatch, duplicate-label, and invalid subparts
  - Legacy text-parsing fallback exclusively when sub_parts is None
- Track G4-B: SubPart Bloom & CO Metadata Preservation
  - SubPart dataclass schema and positional constructor compatibility
  - Exact string type validation and from_dict hydration
  - WritingAgent propagation from SubPartPlan
  - RefinementAgent preservation across successful and failed repairs
  - legacy_adapter per-subquestion Bloom/CO precedence with legacy fallback
  - Explicit planned Bloom/CO cannot silently disappear
- Track G4-C: Complete Multi-Visual Physical Asset Validation (checking_agent.py)
  - Valid first / invalid second image blocks
  - Invalid first / valid second image blocks
  - Multiple invalid images: non-suppression (all errors accumulated)
  - Parent image cannot mask subpart failure
  - Parent missing image when references_image is True
  - Text-referenced figures without assets fail closed
  - Unused figures do not block text-only subparts
- Integration:
  - DOCX export table rendering of distinct subquestion Bloom/CO
  - Single-part legacy question compatibility
"""

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docx import Document

from core.generation.agents.checking_agent import (
    CheckingAgent,
    BLOCK_REASON_MISSING_VISUAL_ASSET,
    STATUS_BLOCKED,
    STATUS_OK,
)
from core.generation.agents.evaluation_contracts import SlotVerdict
from core.generation.agents.generated_question import GeneratedQuestion, SubPart
from core.generation.agents.legacy_adapter import to_legacy_shape, _to_question_dict
from core.generation.agents.planning_agent import QuestionPlan, SubPartPlan
from core.generation.agents.refinement_agent import RefinementAgent
from core.generation.agents.writing_agent import WritingAgent
from core.generation.agents.base import AgentContext
from v0_1.docx_export import render_docx_document


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_visual_dir(tmp_path):
    """Creates temporary directory with valid and invalid PNG images."""
    from PIL import Image

    ws = tmp_path / "workspace"
    ws.mkdir()
    img_dir = ws / "figures"
    img_dir.mkdir()

    # Valid PNG image (> 100 bytes, structurally decodable)
    valid_file = img_dir / "valid_fig.png"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(str(valid_file), format="PNG")
    valid_png_bytes = valid_file.read_bytes()
    valid_sha = hashlib.sha256(valid_png_bytes).hexdigest()

    # Corrupt file (> 100 bytes to test signature and decodability validation)
    corrupt_file = img_dir / "corrupt_fig.png"
    corrupt_bytes = b"not_a_real_png_image_" * 10
    corrupt_file.write_bytes(corrupt_bytes)
    corrupt_sha = hashlib.sha256(corrupt_bytes).hexdigest()

    return {
        "workspace_root": str(ws),
        "valid_path": "figures/valid_fig.png",
        "valid_sha": valid_sha,
        "corrupt_path": "figures/corrupt_fig.png",
        "corrupt_sha": corrupt_sha,
        "missing_path": "figures/does_not_exist.png",
    }


# =============================================================================
# Track G4-A: Direct Authoritative SubPart Adaptation Tests
# =============================================================================

def test_g4a_math_syntax_parentheses_not_collapsed():
    """Test mathematical and algorithmic notation does not trigger regex collapse."""
    subparts = [
        SubPart(
            label="a",
            marks=6,
            text="Evaluate the function f(x) and its derivative f'(x) for cache bounds.",
            topic="Cache Optimization",
            solution="f'(x) = 2x.",
            marking_scheme=[{"criterion": "Derivative", "marks": 6}],
            bloom="L3",
            co="CO2",
        ),
        SubPart(
            label="b",
            marks=4,
            text="Compare the asymptotic complexity with O(n log n) sorting.",
            topic="Complexity Analysis",
            solution="Merge sort has O(n log n).",
            marking_scheme=[{"criterion": "Complexity", "marks": 4}],
            bloom="L4",
            co="CO3",
        ),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1",
        module_id="module_1",
        global_q_idx=1,
        question_text="(a) Evaluate the function f(x) and its derivative f'(x) for cache bounds.\n\n(b) Compare the asymptotic complexity with O(n log n) sorting.",
        solution="Overall solution.",
        marking_scheme=[{"criterion": "All", "marks": 10}],
        marks=10,
        partition=[6, 4],
        bloom="L4",
        co="CO3",
        sub_parts=subparts,
    )

    d = _to_question_dict(q)
    assert not d["is_unresolved"]
    assert d.get("adaptation_failure_code") is None
    assert len(d["sub_questions"]) == 2
    assert d["sub_questions"][0]["text"] == subparts[0].text
    assert d["sub_questions"][0]["marks"] == 6
    assert d["sub_questions"][1]["text"] == subparts[1].text
    assert d["sub_questions"][1]["marks"] == 4


def test_g4a_authoritative_subparts_bypass_regex_extractor():
    """Verify that when sub_parts is provided, _extract_subpart_structure is never invoked."""
    subparts = [
        SubPart(label="a", marks=6, text="Part A text.", topic="T1", solution="S1", marking_scheme=[{"criterion": "C1", "marks": 6}]),
        SubPart(label="b", marks=4, text="Part B text.", topic="T2", solution="S2", marking_scheme=[{"criterion": "C2", "marks": 4}]),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Part A text.\n\n(b) Part B text.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )

    with patch("core.generation.agents.legacy_adapter._extract_subpart_structure") as mock_extract:
        mock_extract.side_effect = RuntimeError("Regex extractor must not be called when sub_parts is populated!")
        d = _to_question_dict(q)
        assert not d["is_unresolved"]
        assert len(d["sub_questions"]) == 2
        mock_extract.assert_not_called()


def test_g4a_subpart_metadata_bit_for_bit_preservation():
    """Verify subpart attributes survive adaptation bit-for-bit."""
    subparts = [
        SubPart(
            label="a", marks=6, text="Exact Part A text.", topic="Topic A",
            solution="Exact Solution A", marking_scheme=[{"criterion": "Crit A", "marks": 6}],
            figure_id="fig_a", image_path="path/a.png", figure_caption="Cap A", image_sha256="sha_a",
            bloom="L2", co="CO1",
        ),
        SubPart(
            label="b", marks=4, text="Exact Part B text.", topic="Topic B",
            solution="Exact Solution B", marking_scheme=[{"criterion": "Crit B", "marks": 4}],
            bloom="L3", co="CO2",
        ),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Exact Part A text.\n\n(b) Exact Part B text.",
        solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )

    d = _to_question_dict(q)
    sq0 = d["sub_questions"][0]
    assert sq0["label"] == "a"
    assert sq0["marks"] == 6
    assert sq0["text"] == "Exact Part A text."
    assert sq0["topic"] == "Topic A"
    assert sq0["solution"] == "Exact Solution A"
    assert sq0["marking_scheme"] == [{"criterion": "Crit A", "marks": 6}]
    assert sq0["figure_id"] == "fig_a"
    assert sq0["image_path"] == "path/a.png"
    assert sq0["figure_caption"] == "Cap A"
    assert sq0["image_sha256"] == "sha_a"
    assert sq0["bloom"] == "L2"
    assert sq0["co"] == "CO1"


def test_g4a_empty_structured_subparts_fails_closed():
    """Correction 1: Empty structured subparts must fail closed for multi-part questions."""
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Text A\n\n(b) Text B",
        solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=[],
    )

    with patch("core.generation.agents.legacy_adapter._extract_subpart_structure") as mock_extract:
        d = _to_question_dict(q)
        assert d["is_unresolved"] is True
        assert d["adaptation_failure_code"] == "STRUCTURED_SUBPARTS_EMPTY"
        assert len(d["sub_questions"]) == 1
        assert d["sub_questions"][0]["is_unresolved"] is True
        mock_extract.assert_not_called()


def test_g4a_empty_structured_subparts_single_part_fails_closed():
    """Correction 1: Empty structured subparts must fail closed even for single-part questions ([10])."""
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Explain single part question.",
        solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[10], sub_parts=[],
    )

    with patch("core.generation.agents.legacy_adapter._extract_subpart_structure") as mock_extract:
        d = _to_question_dict(q)
        assert d["is_unresolved"] is True
        assert d["adaptation_failure_code"] == "STRUCTURED_SUBPARTS_EMPTY"
        mock_extract.assert_not_called()


def test_g4a_count_mismatch_fails_closed():
    """Count mismatch between partition and sub_parts must fail closed."""
    subparts = [
        SubPart(label="a", marks=6, text="Part A text.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}]),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Part A text.\n\n(b) Part B text.",
        solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )

    d = _to_question_dict(q)
    assert d["is_unresolved"] is True
    assert d["adaptation_failure_code"] == "STRUCTURED_SUBPARTS_COUNT_MISMATCH"


def test_g4a_ordered_marks_mismatch_with_equal_total():
    """Correction G4-A: Equal mark total but mismatched distribution ([5,5] vs [6,4]) fails closed."""
    subparts = [
        SubPart(label="a", marks=5, text="Part A text.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 5}]),
        SubPart(label="b", marks=5, text="Part B text.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 5}]),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Part A text.\n\n(b) Part B text.",
        solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )

    d = _to_question_dict(q)
    assert d["is_unresolved"] is True
    assert d["adaptation_failure_code"] == "STRUCTURED_SUBPARTS_MARKS_MISMATCH"


def test_g4a_invalid_subparts_collection_type():
    """Invalid collection type for sub_parts (e.g. dict, string, int) fails closed."""
    for invalid_val in [{"a": "invalid"}, "not_a_list", 12345, 99.9]:
        q = GeneratedQuestion(
            slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
            question_text="Text.", solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
            marks=10, partition=[6, 4],
        )
        # Force invalid collection type
        q.sub_parts = invalid_val

        d = _to_question_dict(q)
        assert d["is_unresolved"] is True
        assert d["adaptation_failure_code"] == "STRUCTURED_SUBPARTS_INVALID_TYPE"


@pytest.mark.parametrize("invalid_partition", [
    [0, 10],            # Zero mark value
    [-6, 4],            # Negative mark value
    [6.0, 4.0],         # Float mark values
    ["6", "4"],         # String mark values
    [True, 4],          # Boolean mark value (True == 1)
    [6, False],         # Boolean mark value (False == 0)
    [],                 # Empty partition
    "not_a_list",       # Non-sequence partition
])
def test_g4a_partition_invalid_elements_rejected(invalid_partition):
    """Gate 1: Confirm zero, negative, float, string, and Boolean partition elements are rejected."""
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Text.", solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=invalid_partition, sub_parts=None,
    )

    d = _to_question_dict(q)
    assert d["is_unresolved"] is True
    assert d["adaptation_failure_code"] == "PARTITION_INVALID"


def test_g4a_invalid_subpart_object_fails_closed():
    """SubPart with negative marks or invalid label fails closed."""
    # Bypassing constructor validation directly to simulate corrupted state
    sp_a = SubPart(label="a", marks=6, text="Valid", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}])
    sp_b = SubPart(label="b", marks=4, text="Valid", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}])
    sp_b.label = "invalid_label"

    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Valid\n\n(b) Valid", solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=[sp_a, sp_b],
    )

    d = _to_question_dict(q)
    assert d["is_unresolved"] is True
    assert d["adaptation_failure_code"] == "STRUCTURED_SUBPART_VALIDATION_FAILED"


def test_g4a_duplicate_subpart_labels_fails_closed():
    """Correction 2: Duplicate subpart labels fail closed via set comparison."""
    sp_a1 = SubPart(label="a", marks=6, text="Part A1.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}])
    sp_a2 = SubPart(label="a", marks=4, text="Part A2.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}])

    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Part A1.\n\n(a) Part A2.", solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=[sp_a1, sp_a2],
    )

    d = _to_question_dict(q)
    assert d["is_unresolved"] is True
    assert d["adaptation_failure_code"] == "STRUCTURED_SUBPARTS_DUPLICATE_LABELS"


def test_g4a_legacy_fallback_when_subparts_none():
    """When sub_parts is None, legacy regex extraction is permitted and executes."""
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Explain depth-first search.\n\n(b) Explain breadth-first search.",
        solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=None,
    )

    d = _to_question_dict(q)
    assert not d["is_unresolved"]
    assert len(d["sub_questions"]) == 2
    assert d["sub_questions"][0]["label"] == "a"
    assert d["sub_questions"][1]["label"] == "b"


def test_g4a_projection_consistency_preserved():
    """Verify projection consistency invariant holds across direct adaptation."""
    subparts = [
        SubPart(label="a", marks=6, text="Text A.", topic="T1", solution="Sol A.", marking_scheme=[{"criterion": "C1", "marks": 6}]),
        SubPart(label="b", marks=4, text="Text B.", topic="T2", solution="Sol B.", marking_scheme=[{"criterion": "C2", "marks": 4}]),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="",
        solution="", marking_scheme=[],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )
    q.sync_materialized_projections()
    assert q.verify_projection_consistency()[0] is True

    d = _to_question_dict(q)
    assert d["text"] == q.question_text
    assert d["marks"] == q.marks


# =============================================================================
# Track G4-B: SubPart Bloom & CO Metadata Preservation Tests
# =============================================================================

def test_g4b_positional_subpart_constructor_compatibility():
    """Verify SubPart positional constructor remains compatible after adding bloom & co."""
    sp = SubPart("a", 6, "Text A", "Topic A", "Sol A", [{"criterion": "C", "marks": 6}], "fig1", "p.png", "cap", "sha")
    assert sp.bloom == ""
    assert sp.co == ""
    valid, err = sp.validate()
    assert valid, err


def test_g4b_subpart_schema_bloom_and_co():
    """Verify SubPart stores and validates bloom and co."""
    sp = SubPart(
        label="a", marks=6, text="Text A", topic="T", solution="S",
        marking_scheme=[{"criterion": "C", "marks": 6}],
        bloom="L3", co="CO2",
    )
    assert sp.bloom == "L3"
    assert sp.co == "CO2"
    valid, err = sp.validate()
    assert valid, err


def test_g4b_subpart_validation_rejects_non_str_bloom_co():
    """SubPart rejects non-string types for bloom or co."""
    sp = SubPart(label="a", marks=6, text="Text A", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}])
    sp.bloom = 123  # type: ignore
    valid, err = sp.validate()
    assert not valid
    assert "SubPart.bloom must be exact str" in err

    sp.bloom = "L2"
    sp.co = True  # type: ignore
    valid, err = sp.validate()
    assert not valid
    assert "SubPart.co must be exact str" in err


def test_g4b_from_dict_deserialization_with_bloom_co():
    """GeneratedQuestion.from_dict hydrates bloom and co on SubPart."""
    data = {
        "slot_id": "m1_Q1",
        "module_id": "module_1",
        "global_q_idx": 1,
        "question_text": "(a) Text A\n\n(b) Text B",
        "solution": "Sol",
        "marking_scheme": [{"criterion": "C", "marks": 10}],
        "marks": 10,
        "partition": [6, 4],
        "bloom": "L4",
        "co": "CO3",
        "sub_parts": [
            {
                "label": "a", "marks": 6, "text": "Text A", "topic": "T1", "solution": "Sol A",
                "marking_scheme": [{"criterion": "C1", "marks": 6}], "bloom": "L2", "co": "CO1",
            },
            {
                "label": "b", "marks": 4, "text": "Text B", "topic": "T2", "solution": "Sol B",
                "marking_scheme": [{"criterion": "C2", "marks": 4}], "bloom": "L4", "co": "CO3",
            },
        ],
    }
    q = GeneratedQuestion.from_dict(data)
    assert len(q.sub_parts) == 2
    assert q.sub_parts[0].bloom == "L2"
    assert q.sub_parts[0].co == "CO1"
    assert q.sub_parts[1].bloom == "L4"
    assert q.sub_parts[1].co == "CO3"


def test_g4b_writing_agent_populates_subpart_bloom_co():
    """WritingAgent populates SubPart.bloom and SubPart.co from SubPartPlan."""
    plan = QuestionPlan(
        slot_id="m1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        total_marks=10, partition=[6, 4], bloom="L4", co="CO3",
        sub_part_plans=[
            SubPartPlan(label="a", marks=6, topic="Stacks", bloom="L2", co="CO1"),
            SubPartPlan(label="b", marks=4, topic="Queues", bloom="L4", co="CO3"),
        ],
    )

    llm = MagicMock()
    llm.call.side_effect = [
        # Subpart (a) LLM response
        {"text": "Explain stack operations and state representations in detail.", "solution": "Sol stack.", "marking_scheme": [{"criterion": "C", "marks": 6}], "bloom_verb_used": "Explain"},
        # Subpart (b) LLM response
        {"text": "Analyze circular queue amortized bounds and memory trade-offs.", "solution": "Sol queue.", "marking_scheme": [{"criterion": "C", "marks": 4}], "bloom_verb_used": "Analyze"},
    ]
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    q, attempts = agent._generate_structured_subparts(plan)
    assert len(q.sub_parts) == 2
    assert q.sub_parts[0].bloom == "L2"
    assert q.sub_parts[0].co == "CO1"
    assert q.sub_parts[1].bloom == "L4"
    assert q.sub_parts[1].co == "CO3"


def test_g4b_refinement_preserves_subpart_bloom_co():
    """RefinementAgent preserves bloom and co on target and sibling subparts."""
    sp_a = SubPart(label="a", marks=6, text="Original text A.", topic="T1", solution="Sol A", marking_scheme=[{"criterion": "C", "marks": 6}], bloom="L2", co="CO1")
    sp_b = SubPart(label="b", marks=4, text="Original text B.", topic="T2", solution="Sol B", marking_scheme=[{"criterion": "C", "marks": 4}], bloom="L4", co="CO3")
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Original text A.\n\n(b) Original text B.",
        solution="Sol A.\n\nSol B.", marking_scheme=[{"criterion": "C", "marks": 6}, {"criterion": "C2", "marks": 4}],
        marks=10, partition=[6, 4], sub_parts=[sp_a, sp_b],
    )

    mock_llm = MagicMock()
    mock_llm.call.return_value = json.dumps({
        "label": "b",
        "marks": 4,
        "text": "Repaired text B.",
        "solution": "Repaired sol B.",
        "marking_scheme": [{"criterion": "Repaired C", "marks": 4}],
    })
    agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=2)
    verdict = SlotVerdict(
        slot_id="m1_Q1",
        verdict="fail",
        source="api",
        reason_code="FACTUAL_ERROR",
        affected_subparts=("b",),
        detail="SubPart (b) factual error",
    )
    outcome, repaired = agent._refine_one(original=q, verdict=verdict, llm_budget_used=0)
    assert outcome.resolved is True
    assert repaired is not None
    assert repaired.sub_parts[0].bloom == "L2"
    assert repaired.sub_parts[0].co == "CO1"
    assert repaired.sub_parts[1].bloom == "L4"
    assert repaired.sub_parts[1].co == "CO3"
    assert repaired.sub_parts[1].text == "Repaired text B."


def test_g4b_refinement_failure_preserves_original_bloom_co():
    """When refinement fails or is rejected, original object retains original Bloom/CO."""
    sp_a = SubPart(label="a", marks=6, text="Text A.", topic="T1", solution="Sol A", marking_scheme=[{"criterion": "C", "marks": 6}], bloom="L2", co="CO1")
    sp_b = SubPart(label="b", marks=4, text="Text B.", topic="T2", solution="Sol B", marking_scheme=[{"criterion": "C", "marks": 4}], bloom="L4", co="CO3")
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Text A.\n\n(b) Text B.",
        solution="Sol A.\n\nSol B.", marking_scheme=[{"criterion": "C", "marks": 6}, {"criterion": "C2", "marks": 4}],
        marks=10, partition=[6, 4], sub_parts=[sp_a, sp_b],
    )

    mock_llm = MagicMock()
    mock_llm.call.return_value = json.dumps({
        "label": "b",
        "marks": 5,  # Illegal mark change (partition is 4)
        "text": "Illegal mark change text.",
        "solution": "Repaired sol B.",
        "marking_scheme": [{"criterion": "Repaired C", "marks": 5}],
    })
    agent = RefinementAgent(llm_caller=mock_llm, max_attempts_per_slot=2)
    verdict = SlotVerdict(
        slot_id="m1_Q1",
        verdict="fail",
        source="api",
        reason_code="FACTUAL_ERROR",
        affected_subparts=("b",),
        detail="SubPart (b) factual error",
    )
    outcome, repaired = agent._refine_one(original=q, verdict=verdict, llm_budget_used=0)
    assert outcome.resolved is False
    assert repaired is None
    # Original remains completely unchanged
    assert q.sub_parts[0].bloom == "L2"
    assert q.sub_parts[1].bloom == "L4"


def test_g4b_legacy_adapter_emits_distinct_subpart_bloom_co():
    """legacy_adapter populates distinct subpart bloom and co in adapted sub_questions."""
    subparts = [
        SubPart(label="a", marks=6, text="Text A.", topic="T1", solution="S1", marking_scheme=[{"criterion": "C1", "marks": 6}], bloom="L2", co="CO1"),
        SubPart(label="b", marks=4, text="Text B.", topic="T2", solution="S2", marking_scheme=[{"criterion": "C2", "marks": 4}], bloom="L4", co="CO3"),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Text A.\n\n(b) Text B.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], bloom="L4", co="CO3", sub_parts=subparts,
    )

    d = _to_question_dict(q)
    assert d["sub_questions"][0]["bloom"] == "L2"
    assert d["sub_questions"][0]["co"] == "CO1"
    assert d["sub_questions"][1]["bloom"] == "L4"
    assert d["sub_questions"][1]["co"] == "CO3"


def test_g4b_explicit_planned_bloom_co_cannot_silently_disappear():
    """Planned Bloom/CO survives from planning through writing and adaptation."""
    plan = QuestionPlan(
        slot_id="m1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        total_marks=10, partition=[6, 4], bloom="L4", co="CO3",
        sub_part_plans=[
            SubPartPlan(label="a", marks=6, topic="Stacks", bloom="L2", co="CO1"),
            SubPartPlan(label="b", marks=4, topic="Queues", bloom="L4", co="CO3"),
        ],
    )
    llm = MagicMock()
    llm.call.side_effect = [
        {"text": "Explain stack operations and state representations.", "solution": "Sol A", "marking_scheme": [{"criterion": "C", "marks": 6}], "bloom_verb_used": "Explain"},
        {"text": "Analyze circular queue amortized bounds and memory trade-offs.", "solution": "Sol B", "marking_scheme": [{"criterion": "C", "marks": 4}], "bloom_verb_used": "Analyze"},
    ]
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    q, attempts = agent._generate_structured_subparts(plan)

    adapted = _to_question_dict(q)
    assert adapted["sub_questions"][0]["bloom"] == "L2"
    assert adapted["sub_questions"][0]["co"] == "CO1"
    assert adapted["sub_questions"][1]["bloom"] == "L4"
    assert adapted["sub_questions"][1]["co"] == "CO3"


# =============================================================================
# Track G4-C: Complete Multi-Visual Validation Tests
# =============================================================================

def test_g4c_valid_first_invalid_second_image_blocks(temp_visual_dir):
    """SubPart (a) has valid image, SubPart (b) has missing image -> BLOCKED."""
    ws_root = temp_visual_dir["workspace_root"]
    subparts = [
        SubPart(label="a", marks=6, text="Refer to the figure.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}], image_path=temp_visual_dir["valid_path"], image_sha256=temp_visual_dir["valid_sha"]),
        SubPart(label="b", marks=4, text="Refer to the figure.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}], image_path=temp_visual_dir["missing_path"], image_sha256="fake_sha"),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Refer to the figure.\n\n(b) Refer to the figure.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx = AgentContext(plans=[plan], questions=[q], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res = checker.run(ctx)
    assert res.payload.status == STATUS_BLOCKED
    assert any("subpart_b" in r for r in res.payload.blocked_reasons)


def test_g4c_invalid_first_valid_second_image_blocks(temp_visual_dir):
    """SubPart (a) has missing image, SubPart (b) has valid image -> BLOCKED."""
    ws_root = temp_visual_dir["workspace_root"]
    subparts = [
        SubPart(label="a", marks=6, text="Refer to the figure.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}], image_path=temp_visual_dir["missing_path"], image_sha256="fake_sha"),
        SubPart(label="b", marks=4, text="Refer to the figure.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}], image_path=temp_visual_dir["valid_path"], image_sha256=temp_visual_dir["valid_sha"]),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Refer to the figure.\n\n(b) Refer to the figure.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx = AgentContext(plans=[plan], questions=[q], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res = checker.run(ctx)
    assert res.payload.status == STATUS_BLOCKED
    assert any("subpart_a" in r for r in res.payload.blocked_reasons)


def test_g4c_multiple_invalid_images_records_all(temp_visual_dir):
    """Non-suppression: SubPart (a) and SubPart (b) both invalid -> both errors recorded."""
    ws_root = temp_visual_dir["workspace_root"]
    subparts = [
        SubPart(label="a", marks=6, text="Refer to the figure.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}], image_path=temp_visual_dir["missing_path"], image_sha256="fake_sha"),
        SubPart(label="b", marks=4, text="Refer to the figure.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}], image_path=temp_visual_dir["corrupt_path"], image_sha256="wrong_sha"),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Refer to the figure.\n\n(b) Refer to the figure.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx = AgentContext(plans=[plan], questions=[q], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res = checker.run(ctx)
    assert res.payload.status == STATUS_BLOCKED
    reasons = res.payload.blocked_reasons
    assert any("subpart_a" in r for r in reasons)
    assert any("subpart_b" in r for r in reasons)


def test_g4c_parent_and_multiple_subpart_images_all_invalid(temp_visual_dir):
    """Parent and multiple subparts all invalid -> all three errors accumulated."""
    ws_root = temp_visual_dir["workspace_root"]
    subparts = [
        SubPart(label="a", marks=6, text="Refer to the figure.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}], image_path=temp_visual_dir["missing_path"], image_sha256="fake_sha"),
        SubPart(label="b", marks=4, text="Refer to the figure.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}], image_path=temp_visual_dir["corrupt_path"], image_sha256="wrong_sha"),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Refer to the figure.\n\n(b) Refer to the figure.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
        references_image=True, image_path=temp_visual_dir["missing_path"], image_sha256="fake_sha",
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx = AgentContext(plans=[plan], questions=[q], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res = checker.run(ctx)
    assert res.payload.status == STATUS_BLOCKED
    reasons = res.payload.blocked_reasons
    assert any("parent" in r for r in reasons)
    assert any("subpart_a" in r for r in reasons)
    assert any("subpart_b" in r for r in reasons)


def test_g4c_parent_image_cannot_hide_subpart_failure(temp_visual_dir):
    """Parent image is valid, but SubPart (b) image is missing -> BLOCKED."""
    ws_root = temp_visual_dir["workspace_root"]
    subparts = [
        SubPart(label="a", marks=6, text="Text A.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}]),
        SubPart(label="b", marks=4, text="Refer to the figure.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}], image_path=temp_visual_dir["missing_path"], image_sha256="fake_sha"),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Text A.\n\n(b) Refer to the figure.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
        references_image=True, image_path=temp_visual_dir["valid_path"], image_sha256=temp_visual_dir["valid_sha"],
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx = AgentContext(plans=[plan], questions=[q], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res = checker.run(ctx)
    assert res.payload.status == STATUS_BLOCKED
    assert any("subpart_b" in r for r in res.payload.blocked_reasons)


def test_g4c_parent_missing_path_when_references_image_true(temp_visual_dir):
    """Correction 3: Parent references_image=True without image_path blocks even when subparts exist."""
    ws_root = temp_visual_dir["workspace_root"]
    subparts = [
        SubPart(label="a", marks=6, text="Text A.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}], image_path=temp_visual_dir["valid_path"], image_sha256=temp_visual_dir["valid_sha"]),
        SubPart(label="b", marks=4, text="Text B.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}]),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Text A.\n\n(b) Text B.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
        references_image=True, image_path=None,  # Missing parent path
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx = AgentContext(plans=[plan], questions=[q], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res = checker.run(ctx)
    assert res.payload.status == STATUS_BLOCKED
    assert any("parent:missing_path" in r for r in res.payload.blocked_reasons)


def test_g4c_subpart_figure_id_without_path_blocks(temp_visual_dir):
    """SubPart with figure_id but no image_path -> BLOCKED."""
    ws_root = temp_visual_dir["workspace_root"]
    subparts = [
        SubPart(label="a", marks=6, text="Text A.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}], figure_id="fig_99", image_path=None),
        SubPart(label="b", marks=4, text="Text B.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}]),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Text A.\n\n(b) Text B.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx = AgentContext(plans=[plan], questions=[q], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res = checker.run(ctx)
    assert res.payload.status == STATUS_BLOCKED
    assert any("missing_path_for_figure_fig_99" in r for r in res.payload.blocked_reasons)


def test_g4c_subpart_text_references_figure_without_asset_blocks(temp_visual_dir):
    """Correction 4: SubPart text contains figure reference phrase without asset -> BLOCKED."""
    ws_root = temp_visual_dir["workspace_root"]
    subparts = [
        SubPart(label="a", marks=6, text="With reference to the given figure, explain B-tree insertion.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}], image_path=None, figure_id=None),
        SubPart(label="b", marks=4, text="Explain delete operation.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}]),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) With reference to the given figure, explain B-tree insertion.\n\n(b) Explain delete operation.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx = AgentContext(plans=[plan], questions=[q], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res = checker.run(ctx)
    assert res.payload.status == STATUS_BLOCKED
    assert any("text_references_figure_without_asset" in r for r in res.payload.blocked_reasons)


def test_g4c_text_only_and_unused_figures_do_not_block(temp_visual_dir):
    """Text-only subparts do not trigger visual validation false positives."""
    ws_root = temp_visual_dir["workspace_root"]
    subparts = [
        SubPart(label="a", marks=6, text="Explain pure textual concept.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}]),
        SubPart(label="b", marks=4, text="Compare textual concepts.", topic="T", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}]),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Explain pure textual concept.\n\n(b) Compare textual concepts.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
        references_image=False, image_path=None,
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx = AgentContext(plans=[plan], questions=[q], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res = checker.run(ctx)
    assert res.payload.status == STATUS_OK
    assert not any("MISSING_VISUAL_ASSET" in r for r in res.payload.blocked_reasons)


def test_g4c_valid_subpart_visual_binding_passes(temp_visual_dir):
    """Gate 3: Valid physical image bound to SubPart with matching SHA-256 passes checking agent."""
    ws_root = temp_visual_dir["workspace_root"]
    subparts = [
        SubPart(
            label="a", marks=6, text="Refer to Figure 1 for B-tree structure.",
            topic="B-Trees", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}],
            figure_id="fig_btree", image_path=temp_visual_dir["valid_path"], image_sha256=temp_visual_dir["valid_sha"],
        ),
        SubPart(label="b", marks=4, text="Explain search operation in B-tree.", topic="B-Trees", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}]),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Refer to Figure 1 for B-tree structure.\n\n(b) Explain search operation in B-tree.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx = AgentContext(plans=[plan], questions=[q], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res = checker.run(ctx)
    assert res.payload.status == STATUS_OK
    assert len(res.payload.blocked_reasons) == 0


def test_g4c_subpart_figure_sha256_mismatch_blocks(temp_visual_dir):
    """Gate 3: Bound figure with SHA-256 mismatch cannot pass validation and blocks."""
    ws_root = temp_visual_dir["workspace_root"]
    subparts = [
        SubPart(
            label="a", marks=6, text="Refer to Figure 1 for B-tree structure.",
            topic="B-Trees", solution="S", marking_scheme=[{"criterion": "C", "marks": 6}],
            figure_id="fig_btree", image_path=temp_visual_dir["valid_path"], image_sha256="wrong_expected_sha256_hash",
        ),
        SubPart(label="b", marks=4, text="Explain search operation in B-tree.", topic="B-Trees", solution="S", marking_scheme=[{"criterion": "C", "marks": 4}]),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Refer to Figure 1 for B-tree structure.\n\n(b) Explain search operation in B-tree.",
        solution="S", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], sub_parts=subparts,
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx = AgentContext(plans=[plan], questions=[q], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res = checker.run(ctx)
    assert res.payload.status == STATUS_BLOCKED
    assert any("HASH_MISMATCH" in r for r in res.payload.blocked_reasons)


# =============================================================================
# Integration & DOCX Export Tests
# =============================================================================

def test_g4_docx_export_renders_distinct_subquestion_bloom_co(tmp_path):
    """Verify DOCX table renders distinct per-subquestion Bloom and CO values."""
    subparts = [
        SubPart(label="a", marks=6, text="Explain stack operations in detail.", topic="Stacks", solution="Sol", marking_scheme=[{"criterion": "C", "marks": 6}], bloom="L2", co="CO1"),
        SubPart(label="b", marks=4, text="Analyze circular queue asymptotic complexity.", topic="Queues", solution="Sol", marking_scheme=[{"criterion": "C", "marks": 4}], bloom="L4", co="CO3"),
    ]
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) Explain stack operations in detail.\n\n(b) Analyze circular queue asymptotic complexity.",
        solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[6, 4], bloom="L4", co="CO3", sub_parts=subparts,
    )
    adapted_q = _to_question_dict(q)
    adapted_q["question_number"] = 1

    paper_data = {
        "assessmentType": "IA",
        "subject": "Data Structures",
        "subjectCode": "BCS301",
        "questions": [adapted_q],
        "courseOutcomes": ["CO1: Basics", "CO2: Linked Lists", "CO3: Queues"],
    }

    out_docx = tmp_path / "test_g4_export.docx"
    render_docx_document(paper_data, output_path=str(out_docx))
    assert out_docx.exists()
    assert out_docx.stat().st_size > 0

    doc = Document(str(out_docx))
    table = doc.tables[1]  # Question paper table

    # Locate rows for subparts (a) and (b)
    row_a = None
    row_b = None
    for row in table.rows:
        label_text = row.cells[1].text.strip()
        if label_text == "(a)":
            row_a = row
        elif label_text == "(b)":
            row_b = row

    assert row_a is not None, "Row for subpart (a) not found in table"
    assert row_b is not None, "Row for subpart (b) not found in table"

    # Row for subquestion (a): marks 6, CO1, L2
    assert row_a.cells[3].text.strip() == "6"
    assert row_a.cells[4].text.strip() == "CO1"
    assert row_a.cells[5].text.strip() == "L2"

    # Row for subquestion (b): marks 4, CO3, L4
    assert row_b.cells[3].text.strip() == "4"
    assert row_b.cells[4].text.strip() == "CO3"
    assert row_b.cells[5].text.strip() == "L4"


def test_g4_legacy_single_part_compatibility(tmp_path):
    """Verify single-part question compatibility in adaptation and export."""
    q = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Explain Dijkstra's shortest path algorithm.",
        solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[10], bloom="L3", co="CO2", sub_parts=None,
    )
    adapted = _to_question_dict(q)
    assert not adapted["is_unresolved"]
    assert len(adapted["sub_questions"]) == 1
    assert adapted["sub_questions"][0]["marks"] == 10
    assert adapted["sub_questions"][0]["bloom"] == "L3"
    assert adapted["sub_questions"][0]["co"] == "CO2"

    adapted["question_number"] = 1
    paper_data = {
        "assessmentType": "IA",
        "subject": "Algorithms",
        "subjectCode": "BCS302",
        "questions": [adapted],
    }
    out_docx = tmp_path / "test_g4_single_export.docx"
    render_docx_document(paper_data, output_path=str(out_docx))
    assert out_docx.exists()
    assert out_docx.stat().st_size > 0
