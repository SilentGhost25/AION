# tests/integration/test_multi_subject_acceptance.py
"""
Multi-Subject End-to-End Acceptance Integration Suite (PR 1 Merge Gate)
========================================================================
Validates:
  - V1: Multi-subject acceptance across Sat Com, Cloud Computing, and an unseen subject (ELECTIVE_3MOD)
  - G1: Marks allocation is deterministic (exact match with requested marks_split)
  - G2: CO distribution is uniform (dynamic module-based mapping from build_co_map)
  - G3: Fail-closed path active (no emergency salvage backdoor, blocks on quality violation)
  - V2: DEGRADED mode contract and DOCX export surfacing
"""

import os
import json
from unittest.mock import patch, MagicMock

import pytest

from core.contracts.paper_spec import PaperSpec
from core.generation.paper_spec_resolver import resolve_paper_spec
from core.contracts.question_slot import QuestionSlot, SlotStatus, UnresolvedSlotException
from core.contracts.question import GeneratedQuestion
from core.contracts.budgets import AnswerBudget, QuestionBudget
from core.contracts.task_signature import TaskSignature
from core.generation.output_schema import QuestionOutput
from core.validation.export_gate import ExportGate
from v0_1.docx_export import generate_docx_from_paper
from v0_1.question_schema import GeneratedPaper, Module, MainQuestion, SubQuestion
from aion_api import _format_paper


def mock_valid_llm_response(prompt: str, evidence_pack=None):
    """Generate a clean, valid question schema that satisfies all linter rules."""
    import re
    verb_m = re.search(r"Bloom Verb:\s*([A-Za-z]+)", str(prompt))
    bloom_verb = verb_m.group(1).capitalize() if verb_m else "Explain"
    topic_m = re.search(r"Topic:\s*([^\n]+)", str(prompt))
    topic = topic_m.group(1).strip() if topic_m else "the primary architecture"

    text = f"{bloom_verb} the foundational concepts and operational characteristics of {topic} in detail."
    return json.dumps({
        "instruction": text,
        "question_text": text,
        "math_blocks": [],
        "bloom_verb": bloom_verb,
    })


def _run_subject_acceptance(subject_name: str, exam_type: str, requested_marks_split: list, module_count: int, questions_per_module: int):
    # 1. Resolve PaperSpec
    spec = resolve_paper_spec(
        exam_type=exam_type,
        override={"subject": subject_name, "marks_split": requested_marks_split},
    )
    assert spec.exam_type.upper() == exam_type.upper()
    assert spec.module_count == module_count
    assert spec.questions_per_module == questions_per_module

    # 2. Allocate Partitions
    allocated = spec.allocate_partitions(requested_marks_split)
    assert len(allocated) == spec.total_questions

    # 3. Simulate Questions with SlotOrchestrator
    from core.generation.orchestrator import SlotOrchestrator
    orch = SlotOrchestrator()
    orch._call_llm = MagicMock(side_effect=mock_valid_llm_response)

    raw_modules = []
    all_gqs = []
    co_map = spec.build_co_map()

    from core.validation.linter import LintReport
    with patch("core.generation.orchestrator.run_linter", return_value=LintReport("clean", {})):
        for m_zero in range(spec.module_count):
            mod_idx = m_zero + 1
            mod_co = co_map[str(mod_idx)]
            mod_questions = []

            for q_zero in range(spec.questions_per_module):
                global_q_no = m_zero * spec.questions_per_module + (q_zero + 1)
                partition = allocated[(m_zero, q_zero)]
                sub_questions = []

                for s_idx, s_marks in enumerate(partition):
                    sub_label = chr(ord('a') + s_idx)
                    slot_id = f"mod_{mod_idx}_Q{global_q_no}_{sub_label}"
                    slot = QuestionSlot(
                        slot_id=slot_id,
                        question_no=global_q_no,
                        sub_label=sub_label,
                        or_pair_id=f"OR_{m_zero + 1}",
                        is_alternative=(q_zero % 2 == 1),
                        module_id=mod_idx,
                        marks=s_marks,
                        bloom_level="L2" if s_idx == 0 else "L4",
                        bloom_verb="Explain" if s_idx == 0 else "Analyze",
                        bloom_operation="UNDERSTAND" if s_idx == 0 else "ANALYZE",
                        co=mod_co,
                        difficulty="MEDIUM",
                        question_type="descriptive",
                        topic=f"{subject_name} Unit {mod_idx}",
                        evidence_ids=(f"chunk_{mod_idx}_{global_q_no}",),
                        answer_budget=AnswerBudget.from_marks_and_bloom(s_marks, "L2" if s_idx == 0 else "L4"),
                        question_budget=QuestionBudget.from_bloom("L2" if s_idx == 0 else "L4", s_marks),
                        task_signature=TaskSignature.from_bloom_marks_type("L2" if s_idx == 0 else "L4", s_marks, "descriptive"),
                        math_required=False,
                        visual_required=False,
                        status=SlotStatus.PENDING.value,
                    )

                    gq = orch.generate(slot, f"Evidence text for {subject_name} module {mod_idx} question {global_q_no}")
                    assert gq.status in ("VALIDATED", "PASS")
                    all_gqs.append(gq)

                    sub_questions.append({
                        "letter": sub_label,
                        "text": gq.output.question_text if hasattr(gq, "output") else gq.question_text,
                        "marks": s_marks,
                        "co": mod_co,
                        "bloom": "L2" if s_idx == 0 else "L4",
                    })

                mod_questions.append({
                    "mq_index": global_q_no,
                    "total_marks": sum(partition),
                    "bloom_level": 2,
                    "bloom_name": "Understand",
                    "sub_questions": sub_questions,
                    "is_or": (q_zero % 2 == 1),
                })

            raw_modules.append({
                "module_index": mod_idx,
                "module_title": f"Module {mod_idx}: {subject_name} Domain {mod_idx}",
                "questions": mod_questions,
            })

    # 4. Check G3: Export Gate validation
    export_decision = ExportGate.validate(all_gqs)
    assert export_decision.passed is True
    if export_decision.message:
        assert "SALVAGE" not in export_decision.message

    # 5. Format Paper through Unified Schema
    qa_report = {"paper_status": "OK", "degraded_reasons": []}
    formatted = _format_paper(raw_modules, subject_name, exam_type, "turbo", qa_report=qa_report)

    # 6. Check G1: Exact Marks Allocation Match
    formatted_modules = formatted["modules"]
    q_counter = 0
    for mod in formatted_modules:
        for q in mod["questions"]:
            expected_split = requested_marks_split[q_counter]
            actual_split = [sq["marks"] for sq in q["subQuestions"]]
            assert actual_split == expected_split, f"Q{q_counter+1} marks mismatch: actual {actual_split} != expected {expected_split}"
            assert q["totalMarks"] == sum(expected_split)
            q_counter += 1
    assert q_counter == len(requested_marks_split)

    # 7. Check G2: Dynamic CO Distribution
    co_cov = formatted.get("coCoverage", {})
    if spec.co_count == 5 and spec.module_count == 5:
        for co_idx in range(1, 6):
            co_key = f"co{co_idx}"
            assert co_cov.get(co_key, 0) == 20
    elif spec.co_count == 3 and spec.module_count == 3:
        for co_idx in range(1, 4):
            co_key = f"co{co_idx}"
            assert co_cov.get(co_key, 0) in (33, 34)

    # 8. Render DOCX
    docx_buf = generate_docx_from_paper(formatted)
    assert docx_buf is not None
    assert len(docx_buf.getvalue()) > 5000


def test_acceptance_satellite_communication_iat1():
    """V1 Acceptance: Satellite Communication (IAT1) passes marks allocation and uniform CO mapping."""
    _run_subject_acceptance(
        subject_name="Satellite Communication",
        exam_type="IAT1",
        requested_marks_split=[[10], [10], [6, 4], [6, 4], [6, 4], [6, 4], [10], [10], [10], [10]],
        module_count=5,
        questions_per_module=2
    )


def test_acceptance_cloud_computing_iat2():
    """V1 Acceptance: Cloud Computing (IAT2) passes marks allocation and uniform CO mapping."""
    _run_subject_acceptance(
        subject_name="Cloud Computing",
        exam_type="IAT2",
        requested_marks_split=[[6, 4], [6, 4], [10], [10], [6, 4], [6, 4], [6, 4], [6, 4], [10], [10]],
        module_count=5,
        questions_per_module=2
    )


def test_acceptance_unseen_subject_elective_3mod():
    """V1 Acceptance: Unseen Subject (ELECTIVE_3MOD) passes dynamic PaperSpec and allocation."""
    _run_subject_acceptance(
        subject_name="VLSI Design",
        exam_type="ELECTIVE_3MOD",
        requested_marks_split=[[10], [6, 4], [5, 5], [10], [6, 4], [5, 5], [10], [6, 4], [5, 5]],
        module_count=3,
        questions_per_module=3
    )


def test_g3_fail_closed_on_slot_defect():
    """G3 Acceptance: Injected defect triggers fail-closed UnresolvedSlotException without salvage."""
    from core.generation.orchestrator import SlotOrchestrator
    from core.validation.common import CheckResult
    from core.validation.linter import LintReport

    orch = SlotOrchestrator()
    slot = QuestionSlot(
        slot_id="mod_1_Q1_defect",
        question_no=1,
        sub_label="a",
        or_pair_id="OR_1",
        is_alternative=False,
        module_id=1,
        marks=10,
        bloom_level="L2",
        bloom_verb="Explain",
        bloom_operation="UNDERSTAND",
        co="CO1",
        difficulty="MEDIUM",
        question_type="descriptive",
        topic="Sat Com defect slot",
        evidence_ids=("c1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(10, "L2"),
        question_budget=QuestionBudget.from_bloom("L2", 10),
        task_signature=TaskSignature.from_bloom_marks_type("L2", 10, "descriptive"),
        math_required=False,
        visual_required=False,
        status=SlotStatus.PENDING.value,
    )

    orch._call_llm = MagicMock(return_value='{"instruction": "Contaminated text (i) and (ii)", "question_text": "Contaminated", "math_blocks": []}')

    failing_report = LintReport(slot.slot_id, {
        "multi_slot_contamination": CheckResult.fail("MULTI_SLOT_CONTAMINATION", "Slot module_1_Q1 has multi-slot contamination")
    })

    with patch("core.generation.orchestrator.run_linter", return_value=failing_report):
        with pytest.raises(UnresolvedSlotException) as exc_info:
            orch.generate(slot, "Evidence text")

        assert exc_info.value.slot.slot_id == "mod_1_Q1_defect"
        assert exc_info.value.slot.status == SlotStatus.UNRESOLVED.value
        assert exc_info.value.failure_code in ("MULTI_SLOT_CONTAMINATION", "CONTENT_DEFECT")

    # Verify ExportGate strictly blocks this slot fail-closed
    stub = GeneratedQuestion.create_unresolved(slot, failure_code="MULTI_SLOT_CONTAMINATION", reason="Contamination")
    gate_res = ExportGate.validate([stub])
    assert gate_res.passed is False
    assert gate_res.code == "UNRESOLVED_SLOT_DETECTED"
