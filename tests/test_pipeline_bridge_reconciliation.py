"""
Unit and Integration Tests for Phase 4 Stage 2C Phase 2:
Pipeline Bridge Post-Adaptation Reconciliation & Export Eligibility Enforcement.

Covers all 129 deterministic test specifications from the implementation plan:
- Normal reconciliation & population accounting (Tests 1-25)
- Adaptation defects & lifecycle validation (Tests 26-55)
- Safe serialization & JSON snapshot invariants (Tests 56-75)
- Early failure handlers & orchestrator boundaries (Tests 76-96)
- Emergency failure containment & diagnostic preservation (Tests 97-114)
- Legacy runner representation equivalence (Tests 115-121)
- Revision 31/32 regressions (Tests 122-129)
"""

from __future__ import annotations

import collections.abc
import copy
import json
import math
import unittest.mock
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import MagicMock

import pytest

from core.contracts.paper_spec import PaperSpec
from core.generation.agents.base import AgentContext
from core.generation.agents.checking_contracts import (
    BLOCK_REASON_INSUFFICIENT_QUESTIONS,
    BLOCK_REASON_UNRESOLVED_SLOTS,
    QAReport,
    STATUS_BLOCKED,
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_PASS,
)
from core.generation.agents.dependency_factory import DependencyFactory, FactoryResult
from core.generation.agents.generated_question import GeneratedQuestion
from core.generation.agents.orchestrator_contracts import OrchestratorResult
from core.generation.agents.pipeline_bridge import (
    PipelineBridgeResult,
    _build_emergency_meta_dict,
    _build_pipeline_meta,
    _finalize_reconciliation,
    _handle_adaptation_execution_error,
    _handle_context_construction_exception,
    _handle_dependency_factory_failure,
    _handle_emergency_pipeline_failure,
    _handle_initialization_exception,
    _handle_orchestrator_exception,
    _reconcile_post_adaptation,
    _resolve_min_ratio,
    _safe_dict_convert,
    _safe_float_ratio,
    _safe_json_snapshot,
    _safe_list_convert,
    _safe_repr,
    _safe_reset_evaluation,
    _safe_serialize_qa_report,
    _safe_str,
    _safe_telemetry_copy,
    _safe_type_name,
    _safe_wrap_bridge_result,
    _validate_and_extract_reasons,
    _validate_stage2a_lifecycle_contract,
    run_pipeline_v3_or_legacy,
    run_v3_pipeline,
)


# -----------------------------------------------------------------------------
# Fixtures & Test Helpers
# -----------------------------------------------------------------------------


@dataclass
class DummyPlan:
    slot_id: str


def make_test_paper_spec(module_count: int = 2, questions_per_module: int = 5) -> PaperSpec:
    return PaperSpec(
        exam_type="IAT1",
        module_count=module_count,
        questions_per_module=questions_per_module,
        marks_per_question=10,
    )


def make_expected_slot_ids(module_count: int = 2, questions_per_module: int = 5) -> List[str]:
    slots = []
    for m in range(1, module_count + 1):
        for q in range(1, questions_per_module + 1):
            idx = (m - 1) * questions_per_module + q
            slots.append(f"module_{m}_Q{idx}")
    return slots


def make_test_context(
    module_count: int = 2,
    questions_per_module: int = 5,
    request_overrides: Optional[dict] = None,
) -> AgentContext:
    spec = make_test_paper_spec(module_count, questions_per_module)
    expected_ids = make_expected_slot_ids(module_count, questions_per_module)
    plans = [DummyPlan(slot_id=sid) for sid in expected_ids]
    req = {"exam_type": "IAT1", "subject": "Computer Science"}
    if request_overrides:
        req.update(request_overrides)
    ctx = AgentContext(
        request=req,
        paper_spec=spec,
        artifact=None,
        marks_split=[[10]] * len(expected_ids),
    )
    ctx.plans = plans
    return ctx


def make_clean_question(
    slot_id: str,
    module_id: str = "module_1",
    global_q_idx: int = 1,
    question_text: str = "Explain the architecture of a transformer model.",
    solution: str = "The transformer model consists of an encoder and decoder...",
    is_unresolved: bool = False,
    unresolved_scope: Optional[str] = None,
    unresolved_subparts: Any = None,
    failure_code: Optional[str] = None,
    failure_detail: Optional[str] = None,
) -> GeneratedQuestion:
    q = GeneratedQuestion(
        slot_id=slot_id,
        module_id=module_id,
        global_q_idx=global_q_idx,
        question_text=question_text,
        solution=solution,
        marking_scheme=[{"points": 10, "criterion": "Complete and accurate"}],
        marks=10,
        bloom="L2",
    )
    q.is_unresolved = is_unresolved
    q.unresolved_scope = unresolved_scope
    q.unresolved_subparts = unresolved_subparts
    q.failure_code = failure_code
    q.failure_detail = failure_detail
    return q


def make_clean_questions(expected_ids: List[str]) -> List[GeneratedQuestion]:
    questions = []
    for idx, sid in enumerate(expected_ids):
        # Extract module_id from sid (e.g. 'module_1_Q1' -> 'module_1')
        parts = sid.split("_")
        mod_id = f"{parts[0]}_{parts[1]}"
        q = make_clean_question(
            slot_id=sid,
            module_id=mod_id,
            global_q_idx=idx + 1,
            question_text=f"Question text for {sid}",
            solution=f"Solution for {sid}",
        )
        questions.append(q)
    return questions


def make_clean_qa_report(
    expected_ids: List[str],
    telemetry: Optional[dict] = None,
    status: str = STATUS_OK,
    unresolved_slots: Optional[List[str]] = None,
    degraded_reasons: Optional[List[str]] = None,
    blocked_reasons: Optional[List[str]] = None,
) -> QAReport:
    tel = {
        "checking": {"min_questions_ratio": 0.9},
        **(telemetry or {}),
    }
    unres = list(unresolved_slots or [])
    res_count = len(expected_ids) - len(unres)
    return QAReport(
        status=status,
        slot_count=len(expected_ids),
        resolved_slots=res_count,
        unresolved_slots=unres,
        degraded_reasons=list(degraded_reasons or []),
        blocked_reasons=list(blocked_reasons or []),
        telemetry=tel,
        cost_telemetry={},
        cache_telemetry={},
    )


def make_clean_orchestrator_result(
    questions: List[GeneratedQuestion],
    qa_report: QAReport,
    success: bool = True,
    failure_code: Optional[str] = None,
    failure_detail: Optional[str] = None,
    telemetry: Optional[dict] = None,
) -> OrchestratorResult:
    if not success and failure_code is None:
        failure_code = "ORCHESTRATOR_EXECUTION_FAILURE"
    return OrchestratorResult(
        success=success,
        questions=questions,
        qa_report=qa_report,
        failure_code=failure_code,
        failure_detail=failure_detail,
        telemetry=telemetry or {},
    )


def make_clean_adapted_paper(
    questions: List[GeneratedQuestion],
) -> Tuple[List[dict], List[dict]]:
    paper_parts = []
    full_paper = []
    by_mod: Dict[str, List[dict]] = {}

    for q in questions:
        qd = {
            "slot_id": q.slot_id,
            "module_id": q.module_id,
            "global_q_idx": q.global_q_idx,
            "question_text": q.question_text,
            "solution": q.solution,
            "marks": q.marks,
            "bloom": q.bloom,
            "is_unresolved": getattr(q, "is_unresolved", False),
            "adaptation_failure_code": None,
            "adaptation_failure_detail": None,
        }
        full_paper.append(qd)
        by_mod.setdefault(q.module_id, []).append(qd)

    for mod_id, q_list in by_mod.items():
        paper_parts.append({
            "module_id": mod_id,
            "questions": q_list,
        })

    return paper_parts, full_paper


# -----------------------------------------------------------------------------
# Tests 1 - 25: Normal Reconciliation & Population Accounting
# -----------------------------------------------------------------------------


def test_pipeline_bridge_no_adaptation_failures_preserves_semantic_equality():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, ad_defects, rec_meta, pre_qa_s, rec_qa_s, res_tel, eligible = (
        _reconcile_post_adaptation(result=res, context=ctx, paper_parts=parts, full_paper=full)
    )

    assert reconciled_qa.status == STATUS_OK
    assert reconciled_qa.slot_count == 10
    assert reconciled_qa.resolved_slots == 10
    assert reconciled_qa.unresolved_slots == []
    assert reconciled_qa.blocked_reasons == []
    assert reconciled_qa.degraded_reasons == []
    assert len(ad_defects) == 0
    assert rec_meta["expected_slots"] == 10
    assert rec_meta["resolved_slots"] == 10
    assert rec_meta["unresolved_slots"] == 0
    assert rec_meta["fatal_defects"] == []
    assert eligible is True
    assert rec_qa_s["status"] == STATUS_OK


def test_pipeline_bridge_one_clean_slot_becomes_adaptation_failed_detected():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Clean Q1 fails adaptation
    full[0]["adaptation_failure_code"] = "MULTI_PART_STRUCTURE_MISMATCH"
    full[0]["adaptation_failure_detail"] = "Failed to parse subparts"
    full[0]["is_unresolved"] = True
    parts[0]["questions"][0]["adaptation_failure_code"] = "MULTI_PART_STRUCTURE_MISMATCH"
    parts[0]["questions"][0]["adaptation_failure_detail"] = "Failed to parse subparts"
    parts[0]["questions"][0]["is_unresolved"] = True

    reconciled_qa, ad_defects, rec_meta, pre_qa_s, rec_qa_s, res_tel, eligible = (
        _reconcile_post_adaptation(result=res, context=ctx, paper_parts=parts, full_paper=full)
    )

    assert reconciled_qa.status == STATUS_PARTIAL
    assert reconciled_qa.resolved_slots == 9
    assert reconciled_qa.unresolved_slots == ["module_1_Q1"]
    assert len(ad_defects) == 1
    assert "ADAPTATION_FAILURE:module_1_Q1:MULTI_PART_STRUCTURE_MISMATCH" in ad_defects
    assert eligible is True  # 9/10 = 0.9 >= 0.9 min_ratio
    assert rec_meta["fatal_defects"] == []


def test_pipeline_bridge_multiple_adaptation_failures_all_retained():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Q1 and Q2 fail adaptation
    for i in (0, 1):
        full[i]["adaptation_failure_code"] = "ADAPT_FAIL"
        full[i]["adaptation_failure_detail"] = f"Detail {i}"
        full[i]["is_unresolved"] = True
        parts[0]["questions"][i]["adaptation_failure_code"] = "ADAPT_FAIL"
        parts[0]["questions"][i]["adaptation_failure_detail"] = f"Detail {i}"
        parts[0]["questions"][i]["is_unresolved"] = True

    reconciled_qa, ad_defects, rec_meta, pre_qa_s, rec_qa_s, res_tel, eligible = (
        _reconcile_post_adaptation(result=res, context=ctx, paper_parts=parts, full_paper=full)
    )

    assert "module_1_Q1" in reconciled_qa.unresolved_slots
    assert "module_1_Q2" in reconciled_qa.unresolved_slots
    assert len(ad_defects) == 2
    # 8/10 = 0.8 < 0.9 threshold on non-degraded audit -> BLOCKED
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False


def test_pipeline_bridge_partially_unresolved_slot_adaptation_failure_no_duplicate_accounting():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    # Q1 was partially unresolved in pre-adaptation (in P_pre)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = ["b"]
    questions[0].failure_code = "PARTIAL_CODE"
    questions[0].failure_detail = "Part b failed"
    ctx.questions = questions

    qa = make_clean_qa_report(
        expected_ids,
        status=STATUS_PARTIAL,
        unresolved_slots=["module_1_Q1"],
    )
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Q1 additionally fails adaptation
    full[0]["adaptation_failure_code"] = "ADAPT_FAIL"
    full[0]["adaptation_failure_detail"] = "Adapter failed"
    full[0]["is_unresolved"] = True
    parts[0]["questions"][0]["adaptation_failure_code"] = "ADAPT_FAIL"
    parts[0]["questions"][0]["adaptation_failure_detail"] = "Adapter failed"
    parts[0]["questions"][0]["is_unresolved"] = True

    reconciled_qa, ad_defects, rec_meta, pre_qa_s, rec_qa_s, res_tel, eligible = (
        _reconcile_post_adaptation(result=res, context=ctx, paper_parts=parts, full_paper=full)
    )

    # Q1 moved to B_post, counted exactly once in unresolved_slots
    assert reconciled_qa.unresolved_slots.count("module_1_Q1") == 1
    assert reconciled_qa.resolved_slots == 9
    assert rec_meta["b_post"] == ["module_1_Q1"]
    assert rec_meta["p_post"] == []


def test_pipeline_bridge_existing_partial_report_cannot_become_ok():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = ["b"]
    questions[0].failure_code = "PARTIAL_CODE"
    questions[0].failure_detail = "Part b failed"
    ctx.questions = questions

    qa = make_clean_qa_report(
        expected_ids,
        status=STATUS_PARTIAL,
        unresolved_slots=["module_1_Q1"],
    )
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_PARTIAL
    assert eligible is True


def test_pipeline_bridge_existing_degraded_report_preserves_degradation():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions

    qa = make_clean_qa_report(
        expected_ids,
        status=STATUS_DEGRADED,
        degraded_reasons=["UNVERIFIED_EVALUATION"],
    )
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_DEGRADED
    assert "UNVERIFIED_EVALUATION" in reconciled_qa.degraded_reasons
    assert eligible is True


def test_pipeline_bridge_existing_blocked_report_remains_blocked():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    for q in questions:
        q.is_unresolved = True
        q.unresolved_scope = "question"
        q.failure_code = "WRITING_FAILED"
        q.failure_detail = "Failed"
    ctx.questions = questions

    qa = make_clean_qa_report(
        expected_ids,
        status=STATUS_BLOCKED,
        unresolved_slots=list(expected_ids),
        blocked_reasons=[BLOCK_REASON_UNRESOLVED_SLOTS],
    )
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert BLOCK_REASON_UNRESOLVED_SLOTS in reconciled_qa.blocked_reasons


def test_pipeline_bridge_missing_adapted_slot_integrity_failure_detected():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Omit module_1_Q2 from full_paper and paper_parts
    full = [q for q in full if q["slot_id"] != "module_1_Q2"]
    parts[0]["questions"] = [q for q in parts[0]["questions"] if q["slot_id"] != "module_1_Q2"]

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("MISSING_ADAPTED_SLOT:module_1_Q2" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_duplicate_adapted_slot_integrity_failure_detected():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Duplicate module_1_Q1 in full_paper
    dup_q = dict(full[0])
    full.append(dup_q)
    parts[0]["questions"].append(dup_q)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("DUPLICATE_ADAPTED_SLOT_ID:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_unexpected_adapted_slot_integrity_failure_detected():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Extra unexpected slot
    extra = dict(full[0])
    extra["slot_id"] = "unexpected_module_X_Q99"
    full.append(extra)
    parts[0]["questions"].append(extra)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("UNEXPECTED_ADAPTED_SLOT_ID:unexpected_module_X_Q99" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_empty_adapted_paper_diagnostics_remain_accessible():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)

    # Both empty
    parts, full = [], []

    reconciled_qa, _, rec_meta, pre_qa_s, rec_qa_s, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert rec_qa_s is not None
    assert pre_qa_s is not None
    assert pre_qa_s["status"] == STATUS_OK


def test_pipeline_bridge_pre_adaptation_report_remains_independently_available():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    full[0]["adaptation_failure_code"] = "MULTI_PART_STRUCTURE_MISMATCH"
    full[0]["adaptation_failure_detail"] = "Failed"
    full[0]["is_unresolved"] = True
    parts[0]["questions"][0]["adaptation_failure_code"] = "MULTI_PART_STRUCTURE_MISMATCH"
    parts[0]["questions"][0]["adaptation_failure_detail"] = "Failed"
    parts[0]["questions"][0]["is_unresolved"] = True

    reconciled_qa, _, _, pre_qa_s, rec_qa_s, _, _ = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert pre_qa_s["status"] == STATUS_OK
    assert rec_qa_s["status"] == STATUS_PARTIAL


def test_pipeline_bridge_execution_success_with_blocked_qa_status():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Introduce a fatal defect in adapted record (whitespace slot_id)
    full[0]["slot_id"] = " module_1_Q1 "
    parts[0]["questions"][0]["slot_id"] = " module_1_Q1 "

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    meta = _build_pipeline_meta(
        validated_exec_success=True,
        raw_success=True,
        paper_status=reconciled_qa.status,
        export_eligible=eligible,
        failure_code=None,
        failure_detail=None,
        telemetry={},
        factory_notes=[],
        qa_report=reconciled_qa.to_dict(),
        pre_adaptation_qa_report=None,
        adaptation_defects=[],
        reconciliation_meta={},
        blocked_reasons=reconciled_qa.blocked_reasons,
    )

    assert meta["success"] is True
    assert meta["export_eligible"] is False
    assert meta["paper_status"] == STATUS_BLOCKED


def test_pipeline_bridge_adaptation_failure_at_or_above_ratio_threshold():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # 1 failure out of 10: 9/10 = 0.9 >= 0.9 min_ratio
    full[0]["adaptation_failure_code"] = "FAIL"
    full[0]["adaptation_failure_detail"] = "Fail"
    full[0]["is_unresolved"] = True
    parts[0]["questions"][0]["adaptation_failure_code"] = "FAIL"
    parts[0]["questions"][0]["adaptation_failure_detail"] = "Fail"
    parts[0]["questions"][0]["is_unresolved"] = True

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_PARTIAL
    assert eligible is True


def test_pipeline_bridge_adaptation_failure_below_ratio_threshold():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # 2 failures out of 10: 8/10 = 0.8 < 0.9 min_ratio
    for i in (0, 1):
        full[i]["adaptation_failure_code"] = "FAIL"
        full[i]["adaptation_failure_detail"] = "Fail"
        full[i]["is_unresolved"] = True
        parts[0]["questions"][i]["adaptation_failure_code"] = "FAIL"
        parts[0]["questions"][i]["adaptation_failure_detail"] = "Fail"
        parts[0]["questions"][i]["is_unresolved"] = True

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False


def test_pipeline_bridge_pre_existing_partial_below_ratio_no_adaptation_failure_remains_partial():
    ctx = make_test_context()
    # Mock degraded audit mode on context
    ctx.audit_report = SimpleNamespace(degraded_mode=True)
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    # 2 pre-existing failures (ratio 0.8)
    for i in (0, 1):
        questions[i].is_unresolved = True
        questions[i].unresolved_scope = "subquestion"
        questions[i].unresolved_subparts = ["b"]
        questions[i].failure_code = "PARTIAL_CODE"
        questions[i].failure_detail = "Part b"
    ctx.questions = questions

    qa = make_clean_qa_report(
        expected_ids,
        status=STATUS_PARTIAL,
        unresolved_slots=["module_1_Q1", "module_1_Q2"],
    )
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    # 0 adaptation failures, resolved population not reduced -> remains PARTIAL
    assert reconciled_qa.status == STATUS_PARTIAL
    assert eligible is True


def test_pipeline_bridge_pre_existing_degraded_below_ratio_no_adaptation_failure_remains_degraded():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions

    qa = make_clean_qa_report(
        expected_ids,
        status=STATUS_DEGRADED,
        degraded_reasons=["UNVERIFIED_EVALUATION"],
    )
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_DEGRADED
    assert eligible is True


def test_pipeline_bridge_pre_existing_subquestion_failure_valid_lifecycle_preserved_in_p():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = ["b"]
    questions[0].failure_code = "PARTIAL_CODE"
    questions[0].failure_detail = "Subpart b failed"
    ctx.questions = questions

    qa = make_clean_qa_report(
        expected_ids,
        status=STATUS_PARTIAL,
        unresolved_slots=["module_1_Q1"],
    )
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert "module_1_Q1" in rec_meta["p_post"]
    assert "module_1_Q1" in reconciled_qa.unresolved_slots
    assert reconciled_qa.status == STATUS_PARTIAL
    assert eligible is True


def test_pipeline_bridge_adaptation_failure_fails_closed_into_b():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    full[0]["adaptation_failure_code"] = "PARSE_ERROR"
    full[0]["adaptation_failure_detail"] = "Could not parse"
    full[0]["is_unresolved"] = True
    parts[0]["questions"][0]["adaptation_failure_code"] = "PARSE_ERROR"
    parts[0]["questions"][0]["adaptation_failure_detail"] = "Could not parse"
    parts[0]["questions"][0]["is_unresolved"] = True

    _, _, rec_meta, _, _, _, _ = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    # Clean question failing adaptation strictly enters B_post
    assert "module_1_Q1" in rec_meta["b_post"]
    assert "module_1_Q1" not in rec_meta["p_post"]
    assert "module_1_Q1" not in rec_meta["f_post"]


def test_pipeline_bridge_whitespace_padded_slot_id_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    full[0]["slot_id"] = " module_1_Q1 "
    parts[0]["questions"][0]["slot_id"] = " module_1_Q1 "

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("MALFORMED_ADAPTED_SLOT_ID" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_non_dict_adapted_record_handled_safely():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    full[0] = "corrupted_string_not_dict"

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("MALFORMED_ADAPTED_RECORD" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_non_string_adaptation_failure_code_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    full[0]["adaptation_failure_code"] = 12345
    full[0]["is_unresolved"] = True
    parts[0]["questions"][0]["adaptation_failure_code"] = 12345
    parts[0]["questions"][0]["is_unresolved"] = True

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("MALFORMED_ADAPTATION_METADATA" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_unknown_adaptation_failure_code_preserved_and_handled():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    full[0]["adaptation_failure_code"] = "CUSTOM_VENDOR_ADAPTATION_ERROR"
    full[0]["adaptation_failure_detail"] = "Vendor detail"
    full[0]["is_unresolved"] = True
    parts[0]["questions"][0]["adaptation_failure_code"] = "CUSTOM_VENDOR_ADAPTATION_ERROR"
    parts[0]["questions"][0]["adaptation_failure_detail"] = "Vendor detail"
    parts[0]["questions"][0]["is_unresolved"] = True

    reconciled_qa, ad_defects, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert "ADAPTATION_FAILURE:module_1_Q1:CUSTOM_VENDOR_ADAPTATION_ERROR" in ad_defects
    assert "module_1_Q1" in rec_meta["b_post"]
    assert reconciled_qa.status == STATUS_PARTIAL
    assert eligible is True


def test_pipeline_bridge_telemetry_deep_copy_prevents_pre_qa_mutation():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    _reconcile_post_adaptation(result=res, context=ctx, paper_parts=parts, full_paper=full)

    # pre_qa.telemetry must not contain bridge_reconciliation
    assert "bridge_reconciliation" not in qa.telemetry


def test_pipeline_bridge_duplicate_unresolved_accounting_prevented():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = ["b"]
    questions[0].failure_code = "PARTIAL"
    questions[0].failure_detail = "Detail"
    ctx.questions = questions

    qa = make_clean_qa_report(expected_ids, status=STATUS_PARTIAL, unresolved_slots=["module_1_Q1"])
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Q1 also fails adaptation
    full[0]["adaptation_failure_code"] = "FAIL"
    full[0]["adaptation_failure_detail"] = "Fail"
    full[0]["is_unresolved"] = True
    parts[0]["questions"][0]["adaptation_failure_code"] = "FAIL"
    parts[0]["questions"][0]["adaptation_failure_detail"] = "Fail"
    parts[0]["questions"][0]["is_unresolved"] = True

    reconciled_qa, _, _, _, _, _, _ = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert len(reconciled_qa.unresolved_slots) == len(set(reconciled_qa.unresolved_slots))
    assert reconciled_qa.unresolved_slots.count("module_1_Q1") == 1


# -----------------------------------------------------------------------------
# Tests 26 - 60: Adaptation Defects, Pre-QA Invariants & Lifecycle Validation
# -----------------------------------------------------------------------------


def test_pipeline_bridge_missing_pre_adaptation_question_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    # Remove Q1 from questions
    questions = questions[1:]
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("MISSING_PRE_ADAPTATION_QUESTION:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_duplicate_pre_adaptation_question_ids_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    # Duplicate Q1
    dup_q = make_clean_question(slot_id="module_1_Q1", module_id="module_1", global_q_idx=1)
    questions.append(dup_q)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("DUPLICATE_PRE_ADAPTATION_QUESTION_ID:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_qa_ok_but_question_metadata_unresolved_contradiction():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    # Q1 is marked unresolved in questions but qa report says OK (0 unresolved)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].failure_code = "FAIL"
    questions[0].failure_detail = "Fail"
    ctx.questions = questions

    qa = make_clean_qa_report(expected_ids, status=STATUS_OK, unresolved_slots=[])
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("PRE_ADAPTATION_CONTRADICTION:module_1_Q1:question_unresolved_but_not_in_qa_report" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_qa_unresolved_but_question_metadata_clean_contradiction():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    # Q1 is clean in questions, but qa report lists Q1 in unresolved_slots
    ctx.questions = questions

    qa = make_clean_qa_report(expected_ids, status=STATUS_PARTIAL, unresolved_slots=["module_1_Q1"])
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("PRE_ADAPTATION_CONTRADICTION:module_1_Q1:qa_report_unresolved_but_question_clean" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_clean_slot_becomes_unresolved_without_adaptation_code_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Adapted record has is_unresolved=True but adaptation_failure_code=None
    full[0]["is_unresolved"] = True
    full[0]["adaptation_failure_code"] = None
    parts[0]["questions"][0]["is_unresolved"] = True
    parts[0]["questions"][0]["adaptation_failure_code"] = None

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("UNEXPECTED_UNRESOLVED_ADAPTED_SLOT:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_adapted_is_unresolved_invalid_type_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # is_unresolved is string "true" instead of bool
    full[0]["is_unresolved"] = "true"
    parts[0]["questions"][0]["is_unresolved"] = "true"

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("MALFORMED_ADAPTED_RECORD_METADATA:module_1_Q1:is_unresolved_not_boolean" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_unresolved_p_slot_silently_cleared_inconsistency():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    # Q1 was unresolved subquestion in P_pre
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = ["b"]
    questions[0].failure_code = "PARTIAL"
    questions[0].failure_detail = "Part b"
    ctx.questions = questions

    qa = make_clean_qa_report(expected_ids, status=STATUS_PARTIAL, unresolved_slots=["module_1_Q1"])
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Adapted output silently clears is_unresolved to False without repair
    full[0]["is_unresolved"] = False
    parts[0]["questions"][0]["is_unresolved"] = False

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("UNRESOLVED_STATE_SILENTLY_CLEARED:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_adaptation_detail_without_failure_code_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # detail present but code is None
    full[0]["adaptation_failure_code"] = None
    full[0]["adaptation_failure_detail"] = "Some detail"
    parts[0]["questions"][0]["adaptation_failure_code"] = None
    parts[0]["questions"][0]["adaptation_failure_detail"] = "Some detail"

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("ORPHANED_ADAPTATION_DETAIL:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_adaptation_failure_on_p_slot_does_not_reduce_f_population():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = ["b"]
    questions[0].failure_code = "PARTIAL"
    questions[0].failure_detail = "Part b"
    ctx.questions = questions

    qa = make_clean_qa_report(expected_ids, status=STATUS_PARTIAL, unresolved_slots=["module_1_Q1"])
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Q1 fails adaptation
    full[0]["adaptation_failure_code"] = "ADAPT_FAIL"
    full[0]["adaptation_failure_detail"] = "Fail"
    full[0]["is_unresolved"] = True
    parts[0]["questions"][0]["adaptation_failure_code"] = "ADAPT_FAIL"
    parts[0]["questions"][0]["adaptation_failure_detail"] = "Fail"
    parts[0]["questions"][0]["is_unresolved"] = True

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    # F population was 9 before and remains 9 after
    assert len(rec_meta["f_post"]) == 9
    assert reconciled_qa.status == STATUS_PARTIAL
    assert eligible is True


def test_pipeline_bridge_invalid_qa_telemetry_ratio_fatal_defect_no_silent_fallback():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(
        expected_ids,
        telemetry={"checking": {"min_questions_ratio": "invalid_string"}},
    )
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("INVALID_MIN_QUESTIONS_RATIO_CONFIG" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_adaptation_is_unresolved_invalid_type_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = "False"  # String instead of bool
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("MALFORMED_QUESTION_LIFECYCLE_FLAG:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_adaptation_unresolved_subparts_none_safe_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = None  # None for subquestion failure
    questions[0].failure_code = "PARTIAL"
    questions[0].failure_detail = "Fail"
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids, status=STATUS_PARTIAL, unresolved_slots=["module_1_Q1"])
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("INVALID_UNRESOLVED_SUBPARTS_TYPE:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_adaptation_unresolved_subparts_string_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = "ab"  # String instead of list/tuple
    questions[0].failure_code = "PARTIAL"
    questions[0].failure_detail = "Fail"
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids, status=STATUS_PARTIAL, unresolved_slots=["module_1_Q1"])
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("INVALID_UNRESOLVED_SUBPARTS_TYPE:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_qa_unresolved_slots_contains_duplicates_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].failure_code = "FAIL"
    questions[0].failure_detail = "Fail"
    ctx.questions = questions
    # Duplicates in unresolved_slots
    qa = make_clean_qa_report(
        expected_ids,
        status=STATUS_PARTIAL,
        unresolved_slots=["module_1_Q1", "module_1_Q1"],
    )
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("DUPLICATE_PRE_QA_UNRESOLVED_SLOTS" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_qa_unresolved_slots_contains_unexpected_id_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    # Unexpected ID in unresolved_slots
    qa = make_clean_qa_report(
        expected_ids,
        status=STATUS_PARTIAL,
        unresolved_slots=["module_99_Q1"],
    )
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("UNEXPECTED_PRE_QA_UNRESOLVED_SLOT_ID:module_99_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_qa_resolved_slots_invalid_type_or_range_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    qa.resolved_slots = -1  # Out of range
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("MALFORMED_PRE_QA_RESOLVED_SLOTS" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_qa_unknown_status_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    qa.status = "SUCCESS"  # Not a canonical status
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("INVALID_PRE_QA_STATUS:SUCCESS" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_slot_classification_order_independent_with_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    # Q5 has a fatal defect
    questions[4].is_unresolved = "False"
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    _, _, rec_meta1, _, _, _, _ = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    # Test order independence directly via _finalize_reconciliation with reversed partition lists
    f_post = [s for s in expected_ids if s != "module_1_Q5"]
    b_post = ["module_1_Q5"]

    _, _, rec_meta2, _, _, _, _ = _finalize_reconciliation(
        raw_success=True,
        exec_succeeded=True,
        expected_ids=list(reversed(expected_ids)),
        f_post=list(reversed(f_post)),
        p_post=[],
        b_post=b_post,
        fatal_defects=["MALFORMED_QUESTION_LIFECYCLE_FLAG:module_1_Q5"],
        adaptation_defects=[],
        pre_status=STATUS_OK,
        context=ctx,
        copied_telemetry={},
        copied_cost_telemetry={},
        copied_cache_telemetry={},
        copied_result_telemetry={},
        serialized_pre_qa=None,
        failure_code=None,
        validated_pre_blocked=[],
        validated_pre_degraded=[],
        resolved_population_changed=True,
        min_ratio=0.9,
        accounting_complete=True,
    )

    assert set(rec_meta1["f_post"]) == set(rec_meta2["f_post"])
    assert set(rec_meta1["b_post"]) == set(rec_meta2["b_post"])
    assert "module_1_Q5" in rec_meta1["b_post"]
    assert "module_1_Q5" in rec_meta2["b_post"]


def test_pipeline_bridge_expected_slot_validation_exception_handled_cleanly():
    spec = make_test_paper_spec()
    ctx = AgentContext(request={}, paper_spec=spec, artifact=None, marks_split=[[10]] * 10)
    ctx.plans = "not_a_list"  # Will trigger ValueError in _get_validated_expected_slots

    parts, full, meta = run_v3_pipeline(
        paper_spec=spec,
        artifact=None,
        marks_split=[[10]] * 10,
        request={},
    )

    assert parts == []
    assert full == []
    assert meta["success"] is False
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False


def test_pipeline_bridge_result_vs_context_questions_disagreement_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    q_res = make_clean_questions(expected_ids)
    q_ctx = make_clean_questions(expected_ids)
    # Lifecycle disagreement on Q1
    q_res[0].is_unresolved = True
    q_res[0].unresolved_scope = "question"
    q_res[0].failure_code = "FAIL"
    q_res[0].failure_detail = "Fail"
    q_ctx[0].is_unresolved = False

    ctx.questions = q_ctx
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(q_res, qa)
    parts, full = make_clean_adapted_paper(q_res)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("QUESTION_POPULATION_DISAGREEMENT:module_1_Q1:lifecycle_mismatch" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_qa_unresolved_slots_unhashable_entry_handled_safely():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    qa.unresolved_slots = ["module_1_Q1", ["module_1_Q2"], {"slot": "module_1_Q3"}]
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    # Zero uncaught TypeError
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("MALFORMED_PRE_QA_UNRESOLVED_SLOT_ENTRY" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_clean_question_malformed_subparts_type_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = False
    questions[0].unresolved_subparts = "ab"  # Unexpected on clean question
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("UNEXPECTED_UNRESOLVED_SUBPARTS:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_question_level_unresolved_malformed_subparts_type_fatal_defect():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].unresolved_subparts = ["b"]  # Unexpected subparts for question-level failure
    questions[0].failure_code = "FAIL"
    questions[0].failure_detail = "Fail"
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids, status=STATUS_PARTIAL, unresolved_slots=["module_1_Q1"])
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("UNEXPECTED_UNRESOLVED_SUBPARTS:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_orphaned_adaptation_detail_slot_enters_b_post():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    full[0]["adaptation_failure_code"] = None
    full[0]["adaptation_failure_detail"] = "orphaned detail"
    parts[0]["questions"][0]["adaptation_failure_code"] = None
    parts[0]["questions"][0]["adaptation_failure_detail"] = "orphaned detail"

    _, _, rec_meta, _, _, _, _ = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert "module_1_Q1" in rec_meta["b_post"]
    assert "module_1_Q1" not in rec_meta["f_post"]


def test_pipeline_bridge_non_string_adaptation_detail_slot_enters_b_post():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    full[0]["adaptation_failure_detail"] = 12345
    parts[0]["questions"][0]["adaptation_failure_detail"] = 12345

    _, _, rec_meta, _, _, _, _ = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert "module_1_Q1" in rec_meta["b_post"]
    assert "module_1_Q1" not in rec_meta["f_post"]


def test_pipeline_bridge_adaptation_failure_with_is_unresolved_false_inconsistent():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    full[0]["adaptation_failure_code"] = "MULTI_PART_STRUCTURE_MISMATCH"
    full[0]["is_unresolved"] = False
    parts[0]["questions"][0]["adaptation_failure_code"] = "MULTI_PART_STRUCTURE_MISMATCH"
    parts[0]["questions"][0]["is_unresolved"] = False

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("INCONSISTENT_ADAPTED_METADATA:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_result_vs_context_identical_ids_different_subparts_disagreement():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    q_res = make_clean_questions(expected_ids)
    q_ctx = make_clean_questions(expected_ids)

    q_res[0].is_unresolved = True
    q_res[0].unresolved_scope = "subquestion"
    q_res[0].unresolved_subparts = ["b"]
    q_res[0].failure_code = "FAIL"
    q_res[0].failure_detail = "Fail"

    q_ctx[0].is_unresolved = True
    q_ctx[0].unresolved_scope = "subquestion"
    q_ctx[0].unresolved_subparts = ["c"]
    q_ctx[0].failure_code = "FAIL"
    q_ctx[0].failure_detail = "Fail"

    ctx.questions = q_ctx
    qa = make_clean_qa_report(expected_ids, status=STATUS_PARTIAL, unresolved_slots=["module_1_Q1"])
    res = make_clean_orchestrator_result(q_res, qa)
    parts, full = make_clean_adapted_paper(q_res)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("QUESTION_POPULATION_DISAGREEMENT:module_1_Q1:lifecycle_mismatch" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_qa_status_unhashable_type_handled_safely():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    qa.status = ["OK"]  # List instead of str
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("INVALID_PRE_QA_STATUS_TYPE:list" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_result_vs_context_different_valid_ids_both_enter_b_post():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    q_res = make_clean_questions(expected_ids)
    q_ctx = make_clean_questions(expected_ids)

    # Swap slot_id at index 0 and 1 between result and context
    q_res[0].slot_id = "module_1_Q2"
    q_res[1].slot_id = "module_1_Q1"

    ctx.questions = q_ctx
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(q_res, qa)
    parts, full = make_clean_adapted_paper(q_res)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert "module_1_Q1" in rec_meta["b_post"]
    assert "module_1_Q2" in rec_meta["b_post"]
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False


def test_pipeline_bridge_missing_pre_adaptation_qa_report_fails_closed():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    res.qa_report = None  # Simulate missing qa_report after construction
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, pre_qa_s, rec_qa_s, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert pre_qa_s is None
    assert rec_qa_s["telemetry"]["bridge_reconciliation"]["is_synthetic"] is True


def test_pipeline_bridge_population_length_mismatch_result_longer_blocks_affected_slots():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    q_res = make_clean_questions(expected_ids)
    q_ctx = make_clean_questions(expected_ids)
    # Context has only 9 questions
    q_ctx = q_ctx[:9]

    ctx.questions = q_ctx
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(q_res, qa)
    parts, full = make_clean_adapted_paper(q_res)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert "module_2_Q10" in rec_meta["b_post"]


def test_pipeline_bridge_population_length_mismatch_context_longer_blocks_affected_slots():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    q_res = make_clean_questions(expected_ids)
    q_ctx = make_clean_questions(expected_ids)
    # Result has only 9 questions
    q_res = q_res[:9]

    ctx.questions = q_ctx
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(q_res, qa)
    parts, full = make_clean_adapted_paper(q_res)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert "module_2_Q10" in rec_meta["b_post"]


def test_pipeline_bridge_pre_adaptation_unresolved_question_missing_failure_code_or_detail_fatal():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].failure_code = None  # Missing failure code
    questions[0].failure_detail = "Detail"
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids, status=STATUS_PARTIAL, unresolved_slots=["module_1_Q1"])
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("MISSING_OR_MALFORMED_FAILURE_CODE:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_pre_adaptation_clean_question_with_unexpected_failure_code_fatal():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    questions[0].is_unresolved = False
    questions[0].failure_code = "UNEXPECTED_ERROR"  # Unexpected failure code on clean question
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("UNEXPECTED_FAILURE_CODE:module_1_Q1" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_unserializable_telemetry_records_defect_without_data_loss():
    fatal_defects = []

    # An object that raises during deepcopy
    class UncopyableObject:
        def __deepcopy__(self, memo):
            raise RuntimeError("Deepcopy forbidden")

    bad_tel = {"nested_obj": UncopyableObject()}
    copied = _safe_telemetry_copy(bad_tel, fatal_defects, label="test_tel")

    assert copied["_is_corrupted"] is True
    assert "TELEMETRY_COPY_ERROR:test_tel:RuntimeError" in fatal_defects


# -----------------------------------------------------------------------------
# Tests 61 - 95: Serialization Invariants, Precedence & Early Exception Handlers
# -----------------------------------------------------------------------------


def test_pipeline_bridge_partial_paper_with_both_pre_existing_and_new_adaptation_failure_remains_partial():
    ctx = make_test_context()
    ctx.audit_report = SimpleNamespace(degraded_mode=True)
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    # Q1 pre-existing subquestion failure
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "subquestion"
    questions[0].unresolved_subparts = ["b"]
    questions[0].failure_code = "PARTIAL"
    questions[0].failure_detail = "Part b"
    ctx.questions = questions

    qa = make_clean_qa_report(
        expected_ids,
        status=STATUS_PARTIAL,
        unresolved_slots=["module_1_Q1"],
    )
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Q2 new adaptation failure
    full[1]["adaptation_failure_code"] = "FAIL"
    full[1]["adaptation_failure_detail"] = "Fail"
    full[1]["is_unresolved"] = True
    parts[0]["questions"][1]["adaptation_failure_code"] = "FAIL"
    parts[0]["questions"][1]["adaptation_failure_detail"] = "Fail"
    parts[0]["questions"][1]["is_unresolved"] = True

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert "module_1_Q1" in reconciled_qa.unresolved_slots
    assert "module_1_Q2" in reconciled_qa.unresolved_slots
    assert reconciled_qa.resolved_slots == 8
    assert reconciled_qa.status == STATUS_PARTIAL
    assert eligible is True


def test_pipeline_bridge_run_pipeline_v3_or_legacy_empty_output_preserves_meta_access(monkeypatch):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "true")
    spec = make_test_paper_spec()

    # Call with non-list plans to trigger empty failure output
    out_meta = {}
    res = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: ([], []),
        v3_kwargs={
            "paper_spec": spec,
            "artifact": None,
            "marks_split": [[10]] * 10,
            "request": {"plans": "invalid"},
        },
        legacy_kwargs={},
        out_meta=out_meta,
    )

    # Unpacks as 2-tuple
    parts, full = res
    assert parts == []
    assert full == []
    # Retains meta on wrapper
    assert res.paper_status == STATUS_FAILED
    assert res.export_eligible is False
    assert out_meta.get("export_eligible") is False


def test_pipeline_bridge_execution_failure_strictly_export_ineligible():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    # Execution failed
    res = make_clean_orchestrator_result(
        questions,
        qa,
        success=False,
        failure_code="AGENT_TIMEOUT",
        failure_detail="Writing agent timed out",
    )
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_FAILED
    assert eligible is False
    assert any("EXECUTION_FAILURE:AGENT_TIMEOUT" in r for r in reconciled_qa.blocked_reasons)


def test_pipeline_bridge_public_caller_three_way_success_semantics():
    # 1. Execution OK + Export Blocked (Tier 3)
    meta1 = _build_pipeline_meta(
        validated_exec_success=True,
        raw_success=True,
        paper_status=STATUS_BLOCKED,
        export_eligible=False,
        failure_code=None,
        failure_detail=None,
        telemetry={},
        factory_notes=[],
        qa_report=None,
        pre_adaptation_qa_report=None,
        adaptation_defects=[],
        reconciliation_meta={},
        blocked_reasons=["INTEGRITY_DEFECT:DEFECT"],
    )
    assert meta1["success"] is True
    assert meta1["paper_status"] == STATUS_BLOCKED
    assert meta1["export_eligible"] is False

    # 2. Execution Failed + Export Ineligible (Tier 2)
    meta2 = _build_pipeline_meta(
        validated_exec_success=False,
        raw_success=False,
        paper_status=STATUS_FAILED,
        export_eligible=False,
        failure_code="FAIL",
        failure_detail="Fail",
        telemetry={},
        factory_notes=[],
        qa_report=None,
        pre_adaptation_qa_report=None,
        adaptation_defects=[],
        reconciliation_meta={},
        blocked_reasons=[],
    )
    assert meta2["success"] is False
    assert meta2["paper_status"] == STATUS_FAILED
    assert meta2["export_eligible"] is False

    # 3. Execution OK + Export Eligible
    meta3 = _build_pipeline_meta(
        validated_exec_success=True,
        raw_success=True,
        paper_status=STATUS_OK,
        export_eligible=True,
        failure_code=None,
        failure_detail=None,
        telemetry={},
        factory_notes=[],
        qa_report=None,
        pre_adaptation_qa_report=None,
        adaptation_defects=[],
        reconciliation_meta={},
        blocked_reasons=[],
    )
    assert meta3["success"] is True
    assert meta3["paper_status"] == STATUS_OK
    assert meta3["export_eligible"] is True


def test_pipeline_bridge_execution_success_string_flag_fails_closed():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    res.success = "False"  # Non-boolean string flag
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    # Tier 1 applies -> BLOCKED
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("INVALID_EXECUTION_SUCCESS_FLAG:type_str" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_execution_success_none_flag_fails_closed():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    res.success = None  # NoneType flag
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    # Tier 1 applies -> BLOCKED
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("INVALID_EXECUTION_SUCCESS_FLAG:type_NoneType" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_to_legacy_shape_exception_boundary_structured_blocked():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)

    exc = RuntimeError("Adapter crashed")
    reconciled_qa, _, rec_meta, pre_qa_s, rec_qa_s, _, eligible = _handle_adaptation_execution_error(
        result=res,
        context=ctx,
        adaptation_exception=exc,
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("ADAPTATION_EXECUTION_ERROR:RuntimeError:Adapter crashed" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_malformed_pre_qa_reasons_fields_fatal_integrity_defects():
    fatal_defects = []
    # string instead of list/tuple
    res1 = _validate_and_extract_reasons("SINGLE_STRING", "blocked_reasons", fatal_defects)
    assert res1 == []
    assert any("INVALID_PRE_QA_BLOCKED_REASONS_TYPE:type_str" in d for d in fatal_defects)

    # dict instead of list/tuple
    res2 = _validate_and_extract_reasons({"bad": "dict"}, "degraded_reasons", fatal_defects)
    assert res2 == []
    assert any("INVALID_PRE_QA_DEGRADED_REASONS_TYPE:type_dict" in d for d in fatal_defects)


def test_pipeline_bridge_result_metadata_serialization_failure_fails_closed():
    # 1. Telemetry deepcopy failure
    class BadObj:
        def __deepcopy__(self, memo):
            raise ValueError("No copy")
    defects = []
    _safe_telemetry_copy({"bad": BadObj()}, fatal_defects=defects)
    assert any("TELEMETRY_COPY_ERROR" in d for d in defects)

    # 2. QAReport to_dict raising
    class BadQA:
        def to_dict(self):
            raise RuntimeError("to_dict failed")
    is_ok, snap, defect = _safe_serialize_qa_report(BadQA(), label="test")
    assert is_ok is False
    assert "QA_REPORT_SERIALIZATION_ERROR" in defect


def test_pipeline_bridge_invalid_ratio_with_unstringable_value_fails_closed():
    class Unstringable:
        def __str__(self):
            raise RuntimeError("Cannot stringify")

    fatal_defects = []
    ctx = make_test_context()
    ctx.request["min_questions_ratio"] = Unstringable()

    ratio = _resolve_min_ratio(ctx, pre_qa=None, fatal_defects=fatal_defects)
    assert ratio == 0.9  # Safe fallback value
    assert any("INVALID_MIN_QUESTIONS_RATIO_CONFIG:request:<unstringable_exception>" in d for d in fatal_defects)


def test_pipeline_bridge_reason_container_order_is_deterministic():
    defects = []
    input_reasons = ["REASON_A", "REASON_B", "REASON_C"]
    out_list = _validate_and_extract_reasons(input_reasons, "blocked_reasons", defects)
    assert out_list == input_reasons

    out_tuple = _validate_and_extract_reasons(tuple(input_reasons), "blocked_reasons", defects)
    assert out_tuple == input_reasons

    # Sets are strictly rejected
    out_set = _validate_and_extract_reasons(set(input_reasons), "blocked_reasons", defects)
    assert out_set == []
    assert any("INVALID_PRE_QA_BLOCKED_REASONS_TYPE:type_set" in d for d in defects)


def test_pipeline_bridge_expected_slot_exception_preserves_pre_qa_diagnostics():
    ctx = make_test_context()
    ctx.plans = "invalid_plans_type"
    qa = make_clean_qa_report(
        make_expected_slot_ids(),
        status=STATUS_BLOCKED,
        blocked_reasons=["PRE_EXISTING_BLOCK"],
        degraded_reasons=["PRE_EXISTING_DEGRADE"],
    )
    res = make_clean_orchestrator_result([], qa)
    parts, full = make_clean_adapted_paper([])

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert "PRE_EXISTING_BLOCK" in reconciled_qa.blocked_reasons
    assert "PRE_EXISTING_DEGRADE" in reconciled_qa.degraded_reasons
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False


def test_pipeline_bridge_all_slots_adaptation_failure_zero_resolved_blocks():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # All 10 slots fail adaptation
    for idx, qd in enumerate(full):
        qd["adaptation_failure_code"] = "FAIL"
        qd["adaptation_failure_detail"] = "Fail"
        qd["is_unresolved"] = True
        mod_idx = idx // 5
        q_sub_idx = idx % 5
        parts[mod_idx]["questions"][q_sub_idx]["adaptation_failure_code"] = "FAIL"
        parts[mod_idx]["questions"][q_sub_idx]["adaptation_failure_detail"] = "Fail"
        parts[mod_idx]["questions"][q_sub_idx]["is_unresolved"] = True

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.resolved_slots == 0
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert BLOCK_REASON_UNRESOLVED_SLOTS in reconciled_qa.blocked_reasons


def test_pipeline_bridge_qa_report_serialization_failure_synchronizes_all_representations():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    # Q1 is unresolved
    questions[0].is_unresolved = True
    questions[0].unresolved_scope = "question"
    questions[0].failure_code = "FAIL"
    questions[0].failure_detail = "Fail"
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids, status=STATUS_PARTIAL, unresolved_slots=["module_1_Q1"])
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Make QAReport.to_dict raise by monkeypatching
    original_to_dict = QAReport.to_dict
    try:
        QAReport.to_dict = lambda self: (_ for _ in ()).throw(RuntimeError("Serialization crashed"))
        reconciled_qa, _, _, _, rec_qa_s, _, eligible = _reconcile_post_adaptation(
            result=res, context=ctx, paper_parts=parts, full_paper=full
        )

        assert reconciled_qa.status == STATUS_BLOCKED
        assert rec_qa_s["status"] == STATUS_BLOCKED
        assert reconciled_qa.slot_count == rec_qa_s["slot_count"] == 10
        assert reconciled_qa.resolved_slots == rec_qa_s["resolved_slots"] == 9
        assert eligible is False
    finally:
        QAReport.to_dict = original_to_dict


def test_pipeline_bridge_qa_report_property_accessor_failure_fails_closed():
    class AccessorFailingQA:
        @property
        def to_dict(self):
            raise RuntimeError("Property accessor crashed")

    defects = []
    is_ok, snap, defect = _safe_serialize_qa_report(AccessorFailingQA(), label="test", fatal_defects=defects)
    assert is_ok is False
    assert "QA_REPORT_ACCESSOR_ERROR:test:RuntimeError" in defect
    assert defect in defects


def test_pipeline_bridge_qa_report_nested_unserializable_value_fails_closed():
    class NestedUnserializableQA:
        def to_dict(self):
            return {"key": {1, 2, 3}}  # Set cannot be serialized to JSON

    defects = []
    is_ok, snap, defect = _safe_serialize_qa_report(NestedUnserializableQA(), label="test", fatal_defects=defects)
    assert is_ok is False
    assert "QA_REPORT_JSON_SERIALIZATION_ERROR:test:TypeError" in defect


def test_pipeline_bridge_expected_slot_exception_with_execution_failure_applies_tier2_precedence():
    ctx = make_test_context()
    ctx.plans = "invalid_type"
    qa = make_clean_qa_report(make_expected_slot_ids())
    res = make_clean_orchestrator_result([], qa, success=False, failure_code="RUN_FAIL")
    parts, full = make_clean_adapted_paper([])

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    # Tier 2 applies: genuinely failed execution becomes FAILED
    assert reconciled_qa.status == STATUS_FAILED
    assert eligible is False
    assert any("EXECUTION_FAILURE:RUN_FAIL" in r for r in reconciled_qa.blocked_reasons)
    assert any("INVALID_EXPECTED_SLOT_SPECIFICATION" in r for r in reconciled_qa.blocked_reasons)


def test_pipeline_bridge_missing_pre_qa_with_execution_failure_applies_tier2_precedence():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    res = make_clean_orchestrator_result(questions, qa_report=None, success=False, failure_code="EXEC_FAIL")
    res.qa_report = None
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_FAILED
    assert eligible is False
    assert any("EXECUTION_FAILURE:EXEC_FAIL" in r for r in reconciled_qa.blocked_reasons)
    assert any("MISSING_PRE_ADAPTATION_QA_REPORT" in r for r in reconciled_qa.blocked_reasons)


def test_pipeline_bridge_adapter_exception_with_execution_failure_applies_tier2_precedence():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=False, failure_code="FAIL")

    reconciled_qa, _, _, _, _, _, eligible = _handle_adaptation_execution_error(
        result=res,
        context=ctx,
        adaptation_exception=RuntimeError("Adapter crashed"),
    )

    assert reconciled_qa.status == STATUS_FAILED
    assert eligible is False
    assert any("EXECUTION_FAILURE:FAIL" in r for r in reconciled_qa.blocked_reasons)
    assert any("ADAPTATION_EXECUTION_ERROR" in r for r in reconciled_qa.blocked_reasons)


def test_pipeline_bridge_exception_paths_with_invalid_execution_flag_applies_tier1_precedence():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    res.success = "False"  # String flag -> Tier 1 BLOCKED

    # Path A: adapter exception
    reconciled_qa, _, _, _, _, _, eligible = _handle_adaptation_execution_error(
        result=res,
        context=ctx,
        adaptation_exception=RuntimeError("Adapter error"),
    )
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False


def test_pipeline_bridge_dependency_factory_failure_uses_structured_finalization():
    ctx = make_test_context()
    factory = SimpleNamespace(
        success=False,
        failure_code="LLM_UNREACHABLE",
        failure_detail="Connection refused on port 8000",
        notes=["Tried 3 times"],
    )

    parts, full, meta = _handle_dependency_factory_failure(factory=factory, context=ctx)

    assert parts == []
    assert full == []
    assert meta["success"] is False
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False
    assert meta["failure_code"] == "LLM_UNREACHABLE"
    assert meta["qa_report"]["status"] == STATUS_FAILED


def test_pipeline_bridge_orchestrator_exception_returns_structured_failure():
    ctx = make_test_context()
    factory = SimpleNamespace(notes=[])
    run_exc = RuntimeError("Agent execution timeout")
    cleanup_errors = ["CLEANUP_RESET_ERROR:RuntimeError:Failed reset"]

    parts, full, meta = _handle_orchestrator_exception(
        orchestrator_exception=run_exc,
        context=ctx,
        factory=factory,
        cleanup_errors=cleanup_errors,
    )

    assert parts == []
    assert full == []
    assert meta["success"] is False
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False
    assert meta["error"] == "Agent execution timeout"
    assert meta["cleanup_errors"] == cleanup_errors


def test_pipeline_bridge_reconciled_serialization_fallback_preserves_actual_counts():
    qa = QAReport(
        status=STATUS_PARTIAL,
        slot_count=10,
        resolved_slots=9,
        unresolved_slots=["module_1_Q1"],
        degraded_reasons=[],
        blocked_reasons=[],
        telemetry={},
        cost_telemetry={},
        cache_telemetry={},
    )
    qa.to_dict = lambda: (_ for _ in ()).throw(RuntimeError("Serialization exploded"))

    is_ok, snap, defect = _safe_serialize_qa_report(qa, label="test")
    assert is_ok is False

    reconciled_qa, _, _, _, rec_qa_s, _, _ = _finalize_reconciliation(
        raw_success=True,
        exec_succeeded=True,
        expected_ids=make_expected_slot_ids(),
        f_post=make_expected_slot_ids()[1:],
        p_post=[],
        b_post=["module_1_Q1"],
        fatal_defects=[defect],
        adaptation_defects=[],
        pre_status=STATUS_PARTIAL,
        context=None,
        copied_telemetry={},
        copied_cost_telemetry={},
        copied_cache_telemetry={},
        copied_result_telemetry={},
        serialized_pre_qa=None,
        failure_code=None,
        validated_pre_blocked=[],
        validated_pre_degraded=[],
        resolved_population_changed=False,
        min_ratio=0.9,
        accounting_complete=True,
    )

    assert rec_qa_s["slot_count"] == 10
    assert rec_qa_s["resolved_slots"] == 9
    assert rec_qa_s["unresolved_slots"] == ["module_1_Q1"]


def test_pipeline_bridge_non_callable_reconciled_to_dict_fails_closed():
    class BadQA:
        to_dict = "not_a_callable"

    defects = []
    is_ok, snap, defect = _safe_serialize_qa_report(BadQA(), label="test", fatal_defects=defects)
    assert is_ok is False
    assert "QA_REPORT_NOT_SERIALIZABLE:test:no_callable_to_dict" in defect


def test_pipeline_bridge_nan_and_infinity_rejected_by_strict_json_serialization():
    # NaN
    is_ok, snap, defect = _safe_json_snapshot({"value": float("nan")}, label="test_nan")
    assert is_ok is False
    assert "SERIALIZATION_ERROR:test_nan:ValueError" in defect

    # +Infinity
    is_ok_inf, _, defect_inf = _safe_json_snapshot({"value": float("inf")}, label="test_inf")
    assert is_ok_inf is False
    assert "SERIALIZATION_ERROR:test_inf:ValueError" in defect_inf


def test_pipeline_bridge_result_population_present_context_absent_fatal_disagreement():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = None  # Context questions absent
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("QUESTION_POPULATION_DISAGREEMENT:context_questions_missing" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_context_population_present_result_absent_fatal_disagreement():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa)
    res.questions = None  # Result questions absent
    parts, full = make_clean_adapted_paper(questions)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any("QUESTION_POPULATION_DISAGREEMENT:result_questions_missing" in d for d in rec_meta["fatal_defects"])


def test_pipeline_bridge_zero_post_adaptation_resolved_slots_from_valid_ok_baseline_blocks():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids, status=STATUS_OK)
    res = make_clean_orchestrator_result(questions, qa)
    parts, full = make_clean_adapted_paper(questions)

    # Force all 10 slots to fail adaptation
    for idx, qd in enumerate(full):
        qd["adaptation_failure_code"] = "FAIL"
        qd["is_unresolved"] = True
        mod_idx = idx // 5
        parts[mod_idx]["questions"][idx % 5]["adaptation_failure_code"] = "FAIL"
        parts[mod_idx]["questions"][idx % 5]["is_unresolved"] = True

    reconciled_qa, _, _, _, _, _, eligible = _reconcile_post_adaptation(
        result=res, context=ctx, paper_parts=parts, full_paper=full
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert reconciled_qa.resolved_slots == 0


def test_pipeline_bridge_dependency_factory_build_exception_structured_failure():
    ctx = make_test_context()
    class CrashingFactory:
        def build(self):
            raise TypeError("Missing factory dependency")

    exc = TypeError("Missing factory dependency")
    parts, full, meta = _handle_initialization_exception(
        exc=exc,
        context=ctx,
        failure_label="DEPENDENCY_FACTORY_BUILD_ERROR",
    )

    assert parts == []
    assert full == []
    assert meta["success"] is False
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False
    assert meta["failure_code"] == "DEPENDENCY_FACTORY_BUILD_ERROR"


def test_pipeline_bridge_evaluation_reset_exceptions_handled_without_masking():
    # Case A: Pre-run reset failure
    errors_a = []
    class FailingEval:
        def reset(self):
            raise RuntimeError("Reset fail")
    factory_a = SimpleNamespace(evaluation=FailingEval())
    _safe_reset_evaluation(factory_a, errors_a, stage_label="PRE_RUN")
    assert len(errors_a) == 1
    assert "PRE_RUN_RESET_ERROR:RuntimeError:Reset fail" in errors_a[0]

    # Case B: Post-run reset in finally does not mask run() error
    errors_b = []
    _safe_reset_evaluation(factory_a, errors_b, stage_label="CLEANUP")
    assert len(errors_b) == 1
    assert "CLEANUP_RESET_ERROR" in errors_b[0]


def test_pipeline_bridge_kg_or_orchestrator_construction_exception_structured_failure():
    ctx = make_test_context()
    exc = RuntimeError("KG construction exploded")
    parts, full, meta = _handle_initialization_exception(
        exc=exc,
        context=ctx,
        failure_label="KNOWLEDGE_GRAPH_BUILD_ERROR",
    )

    assert parts == []
    assert full == []
    assert meta["success"] is False
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False
    assert meta["failure_code"] == "KNOWLEDGE_GRAPH_BUILD_ERROR"


def test_pipeline_bridge_dependency_failure_combined_with_slot_validation_failure():
    ctx = make_test_context()
    ctx.plans = "invalid_type"  # Slot validation will fail
    factory = SimpleNamespace(
        success=False,
        failure_code="BUILD_FAIL",
        failure_detail="Detail",
        notes=[],
    )

    parts, full, meta = _handle_dependency_factory_failure(factory=factory, context=ctx)

    assert meta["success"] is False
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False
    assert any("DEPENDENCY_FACTORY_FAILURE:BUILD_FAIL" in r for r in meta["blocked_reasons"])
    assert any("INVALID_EXPECTED_SLOT_SPECIFICATION" in r for r in meta["blocked_reasons"])


def test_pipeline_bridge_orchestrator_exception_combined_with_slot_validation_failure():
    ctx = make_test_context()
    ctx.plans = "invalid_type"  # Slot validation will fail
    factory = SimpleNamespace(notes=[])
    run_exc = RuntimeError("Run crash")

    parts, full, meta = _handle_orchestrator_exception(
        orchestrator_exception=run_exc,
        context=ctx,
        factory=factory,
    )

    assert meta["success"] is False
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False
    assert any("ORCHESTRATOR_EXECUTION_EXCEPTION:RuntimeError:Run crash" in r for r in meta["blocked_reasons"])
    assert any("INVALID_EXPECTED_SLOT_SPECIFICATION" in r for r in meta["blocked_reasons"])


def test_pipeline_bridge_adapter_failure_with_execution_success_reports_incomplete_accounting():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=True)

    reconciled_qa, _, rec_meta, _, _, _, eligible = _handle_adaptation_execution_error(
        result=res,
        context=ctx,
        adaptation_exception=RuntimeError("Adapter crashed"),
    )

    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    bridge_tel = reconciled_qa.telemetry.get("bridge_reconciliation", {})
    assert bridge_tel.get("accounting_incomplete") is True


def test_pipeline_bridge_public_dispatch_empty_outputs_metadata_access(monkeypatch):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "true")
    spec = make_test_paper_spec()
    out_meta = {}

    res = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: ([], []),
        v3_kwargs={
            "paper_spec": spec,
            "artifact": None,
            "marks_split": [[10]] * 10,
            "request": {"plans": "invalid"},
        },
        legacy_kwargs={},
        out_meta=out_meta,
    )

    parts, full = res
    assert parts == []
    assert full == []
    assert res.meta is not None
    assert res.paper_status == STATUS_FAILED
    assert res.export_eligible is False
    assert out_meta["export_eligible"] is False



# -----------------------------------------------------------------------------
# Tests 96 - 110
# -----------------------------------------------------------------------------


def test_pipeline_bridge_shared_meta_builder_contract_keys_identical():
    m1 = _build_pipeline_meta(
        validated_exec_success=True,
        raw_success=True,
        paper_status=STATUS_OK,
        export_eligible=True,
        failure_code=None,
        failure_detail=None,
        telemetry={},
        factory_notes=[],
        qa_report={},
        pre_adaptation_qa_report={},
        adaptation_defects=[],
        reconciliation_meta={},
        blocked_reasons=[],
    )
    m2 = _build_emergency_meta_dict(
        validated_exec_success=False,
        raw_success=False,
        paper_status=STATUS_FAILED,
        export_eligible=False,
        failure_code="ERR",
        failure_detail="detail",
        telemetry={},
        factory_notes=[],
        qa_report=None,
        pre_adaptation_qa_report=None,
        adaptation_defects=[],
        reconciliation_meta={},
        blocked_reasons=[],
    )
    _, _, m3 = _handle_context_construction_exception(exc=ValueError("spec"))
    _, _, m4 = _handle_dependency_factory_failure(
        factory=SimpleNamespace(notes=["note"]),
        context=None,
    )
    _, _, m5 = _handle_orchestrator_exception(
        orchestrator_exception=RuntimeError("timeout"),
        context=None,
        factory=None,
        cleanup_errors=[],
    )
    _, _, m6 = _handle_emergency_pipeline_failure(
        exc=RuntimeError("emergency"),
    )

    canonical_13_keys = {
        "success",
        "raw_execution_success",
        "paper_status",
        "export_eligible",
        "failure_code",
        "failure_detail",
        "telemetry",
        "factory_notes",
        "qa_report",
        "pre_adaptation_qa_report",
        "adaptation_defects",
        "reconciliation_meta",
        "blocked_reasons",
    }
    for m in [m1, m2, m3, m4, m5, m6]:
        assert canonical_13_keys.issubset(m.keys())


def test_pipeline_bridge_context_construction_exception_structured_failure():
    parts, full, meta = _handle_context_construction_exception(
        exc=ValueError("Invalid paper spec parameters"),
        failure_label="CONTEXT_CONSTRUCTION_ERROR",
    )
    assert parts == []
    assert full == []
    assert meta["success"] is False
    assert meta["raw_execution_success"] is False
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False
    assert meta["failure_code"] == "CONTEXT_CONSTRUCTION_ERROR"
    assert "Invalid paper spec parameters" in meta["failure_detail"]
    bridge_tel = meta["qa_report"]["telemetry"].get("bridge_reconciliation", {})
    assert bridge_tel.get("accounting_incomplete") is True
    assert meta["qa_report"] is not None
    assert meta["qa_report"]["status"] == STATUS_FAILED


def test_pipeline_bridge_malformed_factory_result_structured_failure():
    class MalformedFactory:
        def build(self):
            return SimpleNamespace(success="True", notes=[])

    spec = make_test_paper_spec()
    parts, full, meta = run_v3_pipeline(
        paper_spec=spec,
        artifact=None,
        marks_split=[[10]] * 10,
        dependency_factory=MalformedFactory(),
    )
    assert parts == []
    assert full == []
    assert meta["success"] is False
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False
    assert meta["failure_code"] == "MALFORMED_DEPENDENCY_FACTORY_RESULT"


def test_pipeline_bridge_evaluation_agent_attribute_access_failure():
    class RaisingEvalFactoryResult:
        success = True
        notes = []
        planning = None
        writing = None
        refinement = None
        checking = None

        @property
        def evaluation(self):
            raise RuntimeError("Access denied to evaluation")

    class RaisingEvalFactory:
        def build(self):
            return RaisingEvalFactoryResult()

    spec = make_test_paper_spec()
    parts, full, meta = run_v3_pipeline(
        paper_spec=spec,
        artifact=None,
        marks_split=[[10]] * 10,
        dependency_factory=RaisingEvalFactory(),
    )
    assert parts == []
    assert full == []
    assert meta["success"] is False
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False
    assert meta["failure_code"] in (
        "ORCHESTRATOR_CONSTRUCTION_ERROR",
        "EVALUATION_RESET_PRE_RUN_ERROR",
    )


def test_pipeline_bridge_reconciliation_exception_precedence():
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    qa = make_clean_qa_report(expected_ids)

    res_true = make_clean_orchestrator_result(questions, qa, success=True)
    _, _, meta_a = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Reconciliation crash"),
        stage="RECONCILIATION",
        result=res_true,
    )
    assert meta_a["success"] is True
    assert meta_a["raw_execution_success"] is True
    assert meta_a["paper_status"] == STATUS_BLOCKED
    assert meta_a["export_eligible"] is False

    res_false = make_clean_orchestrator_result(questions, qa, success=False)
    _, _, meta_b = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Reconciliation crash"),
        stage="RECONCILIATION",
        result=res_false,
    )
    assert meta_b["success"] is False
    assert meta_b["raw_execution_success"] is False
    assert meta_b["paper_status"] == STATUS_FAILED
    assert meta_b["export_eligible"] is False


def test_pipeline_bridge_metadata_builder_exception_precedence():
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    qa = make_clean_qa_report(expected_ids)

    res_true = make_clean_orchestrator_result(questions, qa, success=True)
    _, _, meta_a = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Metadata builder crash"),
        stage="METADATA_BUILD",
        result=res_true,
    )
    assert meta_a["success"] is True
    assert meta_a["paper_status"] == STATUS_BLOCKED
    assert meta_a["export_eligible"] is False

    res_false = make_clean_orchestrator_result(questions, qa, success=False)
    _, _, meta_b = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Metadata builder crash"),
        stage="METADATA_BUILD",
        result=res_false,
    )
    assert meta_b["success"] is False
    assert meta_b["paper_status"] == STATUS_FAILED
    assert meta_b["export_eligible"] is False

    _, _, meta_c = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Metadata builder crash"),
        stage="METADATA_BUILD",
        result=None,
    )
    assert meta_c["success"] is False
    assert meta_c["raw_execution_success"] is None
    assert meta_c["paper_status"] == STATUS_FAILED
    assert meta_c["export_eligible"] is False


def test_pipeline_bridge_serialization_fallback_preserves_all_nine_fields(monkeypatch):
    expected_ids = make_expected_slot_ids()
    ctx = make_test_context()

    def raising_to_dict(self):
        raise RuntimeError("to_dict failed")

    monkeypatch.setattr(QAReport, "to_dict", raising_to_dict)

    fatal_defects = []
    (
        reconciled_qa,
        adaptation_defects,
        rec_meta,
        ser_pre,
        serialized_reconciled_qa,
        tel,
        eligible,
    ) = _finalize_reconciliation(
        raw_success=True,
        exec_succeeded=True,
        expected_ids=expected_ids,
        f_post=expected_ids,
        p_post=[],
        b_post=[],
        fatal_defects=fatal_defects,
        adaptation_defects=[],
        pre_status=STATUS_OK,
        context=ctx,
        copied_telemetry={"custom_t": 123},
        copied_cost_telemetry={"total_cost": 0.42},
        copied_cache_telemetry={"cache_hits": 5},
        copied_result_telemetry={},
        serialized_pre_qa={},
        failure_code=None,
        validated_pre_blocked=[],
        validated_pre_degraded=[],
        resolved_population_changed=False,
        min_ratio=0.9,
        accounting_complete=True,
    )

    assert any("QA_REPORT_SERIALIZATION_ERROR" in d for d in fatal_defects)

    canonical_fields = {
        "status",
        "slot_count",
        "resolved_slots",
        "unresolved_slots",
        "degraded_reasons",
        "blocked_reasons",
        "telemetry",
        "cost_telemetry",
        "cache_telemetry",
    }
    assert canonical_fields.issubset(serialized_reconciled_qa.keys())
    assert serialized_reconciled_qa["status"] == STATUS_BLOCKED
    assert serialized_reconciled_qa["slot_count"] == 10
    assert serialized_reconciled_qa["resolved_slots"] == 10
    assert serialized_reconciled_qa["unresolved_slots"] == []
    assert serialized_reconciled_qa["cost_telemetry"] == {"total_cost": 0.42}
    assert serialized_reconciled_qa["cache_telemetry"] == {"cache_hits": 5}
    assert serialized_reconciled_qa["_is_synthetic_fallback"] is True


def test_pipeline_bridge_none_reason_fields_rejected_by_strict_contract():
    fatal_defects = []
    br = _validate_and_extract_reasons(None, "blocked_reasons", fatal_defects)
    dr = _validate_and_extract_reasons(None, "degraded_reasons", fatal_defects)
    assert br == []
    assert dr == []
    assert any(
        "INVALID_PRE_QA_BLOCKED_REASONS_TYPE:type_NoneType" in d
        for d in fatal_defects
    )
    assert any(
        "INVALID_PRE_QA_DEGRADED_REASONS_TYPE:type_NoneType" in d
        for d in fatal_defects
    )


def test_pipeline_bridge_public_caller_retains_metadata_on_empty_outputs(
    monkeypatch,
):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "true")
    spec = make_test_paper_spec()
    parts_a, full_a, meta_a = run_v3_pipeline(
        paper_spec=spec,
        artifact=None,
        marks_split=[[10]] * 10,
        request={"plans": "invalid_for_v3"},
    )
    assert parts_a == [] and full_a == []
    assert meta_a["paper_status"] == STATUS_FAILED
    assert meta_a["export_eligible"] is False

    dummy_meta = {
        "paper_status": STATUS_BLOCKED,
        "export_eligible": False,
        "success": True,
        "qa_report": {"status": "BLOCKED"},
    }
    bridge_res = PipelineBridgeResult([], [], dummy_meta)
    p_b, f_b = bridge_res
    assert p_b == [] and f_b == []
    assert bridge_res.meta == dummy_meta
    assert bridge_res.paper_status == STATUS_BLOCKED
    assert bridge_res.export_eligible is False
    assert bridge_res.success is True
    assert bridge_res.qa_report == {"status": "BLOCKED"}

    res_c = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: ([], []),
        v3_kwargs={
            "paper_spec": spec,
            "artifact": None,
            "marks_split": [[10]] * 10,
            "request": {"plans": "invalid"},
        },
        legacy_kwargs={},
        return_meta=True,
    )
    assert len(res_c) == 3
    assert res_c[0] == [] and res_c[1] == []
    assert res_c[2]["export_eligible"] is False

    out_dict = {}
    run_pipeline_v3_or_legacy(
        legacy_runner=lambda: ([], []),
        v3_kwargs={
            "paper_spec": spec,
            "artifact": None,
            "marks_split": [[10]] * 10,
            "request": {"plans": "invalid"},
        },
        legacy_kwargs={},
        out_meta=out_dict,
    )
    assert out_dict["export_eligible"] is False
    assert out_dict["paper_status"] == STATUS_FAILED


def test_pipeline_bridge_emergency_failure_with_successful_execution_applies_tier3_precedence():
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=True)

    _, _, meta = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Post-run unexpected emergency crash"),
        stage="POST_RUN",
        result=res,
    )
    assert meta["success"] is True
    assert meta["raw_execution_success"] is True
    assert meta["paper_status"] == STATUS_BLOCKED
    assert meta["export_eligible"] is False
    assert any("EMERGENCY_POST_RUN" in br for br in meta["blocked_reasons"])


def test_pipeline_bridge_emergency_failure_with_failed_execution_applies_tier2_precedence():
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=False)

    _, _, meta = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Post-run unexpected emergency crash"),
        stage="POST_RUN",
        result=res,
    )
    assert meta["success"] is False
    assert meta["raw_execution_success"] is False
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False
    assert any("EMERGENCY_POST_RUN" in br for br in meta["blocked_reasons"])


def test_pipeline_bridge_empty_paper_parts_with_populated_full_paper_blocks():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=True)
    full_paper = [
        {"slot_id": s, "text": f"Q text for {s}", "module_id": "module_1"}
        for s in expected_ids
    ]

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res,
        context=ctx,
        paper_parts=[],
        full_paper=full_paper,
    )
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any(
        "PAPER_PARTS_EMPTY_WITH_POPULATED_FULL_PAPER" in d
        for d in rec_meta["fatal_defects"]
    )


def test_pipeline_bridge_paper_parts_and_full_paper_slot_mismatch_blocks():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=True)

    parts = [
        {
            "module_id": "module_1",
            "questions": [
                {
                    "slot_id": "module_1_Q1",
                    "text": "Q1",
                    "module_id": "module_1",
                },
                {
                    "slot_id": "module_1_Q3",
                    "text": "Q3",
                    "module_id": "module_1",
                },
            ],
        }
    ]
    full = [
        {"slot_id": "module_1_Q1", "text": "Q1", "module_id": "module_1"},
        {"slot_id": "module_1_Q2", "text": "Q2", "module_id": "module_1"},
    ]

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res,
        context=ctx,
        paper_parts=parts,
        full_paper=full,
    )
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any(
        "SLOT_MISMATCH_PARTS_EXTRA:module_1_Q3" in d
        for d in rec_meta["fatal_defects"]
    )
    assert any(
        "SLOT_MISMATCH_PARTS_MISSING:module_1_Q2" in d
        for d in rec_meta["fatal_defects"]
    )


def test_pipeline_bridge_malformed_paper_parts_record_blocks():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=True)
    full = [
        {"slot_id": s, "text": "Q", "module_id": "module_1"}
        for s in expected_ids
    ]

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res,
        context=ctx,
        paper_parts=["corrupted_part_record"],
        full_paper=full,
    )
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False
    assert any(
        "MALFORMED_PAPER_PART_RECORD" in d for d in rec_meta["fatal_defects"]
    )


def test_pipeline_bridge_legacy_runner_clean_empty_output_success_true_export_false(
    monkeypatch,
):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "false")
    res = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: ([], []),
        v3_kwargs={},
        legacy_kwargs={},
    )
    parts, full = res
    assert parts == []
    assert full == []
    assert res.success is True
    assert res.paper_status == "LEGACY_EMPTY"
    assert res.export_eligible is False
    assert res.meta["raw_execution_success"] is True



# -----------------------------------------------------------------------------
# Tests 111 - 129
# -----------------------------------------------------------------------------


def test_pipeline_bridge_legacy_runner_exception_success_false_export_false(
    monkeypatch,
):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "false")

    def crashing_runner():
        raise RuntimeError("Legacy engine crash")

    res = run_pipeline_v3_or_legacy(
        legacy_runner=crashing_runner,
        v3_kwargs={},
        legacy_kwargs={},
    )
    parts, full = res
    assert parts == []
    assert full == []
    assert res.success is False
    assert res.paper_status == "LEGACY_FAILED"
    assert res.export_eligible is False
    assert any(
        "LEGACY_RUNNER_EXCEPTION" in br for br in res.meta["blocked_reasons"]
    )


def test_pipeline_bridge_emergency_failure_preserves_pre_qa_and_cleanup_diagnostics():
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=True)

    _, _, meta = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Emergency pipeline failure"),
        stage="POST_RUN_TEST",
        result=res,
        cleanup_errors=["Evaluation cleanup failed: Timeout"],
    )

    assert meta["pre_adaptation_qa_report"] is not None
    assert "cleanup_errors" in meta
    assert "Evaluation cleanup failed: Timeout" in meta["cleanup_errors"]
    assert any(
        "Evaluation cleanup failed: Timeout" in br
        for br in meta["blocked_reasons"]
    )
    assert any(
        "EMERGENCY_POST_RUN_TEST" in br for br in meta["blocked_reasons"]
    )


def test_pipeline_bridge_metadata_builder_persistent_failure_emergency_recovery(
    monkeypatch,
):
    def crashing_meta_builder(**kwargs):
        raise RuntimeError("Persistent meta builder crash")

    monkeypatch.setattr(
        "core.generation.agents.pipeline_bridge._build_emergency_meta_dict",
        crashing_meta_builder,
    )

    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    qa = make_clean_qa_report(expected_ids)

    # (A) Execution succeeded -> Tier 3 BLOCKED
    res_true = make_clean_orchestrator_result(questions, qa, success=True)
    _, _, meta_a = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Reconciliation crash"),
        stage="TEST_PERSISTENT",
        result=res_true,
    )
    assert meta_a["success"] is True
    assert meta_a["raw_execution_success"] is True
    assert meta_a["paper_status"] == STATUS_BLOCKED
    assert meta_a["export_eligible"] is False
    assert (
        meta_a["failure_code"] == "TERMINAL_EMERGENCY_CONTAINMENT_CRASH"
    )

    # (B) Execution failed -> Tier 2 FAILED
    res_false = make_clean_orchestrator_result(questions, qa, success=False)
    _, _, meta_b = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Reconciliation crash"),
        stage="TEST_PERSISTENT",
        result=res_false,
    )
    assert meta_b["success"] is False
    assert meta_b["raw_execution_success"] is False
    assert meta_b["paper_status"] == STATUS_FAILED
    assert meta_b["export_eligible"] is False


def test_pipeline_bridge_emergency_fallback_preserves_all_telemetry_channels():
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    qa = make_clean_qa_report(expected_ids)
    qa.telemetry = {"custom_t": 1}
    qa.cost_telemetry = {"cost": 0.5}
    qa.cache_telemetry = {"hits": 10}

    res = make_clean_orchestrator_result(questions, qa, success=True)
    res.telemetry = {"nan_val": float("nan")}

    _, _, meta = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Telemetry preservation crash"),
        stage="STAGE_TEL",
        result=res,
    )

    assert meta["qa_report"] is not None
    assert meta["qa_report"]["cost_telemetry"] == {"cost": 0.5}
    assert meta["qa_report"]["cache_telemetry"] == {"hits": 10}
    assert "bridge_reconciliation" in meta["qa_report"]["telemetry"]


def test_pipeline_bridge_emergency_telemetry_contains_bridge_reconciliation_structure():
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=True)

    _, _, meta = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Emergency stage test"),
        stage="ORCHESTRATOR_RUN",
        result=res,
        expected_ids=expected_ids,
    )

    bridge_tel = meta["telemetry"]["bridge_reconciliation"]
    assert bridge_tel["expected_slots"] == 10
    assert bridge_tel["resolved_slots"] == 0
    assert bridge_tel["unresolved_slots"] == 10
    assert bridge_tel["accounting_incomplete"] is True
    assert bridge_tel["emergency_failure"] is True
    assert bridge_tel["emergency_stage"] == "ORCHESTRATOR_RUN"


def test_pipeline_bridge_duplicate_slot_id_in_paper_parts_accounted():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=True)

    parts, full = make_clean_adapted_paper(questions)
    parts[0]["questions"].append(dict(parts[0]["questions"][0]))

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res,
        context=ctx,
        paper_parts=parts,
        full_paper=full,
    )

    assert any(
        "DUPLICATE_PAPER_PARTS_SLOT_ID:module_1_Q1" in d
        for d in rec_meta["fatal_defects"]
    )
    assert "module_1_Q1" in rec_meta["b_post"]
    assert "module_1_Q1" not in rec_meta["f_post"]
    assert len(rec_meta["f_post"]) + len(rec_meta["p_post"]) + len(
        rec_meta["b_post"]
    ) == len(expected_ids)
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False


def test_pipeline_bridge_order_only_disagreement_between_parts_and_full_paper():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=True)

    parts, full = make_clean_adapted_paper(questions)
    parts[0]["questions"][0], parts[0]["questions"][1] = (
        parts[0]["questions"][1],
        parts[0]["questions"][0],
    )

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res,
        context=ctx,
        paper_parts=parts,
        full_paper=full,
    )

    assert any(
        "PAPER_PARTS_ORDER_MISMATCH" in d for d in rec_meta["fatal_defects"]
    )
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False


def test_pipeline_bridge_duplicate_module_id_in_paper_parts_accounted():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=True)

    parts, full = make_clean_adapted_paper(questions)
    parts.append(copy.deepcopy(parts[0]))

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res,
        context=ctx,
        paper_parts=parts,
        full_paper=full,
    )

    assert any(
        "DUPLICATE_PAPER_PART_MODULE_ID:module_1" in d
        for d in rec_meta["fatal_defects"]
    )
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False


def test_pipeline_bridge_question_under_wrong_module_rejected():
    ctx = make_test_context()
    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    ctx.questions = questions
    qa = make_clean_qa_report(expected_ids)
    res = make_clean_orchestrator_result(questions, qa, success=True)

    parts, full = make_clean_adapted_paper(questions)
    parts[0]["questions"][0]["module_id"] = "module_2"

    reconciled_qa, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
        result=res,
        context=ctx,
        paper_parts=parts,
        full_paper=full,
    )

    assert any(
        "QUESTION_MODULE_ATTRIBUTION_MISMATCH" in d
        for d in rec_meta["fatal_defects"]
    )
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False


def test_pipeline_bridge_legacy_malformed_non_empty_output_rejected(
    monkeypatch,
):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "false")

    # (A) question_text="" but text="Valid" -> succeeds as LEGACY
    valid_parts = [
        {
            "module_id": "module_1",
            "questions": [
                {
                    "question_text": "",
                    "text": "Valid question text",
                    "slot_id": "module_1_Q1",
                    "module_id": "module_1",
                }
            ],
        }
    ]
    valid_full = [
        {
            "question_text": "",
            "text": "Valid question text",
            "slot_id": "module_1_Q1",
            "module_id": "module_1",
        }
    ]
    res_a = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: (valid_parts, valid_full),
        v3_kwargs={},
        legacy_kwargs={},
    )
    assert res_a.paper_status == "LEGACY"
    assert res_a.export_eligible is True

    # (B) question_text="   " and text="" -> rejected as LEGACY_MALFORMED
    bad_parts_b = [
        {
            "module_id": "module_1",
            "questions": [
                {
                    "question_text": "   ",
                    "text": "",
                    "slot_id": "module_1_Q1",
                    "module_id": "module_1",
                }
            ],
        }
    ]
    bad_full_b = [
        {
            "question_text": "   ",
            "text": "",
            "slot_id": "module_1_Q1",
            "module_id": "module_1",
        }
    ]
    res_b = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: (bad_parts_b, bad_full_b),
        v3_kwargs={},
        legacy_kwargs={},
    )
    assert res_b.paper_status == "LEGACY_MALFORMED"
    assert res_b.export_eligible is False

    # (C) Non-dict question in full_paper -> rejected
    bad_full_c = ["not_a_dict"]
    res_c = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: (valid_parts, bad_full_c),
        v3_kwargs={},
        legacy_kwargs={},
    )
    assert res_c.paper_status == "LEGACY_MALFORMED"
    assert res_c.export_eligible is False


def test_pipeline_bridge_four_tier_precedence_across_exception_points(
    monkeypatch,
):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "true")

    # (1) Context construction failure -> FAILED
    _, _, m1 = _handle_context_construction_exception(exc=ValueError("spec"))
    assert m1["paper_status"] == STATUS_FAILED
    assert m1["success"] is False

    # (2) Dependency factory failure -> FAILED
    _, _, m2 = _handle_dependency_factory_failure(
        factory=SimpleNamespace(notes=[]), context=None
    )
    assert m2["paper_status"] == STATUS_FAILED
    assert m2["success"] is False

    # (3) Orchestrator execution failure (result.success=False) -> FAILED
    res_f = make_clean_orchestrator_result(
        [], make_clean_qa_report([]), success=False
    )
    _, _, m3 = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Downstream"), stage="STAGE_3", result=res_f
    )
    assert m3["paper_status"] == STATUS_FAILED
    assert m3["success"] is False

    # (4) Invalid execution flag (result.success="True") -> BLOCKED
    res_str = make_clean_orchestrator_result(
        [], make_clean_qa_report([]), success=True
    )
    res_str.success = "True"
    _, _, m4 = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Downstream"), stage="STAGE_4", result=res_str
    )
    assert m4["paper_status"] == STATUS_BLOCKED
    assert m4["success"] is False

    # (5) Successful execution (result.success=True) with downstream crash -> BLOCKED
    res_ok = make_clean_orchestrator_result(
        [], make_clean_qa_report([]), success=True
    )
    _, _, m5 = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Reconciliation crash"), stage="STAGE_5", result=res_ok
    )
    assert m5["paper_status"] == STATUS_BLOCKED
    assert m5["success"] is True

    # (6) Successful execution with terminal fallback crash -> BLOCKED
    def crashing_emergency_builder(**kwargs):
        raise RuntimeError("Crash builder")

    monkeypatch.setattr(
        "core.generation.agents.pipeline_bridge._build_emergency_meta_dict",
        crashing_emergency_builder,
    )
    _, _, m6 = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Crash"), stage="STAGE_6", result=res_ok
    )
    assert m6["paper_status"] == STATUS_BLOCKED
    assert m6["success"] is True
    assert m6["export_eligible"] is False


def test_pipeline_bridge_terminal_fallback_preserves_execution_status(
    monkeypatch,
):
    def crashing_meta_builder(**kwargs):
        raise RuntimeError("Crash")

    monkeypatch.setattr(
        "core.generation.agents.pipeline_bridge._build_emergency_meta_dict",
        crashing_meta_builder,
    )

    expected_ids = make_expected_slot_ids()
    questions = make_clean_questions(expected_ids)
    qa = make_clean_qa_report(expected_ids)

    # (A) result.success = True
    res_true = make_clean_orchestrator_result(questions, qa, success=True)
    _, _, meta_a = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Boom"), stage="TEST_TERMINAL", result=res_true
    )
    assert meta_a["success"] is True
    assert meta_a["raw_execution_success"] is True
    assert meta_a["paper_status"] == STATUS_BLOCKED
    assert meta_a["export_eligible"] is False

    # (B) result.success = False
    res_false = make_clean_orchestrator_result(questions, qa, success=False)
    _, _, meta_b = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Boom"), stage="TEST_TERMINAL", result=res_false
    )
    assert meta_b["success"] is False
    assert meta_b["raw_execution_success"] is False
    assert meta_b["paper_status"] == STATUS_FAILED
    assert meta_b["export_eligible"] is False


def test_pipeline_bridge_safe_json_snapshot_applied_uniformly_surfacing_defects():
    circular = {}
    circular["self"] = circular

    fatal_defects = []
    is_ok, snapshot, defect = _safe_json_snapshot(
        circular, label="test_circular", fatal_defects=fatal_defects
    )
    assert not is_ok
    assert snapshot["_is_corrupted"] is True
    assert any("SERIALIZATION_ERROR:test_circular" in d for d in fatal_defects)

    fatal_defects_nan = []
    is_ok_nan, snapshot_nan, defect_nan = _safe_json_snapshot(
        {"val": float("nan")},
        label="test_nan",
        fatal_defects=fatal_defects_nan,
    )
    assert not is_ok_nan
    assert snapshot_nan["_is_corrupted"] is True
    assert any("SERIALIZATION_ERROR:test_nan" in d for d in fatal_defects_nan)


def test_pipeline_bridge_partition_validation_step_enforces_disjoint_coverage():
    expected_ids = ["module_1_Q1", "module_1_Q2", "module_1_Q3"]
    ctx = make_test_context()

    fatal_defects = []
    (
        reconciled_qa,
        _,
        rec_meta,
        _,
        _,
        _,
        eligible,
    ) = _finalize_reconciliation(
        raw_success=True,
        exec_succeeded=True,
        expected_ids=expected_ids,
        f_post=["module_1_Q1", "module_1_Q2"],
        p_post=["module_1_Q2"],
        b_post=["module_99_Q1"],
        fatal_defects=fatal_defects,
        adaptation_defects=[],
        pre_status=STATUS_OK,
        context=ctx,
        copied_telemetry={},
        copied_cost_telemetry={},
        copied_cache_telemetry={},
        copied_result_telemetry={},
        serialized_pre_qa={},
        failure_code=None,
        validated_pre_blocked=[],
        validated_pre_degraded=[],
        resolved_population_changed=True,
        min_ratio=0.9,
        accounting_complete=True,
    )

    assert any(
        "PARTITION_OVERLAP_F_AND_P:module_1_Q2" in d for d in fatal_defects
    )
    assert any(
        "UNEXPECTED_SLOT_IN_PARTITION:module_99_Q1" in d for d in fatal_defects
    )
    assert any(
        "UNACCOUNTED_EXPECTED_SLOT:module_1_Q3" in d for d in fatal_defects
    )
    assert "module_99_Q1" not in rec_meta["f_post"]
    assert "module_99_Q1" not in rec_meta["p_post"]
    assert "module_99_Q1" not in rec_meta["b_post"]
    all_sanitized = (
        set(rec_meta["f_post"])
        | set(rec_meta["p_post"])
        | set(rec_meta["b_post"])
    )
    assert all_sanitized == set(expected_ids)
    assert reconciled_qa.status == STATUS_BLOCKED
    assert eligible is False


def test_pipeline_bridge_legacy_asymmetric_emptiness_triggers_malformed(
    monkeypatch,
):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "false")

    # (A) paper_parts empty, full_paper populated
    res_a = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: ([], [{"text": "Q1"}]),
        v3_kwargs={},
        legacy_kwargs={},
    )
    assert res_a.paper_status == "LEGACY_MALFORMED"
    assert res_a.export_eligible is False
    assert res_a.success is True

    # (B) paper_parts populated, full_paper empty
    res_b = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: (
            [{"module_id": "m1", "questions": [{"text": "Q1"}]}],
            [],
        ),
        v3_kwargs={},
        legacy_kwargs={},
    )
    assert res_b.paper_status == "LEGACY_MALFORMED"
    assert res_b.export_eligible is False
    assert res_b.success is True


def test_pipeline_bridge_legacy_tuple_arity_strictly_enforced(monkeypatch):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "false")

    res = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: ([], [], {}),
        v3_kwargs={},
        legacy_kwargs={},
    )
    assert res.paper_status == "LEGACY_MALFORMED"
    assert res.export_eligible is False
    assert res.success is True


def test_pipeline_bridge_legacy_bilateral_question_population_equivalence(
    monkeypatch,
):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "false")

    # (A) Question text mismatch
    parts = [
        {
            "module_id": "module_1",
            "questions": [
                {
                    "text": "Text A",
                    "slot_id": "module_1_Q1",
                    "module_id": "module_1",
                }
            ],
        }
    ]
    full = [
        {
            "text": "Text B",
            "slot_id": "module_1_Q1",
            "module_id": "module_1",
        }
    ]
    res_a = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: (parts, full),
        v3_kwargs={},
        legacy_kwargs={},
    )
    assert res_a.paper_status == "LEGACY_MALFORMED"
    assert res_a.export_eligible is False

    # (B) Slot ID mismatch
    parts_b = [
        {
            "module_id": "module_1",
            "questions": [
                {
                    "text": "Same text",
                    "slot_id": "module_1_Q1",
                    "module_id": "module_1",
                }
            ],
        }
    ]
    full_b = [
        {
            "text": "Same text",
            "slot_id": "module_1_Q2",
            "module_id": "module_1",
        }
    ]
    res_b = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: (parts_b, full_b),
        v3_kwargs={},
        legacy_kwargs={},
    )
    assert res_b.paper_status == "LEGACY_MALFORMED"
    assert res_b.export_eligible is False


def test_pipeline_bridge_dispatcher_exception_preserves_v3_technical_success(
    monkeypatch,
):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "true")

    def mock_run_v3(**kwargs):
        meta = {
            "success": True,
            "raw_execution_success": True,
            "paper_status": STATUS_OK,
            "export_eligible": True,
            "qa_report": {"status": "OK"},
        }
        return [], [], meta

    monkeypatch.setattr(
        "core.generation.agents.pipeline_bridge.run_v3_pipeline", mock_run_v3
    )

    import builtins

    def crashing_print(*args, **kwargs):
        raise RuntimeError("Crash in dispatcher logging")

    monkeypatch.setattr(builtins, "print", crashing_print)

    res = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: ([], []),
        v3_kwargs={},
        legacy_kwargs={},
    )
    assert res.success is True
    assert res.paper_status == STATUS_BLOCKED
    assert res.export_eligible is False


def test_pipeline_bridge_strict_pre_status_validation():
    expected_ids = make_expected_slot_ids()
    ctx = make_test_context()

    for invalid_status in [None, "UNKNOWN", "SUCCESS", 123]:
        fatal_defects = []
        (
            reconciled_qa,
            _,
            rec_meta,
            _,
            _,
            _,
            eligible,
        ) = _finalize_reconciliation(
            raw_success=True,
            exec_succeeded=True,
            expected_ids=expected_ids,
            f_post=expected_ids,
            p_post=[],
            b_post=[],
            fatal_defects=fatal_defects,
            adaptation_defects=[],
            pre_status=invalid_status,
            context=ctx,
            copied_telemetry={},
            copied_cost_telemetry={},
            copied_cache_telemetry={},
            copied_result_telemetry={},
            serialized_pre_qa={},
            failure_code=None,
            validated_pre_blocked=[],
            validated_pre_degraded=[],
            resolved_population_changed=False,
            min_ratio=0.9,
            accounting_complete=True,
        )
        assert any("INVALID_PRE_QA_STATUS" in d for d in fatal_defects)
        assert reconciled_qa.status == STATUS_BLOCKED
        assert eligible is False


# =============================================================================
# Targeted Defect Regression Tests (Findings 1 through 9)
# =============================================================================


def test_finding_1_legacy_validation_fail_closed_on_missing_or_non_string_ids(
    monkeypatch,
):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "false")

    # 1A: Missing slot_id in paper_parts
    parts_no_slot = [
        {
            "module_id": "module_1",
            "questions": [{"text": "Q1", "module_id": "module_1"}],
        }
    ]
    full_no_slot = [
        {"text": "Q1", "slot_id": "m1_q1", "module_id": "module_1"}
    ]
    res_1a = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: (parts_no_slot, full_no_slot),
        v3_kwargs={},
        legacy_kwargs={},
        return_meta=True,
    )
    assert res_1a[2]["paper_status"] == "LEGACY_MALFORMED"
    assert res_1a[2]["export_eligible"] is False

    # 1B: Empty string slot_id in full_paper
    parts_valid = [
        {
            "module_id": "module_1",
            "questions": [
                {"text": "Q1", "slot_id": "m1_q1", "module_id": "module_1"}
            ],
        }
    ]
    full_empty_slot = [
        {"text": "Q1", "slot_id": "", "module_id": "module_1"}
    ]
    res_1b = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: (parts_valid, full_empty_slot),
        v3_kwargs={},
        legacy_kwargs={},
        return_meta=True,
    )
    assert res_1b[2]["paper_status"] == "LEGACY_MALFORMED"
    assert res_1b[2]["export_eligible"] is False

    # 1C: Non-string slot_id (integer)
    parts_int_slot = [
        {
            "module_id": "module_1",
            "questions": [
                {"text": "Q1", "slot_id": 123, "module_id": "module_1"}
            ],
        }
    ]
    full_int_slot = [
        {"text": "Q1", "slot_id": 123, "module_id": "module_1"}
    ]
    res_1c = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: (parts_int_slot, full_int_slot),
        v3_kwargs={},
        legacy_kwargs={},
        return_meta=True,
    )
    assert res_1c[2]["paper_status"] == "LEGACY_MALFORMED"
    assert res_1c[2]["export_eligible"] is False

    # 1D: Duplicate slot_id in legacy output
    parts_dup = [
        {
            "module_id": "module_1",
            "questions": [
                {"text": "Q1", "slot_id": "m1_q1", "module_id": "module_1"},
                {"text": "Q2", "slot_id": "m1_q1", "module_id": "module_1"},
            ],
        }
    ]
    full_dup = [
        {"text": "Q1", "slot_id": "m1_q1", "module_id": "module_1"},
        {"text": "Q2", "slot_id": "m1_q1", "module_id": "module_1"},
    ]
    res_1d = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: (parts_dup, full_dup),
        v3_kwargs={},
        legacy_kwargs={},
        return_meta=True,
    )
    assert res_1d[2]["paper_status"] == "LEGACY_MALFORMED"
    assert res_1d[2]["export_eligible"] is False

    # 1E: Missing question-level module_id
    parts_no_mod = [
        {
            "module_id": "module_1",
            "questions": [{"text": "Q1", "slot_id": "m1_q1"}],
        }
    ]
    full_no_mod = [{"text": "Q1", "slot_id": "m1_q1"}]
    res_1e = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: (parts_no_mod, full_no_mod),
        v3_kwargs={},
        legacy_kwargs={},
        return_meta=True,
    )
    assert res_1e[2]["paper_status"] == "LEGACY_MALFORMED"
    assert res_1e[2]["export_eligible"] is False


def test_finding_2_dispatcher_exception_preserves_consistent_blocked_state(
    monkeypatch,
):
    monkeypatch.setenv("AION_ENABLE_V3_AGENTS", "true")

    def mock_run_v3(**kwargs):
        return (
            [{"module_id": "module_1", "questions": []}],
            [],
            {
                "success": True,
                "raw_execution_success": True,
                "paper_status": STATUS_OK,
                "export_eligible": True,
                "pre_adaptation_qa_report": {
                    "status": "PASS",
                    "blocked_reasons": [],
                },
            },
        )

    monkeypatch.setattr(
        "core.generation.agents.pipeline_bridge.run_v3_pipeline", mock_run_v3
    )

    call_count = [0]
    orig_wrap = _safe_wrap_bridge_result

    def failing_wrap(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("POST_EXECUTION_DISPATCH_CRASH")
        return orig_wrap(*args, **kwargs)

    monkeypatch.setattr(
        "core.generation.agents.pipeline_bridge._safe_wrap_bridge_result",
        failing_wrap,
    )

    res = run_pipeline_v3_or_legacy(
        legacy_runner=lambda: ([], []),
        v3_kwargs={},
        legacy_kwargs={},
        return_meta=False,
    )

    assert res.meta["success"] is True
    assert res.meta["raw_execution_success"] is True
    assert res.meta["paper_status"] == STATUS_BLOCKED
    assert res.meta["export_eligible"] is False
    assert res.meta["qa_report"]["status"] == STATUS_BLOCKED
    assert not any(
        "EXECUTION_FAILURE" in r
        for r in res.meta["qa_report"]["blocked_reasons"]
    )


def test_finding_3_accounting_completeness_derived_dynamically():
    mock_pre_qa = MagicMock(spec=QAReport)
    mock_pre_qa.status = "PASS"
    mock_pre_qa.slot_count = 0
    mock_pre_qa.resolved_slots = 0
    mock_pre_qa.unresolved_slots = []
    mock_pre_qa.to_dict.return_value = {
        "status": "PASS",
        "slot_count": 0,
        "resolved_slots": 0,
        "unresolved_slots": [],
        "degraded_reasons": [],
        "blocked_reasons": [],
        "telemetry": {},
        "cost_telemetry": {},
        "cache_telemetry": {},
    }
    mock_pre_qa.telemetry = {}
    mock_pre_qa.cost_telemetry = {}
    mock_pre_qa.cache_telemetry = {}
    mock_pre_qa.blocked_reasons = []
    mock_pre_qa.degraded_reasons = []

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.questions = []
    mock_result.failure_code = None
    mock_result.telemetry = {}
    mock_result.qa_report = mock_pre_qa

    mock_context = MagicMock()

    # Empty expected slots -> accounting_incomplete MUST be True
    with unittest.mock.patch(
        "core.generation.agents.checking_agent.CheckingAgent._get_validated_expected_slots",
        return_value=[],
    ):
        (
            _,
            _,
            rec_meta_empty,
            _,
            ser_rec_empty,
            _,
            _,
        ) = _reconcile_post_adaptation(
            result=mock_result,
            context=mock_context,
            paper_parts=[],
            full_paper=[],
            cleanup_errors=[],
        )
        assert (
            ser_rec_empty["telemetry"]["bridge_reconciliation"][
                "accounting_incomplete"
            ]
            is True
        )
        assert (
            ser_rec_empty["telemetry"]["bridge_reconciliation"][
                "accounting_completeness"
            ]
            == "incomplete"
        )


def test_finding_4_duplicate_module_id_registered_when_malformed_part_appears_first():
    mock_pre_qa = MagicMock(spec=QAReport)
    mock_pre_qa.status = "PASS"
    mock_pre_qa.to_dict.return_value = {"status": "PASS", "blocked_reasons": []}
    mock_pre_qa.telemetry = {}
    mock_pre_qa.cost_telemetry = {}
    mock_pre_qa.cache_telemetry = {}
    mock_pre_qa.blocked_reasons = []
    mock_pre_qa.degraded_reasons = []

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.questions = []
    mock_result.failure_code = None
    mock_result.telemetry = {}
    mock_result.qa_report = mock_pre_qa

    mock_context = MagicMock()

    parts_malformed_first = [
        {"module_id": "module_1", "questions": None},
        {
            "module_id": "module_1",
            "questions": [
                {"slot_id": "m1_q1", "text": "Q1", "module_id": "module_1"}
            ],
        },
    ]
    full = [{"slot_id": "m1_q1", "text": "Q1", "module_id": "module_1"}]

    with unittest.mock.patch(
        "core.generation.agents.checking_agent.CheckingAgent._get_validated_expected_slots",
        return_value=["m1_q1"],
    ):
        _, _, rec_meta, _, _, _, _ = _reconcile_post_adaptation(
            result=mock_result,
            context=mock_context,
            paper_parts=parts_malformed_first,
            full_paper=full,
            cleanup_errors=[],
        )
        defects = rec_meta.get("fatal_defects", [])
        assert any("DUPLICATE_PAPER_PART_MODULE_ID:module_1" in d for d in defects)
        assert any("MALFORMED_PAPER_PART_QUESTIONS:index_0" in d for d in defects)


def test_finding_5_finalizer_enforces_raw_success_consistency():
    expected_ids = make_expected_slot_ids()
    ctx = make_test_context()

    # Inconsistent caller flags: raw_success=False but exec_succeeded=True
    reconciled_qa, _, _, _, _, _, eligible = _finalize_reconciliation(
        raw_success=False,
        exec_succeeded=True,
        expected_ids=expected_ids,
        f_post=expected_ids,
        p_post=[],
        b_post=[],
        fatal_defects=[],
        adaptation_defects=[],
        pre_status="PASS",
        context=ctx,
        copied_telemetry={},
        copied_cost_telemetry={},
        copied_cache_telemetry={},
        copied_result_telemetry={},
        serialized_pre_qa={},
        failure_code="INTERNAL_CRASH",
        validated_pre_blocked=[],
        validated_pre_degraded=[],
        resolved_population_changed=False,
        min_ratio=0.9,
        accounting_complete=True,
    )
    # Must fail-closed to STATUS_FAILED, never STATUS_PASS
    assert reconciled_qa.status == STATUS_FAILED
    assert eligible is False
    assert any("EXECUTION_FAILURE:INTERNAL_CRASH" in r for r in reconciled_qa.blocked_reasons)


def test_finding_6_telemetry_overwrite_protection():
    expected_ids = make_expected_slot_ids()
    ctx = make_test_context()

    malicious_extra = {
        "fatal_defects": ["FORGED_DEFECT_OVERWRITE"],
        "accounting_incomplete": False,
        "custom_plugin_metric": 99.5,
    }

    copied_tel = {}
    reconciled_qa, _, _, _, ser_rec, _, _ = _finalize_reconciliation(
        raw_success=True,
        exec_succeeded=True,
        expected_ids=expected_ids,
        f_post=expected_ids,
        p_post=[],
        b_post=[],
        fatal_defects=["AUTHENTIC_DEFECT"],
        adaptation_defects=[],
        pre_status="PASS",
        context=ctx,
        copied_telemetry=copied_tel,
        copied_cost_telemetry={},
        copied_cache_telemetry={},
        copied_result_telemetry={},
        serialized_pre_qa={},
        failure_code=None,
        validated_pre_blocked=[],
        validated_pre_degraded=[],
        resolved_population_changed=False,
        min_ratio=0.9,
        accounting_complete=False,
        extra_reconciliation_telemetry=malicious_extra,
    )

    bridge_tel = ser_rec["telemetry"]["bridge_reconciliation"]
    # Canonical integrity fields MUST NOT be overwritten
    assert bridge_tel["fatal_defects"] == ["AUTHENTIC_DEFECT"]
    assert bridge_tel["accounting_incomplete"] is True
    # Non-reserved extra field is preserved
    assert bridge_tel["custom_plugin_metric"] == 99.5
    # Raw extra telemetry preserved under segregated namespace
    assert "extra_telemetry" in bridge_tel
    assert bridge_tel["extra_telemetry"]["custom_plugin_metric"] == 99.5


def test_finding_7_full_paper_module_attribution_enforcement():
    mock_pre_qa = MagicMock(spec=QAReport)
    mock_pre_qa.status = "PASS"
    mock_pre_qa.to_dict.return_value = {"status": "PASS", "blocked_reasons": []}
    mock_pre_qa.telemetry = {}
    mock_pre_qa.cost_telemetry = {}
    mock_pre_qa.cache_telemetry = {}
    mock_pre_qa.blocked_reasons = []
    mock_pre_qa.degraded_reasons = []

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.questions = []
    mock_result.failure_code = None
    mock_result.telemetry = {}
    mock_result.qa_report = mock_pre_qa

    mock_context = MagicMock()

    # Case 7A: Mismatched module_id between full_paper and paper_parts
    parts = [
        {
            "module_id": "module_1",
            "questions": [
                {"slot_id": "m1_q1", "text": "Q1", "module_id": "module_1"}
            ],
        }
    ]
    full_mismatched = [
        {"slot_id": "m1_q1", "text": "Q1", "module_id": "module_999"}
    ]

    with unittest.mock.patch(
        "core.generation.agents.checking_agent.CheckingAgent._get_validated_expected_slots",
        return_value=["m1_q1"],
    ):
        _, _, rec_meta, _, _, _, eligible = _reconcile_post_adaptation(
            result=mock_result,
            context=mock_context,
            paper_parts=parts,
            full_paper=full_mismatched,
            cleanup_errors=[],
        )
        defects = rec_meta.get("fatal_defects", [])
        assert any(
            "QUESTION_MODULE_CROSS_REPRESENTATION_MISMATCH" in d
            for d in defects
        )
        assert eligible is False


def test_finding_8_emergency_handler_resilient_to_raising_len_object():
    class RaisingLenSequence(list):
        def __len__(self):
            raise ArithmeticError("RAISING_LEN_PROPERTY")

    # Must NOT escape with ArithmeticError
    parts, full, meta = _handle_emergency_pipeline_failure(
        exc=RuntimeError("Normal error"),
        expected_ids=RaisingLenSequence(["s1"]),
    )
    assert parts == []
    assert full == []
    assert meta["paper_status"] == STATUS_FAILED
    assert meta["export_eligible"] is False


def test_finding_9_serialization_fallback_guaranteed_json_serializable():
    expected_ids = make_expected_slot_ids()
    ctx = make_test_context()

    class UnserializableObject:
        def __repr__(self):
            return "<UnserializableObject>"

    # Intentionally inject an object that fails standard json serialization
    copied_tel = {"hostile_key": UnserializableObject()}

    reconciled_qa, _, _, _, ser_rec, _, _ = _finalize_reconciliation(
        raw_success=True,
        exec_succeeded=True,
        expected_ids=expected_ids,
        f_post=expected_ids,
        p_post=[],
        b_post=[],
        fatal_defects=[],
        adaptation_defects=[],
        pre_status="PASS",
        context=ctx,
        copied_telemetry=copied_tel,
        copied_cost_telemetry={},
        copied_cache_telemetry={},
        copied_result_telemetry={},
        serialized_pre_qa={},
        failure_code=None,
        validated_pre_blocked=[],
        validated_pre_degraded=[],
        resolved_population_changed=False,
        min_ratio=0.9,
        accounting_complete=True,
    )

    # Reconciled QA report should catch serialization error and produce valid JSON fallback
    assert ser_rec.get("_is_synthetic_fallback") is True
    # The ultimate test: json.dumps must succeed without throwing TypeError
    dumped = json.dumps(ser_rec, allow_nan=False)
    assert isinstance(dumped, str)
    assert len(dumped) > 0

