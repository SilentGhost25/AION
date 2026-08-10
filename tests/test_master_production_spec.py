"""
AION Master Production Specification — Automated Pytest Suite
=============================================================
Verifies all core specifications:
  - GenerationRequest contract validation & error handling.
  - PaperStructurePlan OR-pair mark parity invariant.
  - ExtractionGateway TXT_AS_SOURCE hard rejection.
  - QuestionPlanner single Bloom verb bounds & math protection.
  - FinalQualityGate single QA authority & Category A-F scoring.
  - End-to-end deterministic pipeline execution.
"""

import pytest
from core.contracts.generation_request import (
    GenerationRequest,
    GenerationRequestValidationError,
)
from core.contracts.paper_structure import (
    ORPairDescriptor,
    ORParityViolationError,
    PaperStructurePlan,
    SlotDescriptor,
)
from core.contracts.question import GeneratedQuestion, QuestionIntent
from core.extraction.gateway import DocumentArtifact, ExtractionError, ExtractionGateway
from core.planner.paper_planner import PaperStructurePlanner
from core.planner.question_planner import BLOOM_VERB_MAP, QuestionPlanner
from core.validators.final_gate import FinalQualityGate, QAResult
from v0_1.unified_pipeline import RawFile, _extract


def test_generation_request_validation():
    # Valid request
    req = GenerationRequest(
        request_id="req_001",
        received_at="2026-08-10T12:00:00Z",
        subject="Data Structures",
        department="CSE",
        semester=4,
        exam_type="SEE",
        modules=[1, 2, 3, 4, 5],
        total_marks=50,
        subquestion_count=2,
        distribution_policy="PRIMARY_HEAVY",
        bloom_levels=["L2", "L3"],
        document_id="doc_dsa_01",
    )
    validated = req.validate()
    assert validated.validated is True
    assert len(validated.validation_errors) == 0

    # Invalid request — invalid subquestion_count
    req_bad_subq = GenerationRequest(
        request_id="req_bad_subq",
        received_at="2026-08-10T12:00:00Z",
        subject="Data Structures",
        department="CSE",
        semester=4,
        exam_type="SEE",
        modules=[1, 2, 3, 4, 5],
        total_marks=50,
        subquestion_count=5,  # Invalid (>4)
    )
    with pytest.raises(GenerationRequestValidationError):
        req_bad_subq.validate()

    # Invalid request — indivisible total_marks
    req_indivisible = GenerationRequest(
        request_id="req_indiv",
        received_at="2026-08-10T12:00:00Z",
        subject="Data Structures",
        department="CSE",
        semester=4,
        exam_type="SEE",
        modules=[1, 2, 3],
        total_marks=50,  # 50 % 3 != 0
    )
    with pytest.raises(GenerationRequestValidationError):
        req_indivisible.validate()


def test_paper_structure_plan_or_parity():
    req = GenerationRequest(
        request_id="req_plan_01",
        received_at="2026-08-10T12:00:00Z",
        subject="Analog Electronics",
        department="ECE",
        semester=3,
        exam_type="IAT1",
        modules=[1, 2],
        total_marks=20,
        subquestion_count=2,
        distribution_policy="PRIMARY_HEAVY",
        bloom_levels=["L2", "L3"],
        document_id="doc_ece_01",
    )

    plan = PaperStructurePlanner.build(req)
    assert isinstance(plan, PaperStructurePlan)
    assert plan.total_questions == 4
    assert plan.marks_per_module == 10
    assert plan.mark_distribution == (6, 4)

    for pair in plan.or_pairs:
        assert len(pair.slots_a) == 2
        assert len(pair.slots_b) == 2
        marks_a = tuple(s.marks for s in pair.slots_a)
        marks_b = tuple(s.marks for s in pair.slots_b)
        assert marks_a == marks_b == (6, 4)

    # Test ORParityViolationError when parity is broken manually
    slot_a1 = SlotDescriptor("Q1a", 1, "a", 1, 6, "CO1", "L3", "descriptive")
    slot_a2 = SlotDescriptor("Q1b", 1, "b", 1, 4, "CO1", "L2", "descriptive")
    slot_b1_bad = SlotDescriptor("Q2a", 2, "a", 1, 5, "CO1", "L3", "descriptive")
    slot_b2_bad = SlotDescriptor("Q2b", 2, "b", 1, 5, "CO1", "L2", "descriptive")

    with pytest.raises(ORParityViolationError):
        ORPairDescriptor(
            module_id=1,
            alt_a_question_no=1,
            alt_b_question_no=2,
            total_marks=10,
            subquestion_count=2,
            mark_distribution=(6, 4),
            slots_a=(slot_a1, slot_a2),
            slots_b=(slot_b1_bad, slot_b2_bad),
        )


def test_extraction_gateway_txt_rejection(tmp_path):
    txt_file = tmp_path / "sample_syllabus.txt"
    txt_file.write_text("Sample syllabus content", encoding="utf-8")

    # Gateway rejection
    with pytest.raises(ExtractionError) as exc_info:
        ExtractionGateway.extract(str(txt_file))
    assert exc_info.value.code == "TXT_AS_SOURCE"


def test_question_planner_single_bloom_verb():
    slot = SlotDescriptor(
        slot_id="Q1a",
        question_no=1,
        sub_label="a",
        module_id=1,
        marks=6,
        co="CO1",
        bloom="L3",
        question_type="numerical",
    )

    art = DocumentArtifact(
        document_id="doc_test_01",
        source_path="sample.pdf",
        mime_type="application/pdf",
        page_count=5,
        chunks=[{
            "chunk_id": "chk_01",
            "concept_id": "c_01",
            "topic": "Ohm's Law",
            "text": r"Voltage $V = I R$ across a resistor of resistance $R = 10\ \Omega$.",
            "page_start": 2,
            "module_id": 1,
            "concept_tags": ["circuits"],
        }],
    )

    intent = QuestionPlanner.build_intent(slot, art, seed=123)
    assert isinstance(intent, QuestionIntent)
    assert intent.marks == 6
    assert intent.bloom == "L3"
    assert intent.bloom_verb in BLOOM_VERB_MAP["L3"]
    assert "[MATH:" in intent.evidence_text[0]  # Math protected into placeholder


def test_final_quality_gate_single_authority():
    req = GenerationRequest(
        request_id="req_qa_01",
        received_at="2026-08-10T12:00:00Z",
        subject="Control Systems",
        department="EEE",
        semester=5,
        exam_type="SEE",
        modules=[1],
        total_marks=10,
        subquestion_count=2,
        distribution_policy="PRIMARY_HEAVY",
        document_id="doc_eee_01",
    )

    plan = PaperStructurePlanner.build(req)

    gq1a = GeneratedQuestion(
        slot_id="Q1a",
        question_text=r"Calculate the transfer function $G(s)$ for the given system.",
        marks=6,
        bloom="L3",
        co="CO1",
    )
    gq1b = GeneratedQuestion(
        slot_id="Q1b",
        question_text=r"Explain the physical significance of damping ratio $\zeta$.",
        marks=4,
        bloom="L2",
        co="CO1",
    )
    gq2a = GeneratedQuestion(
        slot_id="Q2a",
        question_text=r"Determine the state space model representation of the system.",
        marks=6,
        bloom="L3",
        co="CO1",
    )
    gq2b = GeneratedQuestion(
        slot_id="Q2b",
        question_text=r"Describe the Routh-Hurwitz stability criterion for polynomial $P(s)$.",
        marks=4,
        bloom="L2",
        co="CO1",
    )

    qa_res = FinalQualityGate.evaluate(plan, [gq1a, gq1b, gq2a, gq2b])
    assert isinstance(qa_res, QAResult)
    assert qa_res.status in {"PASS", "PASS_WITH_WARNINGS"}
    assert qa_res.qa_score >= 75.0

    # Incompatible Bloom verbs violation (B04)
    gq_bad_verbs = GeneratedQuestion(
        slot_id="Q1a",
        question_text="Analyze and Create the circuit model for amplifier.",
        marks=6,
        bloom="L4",
        co="CO1",
    )
    qa_bad = FinalQualityGate.evaluate(plan, [gq_bad_verbs, gq1b, gq2a, gq2b])
    assert any("B04" in f for f in qa_bad.failures)

    # M3 Unicode replacement character violation (D01)
    gq_corrupt = GeneratedQuestion(
        slot_id="Q1a",
        question_text="Calculate resistance R = \ufffd ohms.",
        marks=6,
        bloom="L3",
        co="CO1",
    )
    qa_corrupt = FinalQualityGate.evaluate(plan, [gq_corrupt, gq1b, gq2a, gq2b])
    assert any("D01" in f for f in qa_corrupt.failures)


def test_master_production_end_to_end_state_machine():
    req = GenerationRequest(
        request_id="req_e2e_01",
        received_at="2026-08-10T12:00:00Z",
        subject="Algorithms",
        department="CSE",
        semester=4,
        exam_type="IAT1",
        modules=[1, 2],
        total_marks=20,
        subquestion_count=2,
        distribution_policy="PRIMARY_HEAVY",
        bloom_levels=["L2", "L3"],
        document_id="doc_algo_01",
    ).validate()

    # Step 1: Build paper structure plan
    plan = PaperStructurePlanner.build(req)
    assert plan.total_questions == 4

    # Step 2: Extract document
    art = DocumentArtifact(
        document_id=req.document_id,
        source_path="algo_lecture.pdf",
        mime_type="application/pdf",
        page_count=12,
        chunks=[{
            "chunk_id": f"chk_{m}",
            "concept_id": f"c_{m}",
            "topic": f"Algorithm Topic {m}",
            "text": f"Dijkstra algorithm shortest path computation for module {m}.",
            "page_start": m,
            "module_id": m,
            "concept_tags": ["graphs"],
        } for m in [1, 2]],
    )

    # Step 3: Build question intents
    generated_questions = []
    for slot in plan.get_all_slots():
        intent = QuestionPlanner.build_intent(slot, art, seed=req.seed or 42)
        assert intent.marks == slot.marks
        gq = GeneratedQuestion(
            slot_id=slot.slot_id,
            question_text=f"{intent.bloom_verb} the algorithm for {intent.concept}.",
            marks=slot.marks,
            bloom=slot.bloom,
            co=slot.co,
        )
        generated_questions.append(gq)

    # Step 4: Final Quality Gate Single Authority Evaluation
    qa_res = FinalQualityGate.evaluate(plan, generated_questions)
    assert qa_res.status in {"PASS", "PASS_WITH_WARNINGS"}
    assert qa_res.qa_score >= 75.0
