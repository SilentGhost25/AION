"""
Frontend-to-Backend Request Contract Test
==========================================
Verifies that canonical frontend request payloads map cleanly to GenerationRequest
contracts without dropping fields or silently defaulting modules/bloom levels.
"""

import pytest
from v0_1.contracts import Evidence, ExamType, GenerationRequest, QuestionSpec, RawFile, RetrievedEvidence
from v0_1.paper_template import PaperTemplateBuilder


def test_frontend_payload_mapping():
    frontend_payload = {
        "subject": "Data Structures",
        "department": "AIML",
        "semester": 3,
        "exam_type": "IAT-1",
        "selected_modules": [1, 2, 3, 4, 5],
        "bloom_levels": ["L2", "L3", "L4"],
        "difficulty": "MIXED",
        "model": "qwen2.5:14b",
        "visual_mode": True,
    }

    # Verify ExamType mapping
    exam_str = frontend_payload["exam_type"].upper()
    exam_type = ExamType.IA if any(k in exam_str for k in ["IA", "IAT", "MID"]) else ExamType.SEE
    assert exam_type == ExamType.IA

    # Build paper template
    builder = PaperTemplateBuilder()
    template = builder.build(
        exam_type=exam_type.value,
        n_modules=len(frontend_payload["selected_modules"]),
        subject=frontend_payload["subject"],
    )

    assert template.attemptable_marks == 50
    assert len(template.question_slots) > 0

    dummy_ev_unit = Evidence(
        chunk_ids=["chunk_1"],
        texts=["Data structures syllabus"],
        combined_text="Data structures syllabus and algorithms text",
        module_index=1,
        evidence_score=0.95,
        word_count=50,
        query="Data Structures",
    )
    dummy_evidence = RetrievedEvidence(
        doc_id="frontend_doc_001",
        evidence_by_module={i: dummy_ev_unit for i in frontend_payload["selected_modules"]},
    )

    # Build GenerationRequest specs
    specs = []
    for q_slot in template.question_slots:
        for sub in q_slot.sub_slots:
            specs.append(QuestionSpec(
                spec_id=sub.slot_id,
                module_index=q_slot.module_index,
                q_number=q_slot.q_number,
                part_letter=sub.letter,
                marks=sub.marks,
                bloom_level=sub.bloom_level,
                bloom_verb=sub.verb,
                co=sub.co,
                is_or=q_slot.is_or,
                evidence=dummy_evidence,
                exam_type=exam_type,
            ))

    gen_req = GenerationRequest(
        doc_id="frontend_doc_001",
        specs=specs,
        exam_type=exam_type,
        subject=frontend_payload["subject"],
        total_marks=template.attemptable_marks,
    )

    assert gen_req.doc_id == "frontend_doc_001"
    assert gen_req.subject == "Data Structures"
    assert gen_req.total_marks == 50
    assert len(gen_req.specs) > 0

    # Ensure no empty specs or missing module indices
    modules_mapped = {s.module_index for s in gen_req.specs}
    assert modules_mapped == set(frontend_payload["selected_modules"])
