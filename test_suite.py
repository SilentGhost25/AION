"""
=============================================================================
AION Master Model Verification & Parameter Test Suite
=============================================================================
A compact, robust test harness for the current build of AION.
Validates:
  1. PaperSpec & Partition Allocation (IAT1, IAT2, SEE, ELECTIVE_3MOD)
  2. Bloom Taxonomy Verification & Opening Verb Alignment (L1..L6)
  3. Subquestion Diversity within Same Module (Zero Duplicates / Jaccard < 0.45)
  4. Multi-Part Question Prompting & Task Section Generation
  5. Slot Filler & Evidence Grounding Contracts
  6. Full End-to-End Paper Assembly (Q1..Q10, CO1..CO5, Marks Sum)
  7. DOCX Exporter Pipeline & Document Integrity
  8. Fail-Closed Quality Gate & Unresolved Slot Blocking

Can be executed directly: `python test_suite.py`
Or via pytest:            `python -m pytest test_suite.py -v`
=============================================================================
"""

import os
import sys
import json
import re
import io
import time
from typing import Dict, List, Any

# Ensure legacy orchestrator is permitted in test environments
os.environ["AION_ALLOW_LEGACY_ORCHESTRATOR"] = "true"

from core.contracts.paper_spec import PaperSpec
from core.generation.paper_spec_resolver import resolve_paper_spec
from core.contracts.question_slot import QuestionSlot, SlotStatus, UnresolvedSlotException
from core.contracts.question import GeneratedQuestion
from core.contracts.budgets import AnswerBudget, QuestionBudget
from core.contracts.task_signature import TaskSignature
from core.generation.output_schema import QuestionOutput
from core.validation.bloom_validator import BLOOM_VERB_LEVEL_MAP, check_bloom_two_layer
from core.generation.agents.legacy_adapter import (
    to_legacy_shape,
    _extract_subpart_texts,
    _strip_shared_prefix,
    _to_question_dict,
    _clean_topic as legacy_clean_topic,
    _specific_overlap,
)
from core.generation.agents.generated_question import GeneratedQuestion as AgentGeneratedQuestion
from core.generation.agents.prompts import build_task_section, build_topic_section
from core.generation.agents.question_plan import QuestionPlan
from core.generation.agents.local_evaluators import (
    topic_fragment_evaluator,
    bloom_content_evaluator,
    scenario_consistency_evaluator,
    sibling_independence_evaluator,
)
from core.generation.agents.deterministic_repairs import (
    repair_topic_fragment_and_artifacts,
    repair_bloom_depth_mismatch,
    repair_scenario_contradiction,
    repair_sibling_independence,
)
from core.generation.agents.evaluation_contracts import (
    TOPIC_FRAGMENT,
    BLOOM_DEPTH_MISMATCH,
    SCENARIO_CONTRADICTION,
    SIBLING_INTERCONNECTED,
)
from core.generation.agents.planning_agent import PlanningAgent
from v0_1.slot_filler import fill_slot
from v0_1.docx_export import generate_docx_from_paper
from aion_api import _format_paper


# =============================================================================
# Helper Utilities
# =============================================================================

def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute token-level Jaccard similarity between two texts."""
    w1 = set(re.findall(r"\w+", text1.lower()))
    w2 = set(re.findall(r"\w+", text2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


# =============================================================================
# Test 1: PaperSpec & Dynamic Partition Allocation
# =============================================================================

def test_paperspec_and_partition_allocation():
    """Verify PaperSpec resolution, partition allocations, and CO mappings."""
    test_configs = [
        # (exam_type, subject, requested_split, expected_mods, expected_q_per_mod)
        ("IAT1", "Satellite Communication", [[10], [10], [6, 4], [6, 4], [6, 4], [6, 4], [10], [10], [10], [10]], 5, 2),
        ("IAT2", "Cloud Computing", [[6, 4], [6, 4], [10], [10], [6, 4], [6, 4], [6, 4], [6, 4], [10], [10]], 5, 2),
        ("SEE", "Data Structures", [[8, 6, 6], [10, 10], [20], [8, 6, 6], [10, 10], [20], [8, 6, 6], [10, 10], [20], [20]], 5, 2),
        ("ELECTIVE_3MOD", "VLSI Design", [[10], [6, 4], [5, 5], [10], [6, 4], [5, 5], [10], [6, 4], [5, 5]], 3, 3),
    ]

    for exam_type, subject, requested_split, exp_mods, exp_q_per_mod in test_configs:
        spec = resolve_paper_spec(
            exam_type=exam_type,
            override={"subject": subject, "marks_split": requested_split},
        )
        assert spec.module_count == exp_mods, f"Expected {exp_mods} modules for {exam_type}, got {spec.module_count}"
        assert spec.questions_per_module == exp_q_per_mod, f"Expected {exp_q_per_mod} q/mod, got {spec.questions_per_module}"
        assert spec.total_questions == len(requested_split)

        # Verify allocation matching requested split
        allocated = spec.allocate_partitions(requested_split)
        for (m_idx, q_idx), partition in allocated.items():
            flat_idx = m_idx * exp_q_per_mod + q_idx
            expected = requested_split[flat_idx]
            assert partition == expected, f"Partition mismatch at Q{flat_idx+1}: {partition} != {expected}"

        # Verify CO distribution
        co_map = spec.build_co_map()
        assert len(co_map) == exp_mods
        for m in range(1, exp_mods + 1):
            assert str(m) in co_map


# =============================================================================
# Test 2: Bloom Taxonomy & Strict Opening Verb Verification
# =============================================================================

def test_bloom_taxonomy_and_opening_verbs():
    """Verify that every Bloom level (L1..L6) strictly enforces approved opening verbs."""
    for level, canonical_verbs in BLOOM_VERB_LEVEL_MAP.items():
        assert len(canonical_verbs) >= 5, f"Bloom level {level} should have at least 5 canonical verbs"
        for v in canonical_verbs:
            assert v.islower(), f"Canonical verb '{v}' in {level} should be lowercased"

    # Test two-layer validator
    slot_l2 = QuestionSlot(
        slot_id="slot_l2", question_no=1, sub_label="a", or_pair_id="OR_1",
        is_alternative=False, module_id=1, marks=6, bloom_level="L2",
        bloom_verb="Explain", bloom_operation="EXPLAIN", co="CO1",
        difficulty="MEDIUM", question_type="descriptive", topic="Networks",
        evidence_ids=("e1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L2"),
        question_budget=QuestionBudget.from_bloom("L2", 6),
        task_signature=TaskSignature.from_bloom_marks_type("L2", 6, "descriptive"),
        math_required=False, visual_required=False,
    )

    # Valid L2 instruction
    res_valid = check_bloom_two_layer("Explain the operation of sliding window protocol.", slot_l2)
    assert res_valid.passed is True

    # Valid with British spelling variant
    slot_l4 = QuestionSlot(
        slot_id="slot_l4", question_no=2, sub_label="b", or_pair_id="OR_1",
        is_alternative=False, module_id=1, marks=4, bloom_level="L4",
        bloom_verb="Analyze", bloom_operation="ANALYZE", co="CO1",
        difficulty="HARD", question_type="descriptive", topic="Networks",
        evidence_ids=("e2",),
        answer_budget=AnswerBudget.from_marks_and_bloom(4, "L4"),
        question_budget=QuestionBudget.from_bloom("L4", 4),
        task_signature=TaskSignature.from_bloom_marks_type("L4", 4, "descriptive"),
        math_required=False, visual_required=False,
    )
    res_uk = check_bloom_two_layer("Analyse the congestion window dynamics under packet drop.", slot_l4)
    assert res_uk.passed is True

    # Wrong Bloom verb (Describe in an L4 slot)
    res_mismatch = check_bloom_two_layer("Describe the protocol header fields.", slot_l4)
    assert res_mismatch.passed is False
    assert res_mismatch.code == "BLOOM_VERB_WRONG"


# =============================================================================
# Test 3: Subquestion Diversity within Same Module (Zero Duplicates)
# =============================================================================

def test_subquestion_diversity_within_same_module():
    """Verify subquestions addressing the same module are distinct and not paraphrases, and overlap is tracked."""
    # Multi-part question with distinct subquestions
    distinct_question = AgentGeneratedQuestion(
        slot_id="mod_1_Q1",
        module_id="module_1",
        global_q_idx=1,
        question_text="(a) Explain the satellite link budget parameters in detail. (b) Describe the carrier-to-noise ratio parameters for the uplink receiver.",
        solution="Model solution for link budget.",
        marking_scheme=[{"criterion": "Equations", "marks": 4}, {"criterion": "Derivation", "marks": 6}],
        marks=10,
        partition=[6, 4],
        bloom="L2",
        co="CO1",
        topic="Satellite Link Budget",
    )

    q_dict = _to_question_dict(distinct_question)
    subs = q_dict["sub_questions"]
    assert len(subs) == 2, f"Expected 2 subquestions, got {len(subs)}"
    assert subs[0]["marks"] == 6
    assert subs[1]["marks"] == 4

    text_a = subs[0]["text"].strip()
    text_b = subs[1]["text"].strip()

    # Must NOT be identical
    assert text_a != text_b, "Subquestions (a) and (b) must not be identical!"

    # Jaccard similarity must be strictly below 0.45 threshold
    sim = jaccard_similarity(text_a, text_b)
    assert sim < 0.45, f"Subquestions have high similarity ({sim:.2f} >= 0.45): \n(a): {text_a}\n(b): {text_b}"

    # Verify both opening words match approved Bloom verbs
    verb_a = text_a.split()[0].rstrip(".,;:").lower()
    verb_b = text_b.split()[0].rstrip(".,;:").lower()
    valid_verbs = BLOOM_VERB_LEVEL_MAP["L2"]
    assert verb_a in valid_verbs, f"Verb '{verb_a}' not in L2"
    assert verb_b in valid_verbs, f"Verb '{verb_b}' not in L2"
    assert q_dict["subquestion_overlap"] is False

    # Verify that when raw generation omits subpart markers, the adapter NEVER
    # duplicates text across slots or fabricates templates; it emits 1 unresolved
    # aggregate entry with MULTI_PART_STRUCTURE_MISMATCH and skips pairwise overlap.
    missing_markers_question = AgentGeneratedQuestion(
        slot_id="mod_1_Q1_fallback",
        module_id="module_1",
        global_q_idx=1,
        question_text="Explain the satellite link budget parameters and calculate the carrier-to-noise ratio.",
        solution="Model solution for link budget.",
        marking_scheme=[{"criterion": "Equations", "marks": 4}, {"criterion": "Derivation", "marks": 6}],
        marks=10,
        partition=[6, 4],
        bloom="L2",
        co="CO1",
        topic="Satellite Link Budget",
    )
    q_dict_fallback = _to_question_dict(missing_markers_question)
    assert q_dict_fallback["is_unresolved"] is True
    assert q_dict_fallback["adaptation_failure_code"] == "MULTI_PART_STRUCTURE_MISMATCH"
    # Never duplicate question text across partition slots
    assert len(q_dict_fallback["sub_questions"]) == 1
    assert q_dict_fallback["sub_questions"][0]["text"] == missing_markers_question.question_text
    assert q_dict_fallback["sub_questions"][0]["marks"] == 10
    assert q_dict_fallback["sub_questions"][0]["is_unresolved"] is True
    assert q_dict_fallback["sub_questions"][0]["adaptation_failure_code"] == "MULTI_PART_STRUCTURE_MISMATCH"
    # Overlap detection is not incorrectly triggered on the aggregate fallback
    assert q_dict_fallback["subquestion_overlap"] is False
    assert q_dict_fallback["overlap_details"] == []
    for sq in q_dict_fallback["sub_questions"]:
        assert "operational characteristics, critical trade-offs" not in sq["text"]

    # Verify that genuine duplicate subquestions with markers are tracked as overlap
    duplicate_question = AgentGeneratedQuestion(
        slot_id="mod_1_Q1_dup",
        module_id="module_1",
        global_q_idx=1,
        question_text="(a) Explain the satellite link budget parameters in detail. (b) Explain the satellite link budget parameters in detail.",
        solution="Model solution for link budget.",
        marking_scheme=[{"criterion": "Equations", "marks": 4}, {"criterion": "Derivation", "marks": 6}],
        marks=10,
        partition=[6, 4],
        bloom="L2",
        co="CO1",
        topic="Satellite Link Budget",
    )
    q_dict_dup = _to_question_dict(duplicate_question)
    assert q_dict_dup["subquestion_overlap"] is True
    assert len(q_dict_dup["overlap_details"]) > 0
    assert q_dict_dup["telemetry"]["subquestion_overlap"] is True


# =============================================================================
# Test 4: Multi-Part Prompt Instructions & Sibling Independence Directive
# =============================================================================

def test_multipart_prompt_instructions():
    """Verify that multi-part question prompts enforce subquestion diversity, distinct topics, and independence."""
    plan = QuestionPlan(
        slot_id="mod_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[6, 4],
        total_marks=10,
        bloom="L3",
        topic="Machine Learning Deployment",
        topic_a="Challenges of deploying ML in production",
        topic_b="Find-S algorithm trace and hypothesis representation",
    )

    prompt_section = build_task_section(plan)
    assert "Structure the question as sub-parts with these marks: part 1: 6 marks, part 2: 4 marks." in prompt_section
    assert "MULTI-PART SIBLING INDEPENDENCE" in prompt_section
    assert "(a) and (b) MUST be independently answerable" in prompt_section
    assert "MUST test DIFFERENT concepts from the module" in prompt_section
    assert "must NOT say 'with reference to (a)'" in prompt_section
    assert "GOOD EXAMPLE" in prompt_section

    # Test topic section splits into two distinct topics
    topic_section = build_topic_section(plan)
    assert "[TOPIC FOR (a) — 6 MARKS]" in topic_section
    assert "[TOPIC FOR (b) — 4 MARKS]" in topic_section
    assert "Challenges of deploying ML in production" in topic_section
    assert "Find-S algorithm trace and hypothesis representation" in topic_section
    assert "CRITICAL SIBLING INDEPENDENCE RULE" in topic_section


# =============================================================================
# Test 5: Slot Filler & Module Grounding
# =============================================================================

def test_slot_filler_module_grounding():
    """Verify slot_filler generates differentiated subquestions from module context."""
    from unittest.mock import patch
    class MockSpec:
        part_letter = "a"
        bloom_level = 2
        bloom_verb = "Explain"
        marks = 6
        evidence = None

    spec_a = MockSpec()
    spec_a.part_letter = "a"
    spec_a.marks = 6
    spec_a.bloom_verb = "Explain"

    spec_b = MockSpec()
    spec_b.part_letter = "b"
    spec_b.marks = 4
    spec_b.bloom_verb = "Analyze"

    with patch("v0_1.generator.get_vtu_vibe_question", side_effect=RuntimeError("LLM offline")):
        text_a = fill_slot(spec_a, context="Operating Systems Memory Management", subject="Operating Systems")
        text_b = fill_slot(spec_b, context="Operating Systems Memory Management", subject="Operating Systems")

    assert text_a != text_b
    assert "6 Marks" in text_a
    assert "4 Marks" in text_b
    assert text_a.startswith("Explain")
    assert text_b.startswith("Analyze")

    # Verify Jaccard diversity
    sim = jaccard_similarity(text_a, text_b)
    assert sim < 0.45, f"Slot filler generated too-similar questions: sim={sim:.2f}"


# =============================================================================
# Test 6: Full End-to-End VTU Paper Assembly
# =============================================================================

def test_full_vtu_paper_assembly():
    """Verify end-to-end assembly of a complete 10-question VTU examination paper."""
    subject_name = "Cloud Computing"
    exam_type = "IAT1"
    requested_split = [[6, 4], [6, 4], [10], [10], [6, 4], [6, 4], [6, 4], [6, 4], [10], [10]]
    total_marks_per_q = 10

    raw_modules = []
    co_distribution = {}

    module_subtopics = {
        1: [
            "fundamental service models and virtualization architecture",
            "resource scheduling overhead and multitenancy isolation constraints",
        ],
        2: [
            "hypervisor architecture and hardware virtualization mechanisms",
            "live migration performance trade-offs and network latency bottlenecks",
        ],
        3: [
            "distributed storage architectures and block replication invariants",
            "consistency guarantees and Byzantine fault tolerance boundaries",
        ],
        4: [
            "cloud security perimeter defenses and identity federation mechanisms",
            "compliance auditing boundaries and data exfiltration vulnerabilities",
        ],
        5: [
            "serverless event architectures and container orchestration topologies",
            "autoscaling cold-start latency and provisioning economics",
        ],
    }

    for m_zero in range(5):
        mod_idx = m_zero + 1
        co_tag = f"CO{mod_idx}"
        co_distribution[co_tag] = 20
        mod_questions = []

        for q_zero in range(2):
            q_num = m_zero * 2 + (q_zero + 1)
            partition = requested_split[q_num - 1]
            sub_questions = []

            for s_idx, marks in enumerate(partition):
                sub_label = chr(ord('a') + s_idx)
                verb = "Explain" if s_idx == 0 else "Analyze"
                bloom = "L2" if s_idx == 0 else "L4"
                subtopics = module_subtopics.get(mod_idx, ["fundamental principles", "operational trade-offs"])
                topic_desc = subtopics[s_idx % len(subtopics)]
                q_text = f"{verb} {topic_desc} in {subject_name} Module {mod_idx}."

                sub_questions.append({
                    "letter": sub_label,
                    "text": q_text,
                    "marks": marks,
                    "co": co_tag,
                    "bloom": bloom,
                })

            # Check intra-question diversity if multi-part
            if len(sub_questions) > 1:
                sim = jaccard_similarity(sub_questions[0]["text"], sub_questions[1]["text"])
                assert sim < 0.45, f"Intra-question similarity too high in Q{q_num}: {sim:.2f}"

            mod_questions.append({
                "mq_index": q_num,
                "total_marks": total_marks_per_q,
                "bloom_level": 2,
                "bloom_name": "Understand",
                "sub_questions": sub_questions,
                "is_or": (q_zero == 1),
            })

        raw_modules.append({
            "module_index": mod_idx,
            "module_title": f"Module {mod_idx}: {subject_name} Domain {mod_idx}",
            "questions": mod_questions,
        })

    qa_report = {"paper_status": "OK", "degraded_reasons": []}
    formatted = _format_paper(raw_modules, subject_name, exam_type, "turbo", qa_report=qa_report)

    # Validate paper invariants
    modules = formatted["modules"]
    assert len(modules) == 5, f"Expected 5 modules, got {len(modules)}"

    total_q_count = 0
    for mod in modules:
        for q in mod["questions"]:
            total_q_count += 1
            # Marks sum
            sub_marks = sum(sq["marks"] for sq in q["subQuestions"])
            assert sub_marks == total_marks_per_q, f"Q{total_q_count} marks {sub_marks} != {total_marks_per_q}"
            assert q["totalMarks"] == total_marks_per_q
            # Bloom opening keywords
            for sq in q["subQuestions"]:
                first_w = sq["text"].strip().split()[0].rstrip(".,;:").lower()
                bloom_tag = sq["bloom"]
                assert first_w in BLOOM_VERB_LEVEL_MAP[bloom_tag], (
                    f"First word '{first_w}' in Q{total_q_count}({sq['letter']}) does not match {bloom_tag}"
                )

    assert total_q_count == 10, f"Expected 10 questions, got {total_q_count}"


# =============================================================================
# Test 7: DOCX Export Generation & File Integrity
# =============================================================================

def test_docx_export_generation():
    """Verify that the formatted paper renders into a structurally sound DOCX byte stream."""
    sample_paper = {
        "title": "Model Verification Examination",
        "subject": "Distributed Systems",
        "examType": "IAT1",
        "modules": [
            {
                "module_id": "module_1",
                "title": "Module 1: Consistency and Consensus",
                "questions": [
                    {
                        "questionNumber": 1,
                        "totalMarks": 10,
                        "subQuestions": [
                            {
                                "letter": "a",
                                "text": "Explain the architectural models of distributed consensus in detail.",
                                "marks": 6,
                                "co": "CO1",
                                "bloom": "L2",
                            },
                            {
                                "letter": "b",
                                "text": "Analyze the operational trade-offs between Paxos and Raft under partition.",
                                "marks": 4,
                                "co": "CO1",
                                "bloom": "L4",
                            },
                        ],
                    }
                ],
            }
        ],
    }

    docx_buf = generate_docx_from_paper(sample_paper)
    assert docx_buf is not None
    assert isinstance(docx_buf, io.BytesIO)
    content = docx_buf.getvalue()
    assert len(content) > 5000, f"Generated DOCX size ({len(content)} bytes) suspiciously small"
    # Verify standard PK zip header for docx (.docx is a zip archive starting with 'PK\x03\x04')
    assert content[:4] == b"PK\x03\x04", "Generated output is not a valid OpenXML DOCX archive"


# =============================================================================
# Test 8: Fail-Closed Quality Gate
# =============================================================================

def test_fail_closed_quality_gate():
    """Verify that corrupt or contaminated slots fail closed without salvaging."""
    from core.contracts.question_slot import QuestionSlot, SlotStatus, UnresolvedSlotException
    from core.contracts.question import GeneratedQuestion
    from core.validation.export_gate import ExportGate

    slot = QuestionSlot(
        slot_id="mod_1_Q1_corrupt",
        question_no=1,
        sub_label="a",
        or_pair_id="OR_1",
        is_alternative=False,
        module_id=1,
        marks=10,
        bloom_level="L2",
        bloom_verb="Explain",
        bloom_operation="EXPLAIN",
        co="CO1",
        difficulty="MEDIUM",
        question_type="descriptive",
        topic="Corrupt Slot Test",
        evidence_ids=("c1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(10, "L2"),
        question_budget=QuestionBudget.from_bloom("L2", 10),
        task_signature=TaskSignature.from_bloom_marks_type("L2", 10, "descriptive"),
        math_required=False,
        visual_required=False,
        status=SlotStatus.PENDING.value,
    )

    stub = GeneratedQuestion.create_unresolved(
        slot, failure_code="MULTI_SLOT_CONTAMINATION", reason="Contaminated content"
    )
    decision = ExportGate.validate([stub])
    assert decision.passed is False, "ExportGate must fail-closed when an unresolved slot is present"
    assert decision.code == "UNRESOLVED_SLOT_DETECTED"


# =============================================================================
# Test 9: Topic Fragment Rejection & Deterministic Repair
# =============================================================================

def test_topic_fragment_evaluator_and_repair():
    """Verify topic_fragment_evaluator flags ungrammatical fragments and repairs them."""
    # Defect from audit: Q4(b)
    q_garbled_1 = AgentGeneratedQuestion(
        slot_id="slot_q4b",
        module_id="module_2",
        global_q_idx=4,
        question_text="Apply the operational characteristics, critical trade-offs, and boundary constraints of executed by multiple processors parallel in detail.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Trade-offs", "marks": 4}],
        marks=4,
        bloom="L3",
        topic="executed by multiple processors parallel",
    )
    verdict_1 = topic_fragment_evaluator(q_garbled_1)
    assert verdict_1 is not None, "Failed to flag participle topic fragment 'executed by...'"
    assert verdict_1.reason_code == TOPIC_FRAGMENT

    # Defect from audit: Q9(b) with stray bullet and clause fragment
    q_garbled_2 = AgentGeneratedQuestion(
        slot_id="slot_q9b",
        module_id="module_5",
        global_q_idx=9,
        question_text="Appraise the operational characteristics, critical trade-offs, and boundary constraints of ▪ Indefinite blocking, where low-priority processes in detail.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Critique", "marks": 4}],
        marks=4,
        bloom="L5",
        topic="▪ Indefinite blocking, where low-priority processes",
    )
    verdict_2 = topic_fragment_evaluator(q_garbled_2)
    assert verdict_2 is not None, "Failed to flag stray bullet and clause fragment"
    assert verdict_2.reason_code == TOPIC_FRAGMENT

    # Deterministic repair of garbled question 2
    repaired_2 = repair_topic_fragment_and_artifacts(q_garbled_2, verdict_2.reason_code, verdict_2.detail)
    assert repaired_2 is not None
    assert "▪" not in repaired_2.question_text
    assert ", where" not in repaired_2.question_text
    assert "Indefinite blocking" in repaired_2.question_text
    assert topic_fragment_evaluator(repaired_2) is None, f"Repaired question should pass: {repaired_2.question_text}"

    # Valid question should pass cleanly
    q_valid = AgentGeneratedQuestion(
        slot_id="slot_q1",
        module_id="module_1",
        global_q_idx=1,
        question_text="Explain the architectural structure and primary operating principles of virtual memory management in detail.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Principles", "marks": 6}],
        marks=6,
        bloom="L2",
        topic="Virtual Memory Management",
    )
    assert topic_fragment_evaluator(q_valid) is None


# =============================================================================
# Test 10: Bloom Content & Cognitive Depth Verification
# =============================================================================

def test_bloom_content_evaluator_and_depth():
    """Verify bloom_content_evaluator detects descriptive veneer and subpart depth mismatches."""
    # Defect from audit: Q8(a) marked L5 but is descriptive recall
    q_veneer = AgentGeneratedQuestion(
        slot_id="slot_q8",
        module_id="module_4",
        global_q_idx=8,
        question_text="(a) Evaluate the role of the system call table. (b) Provide a step-by-step example of handling a user interrupt.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        bloom="L5",
        topic="System Calls",
    )
    verdict = bloom_content_evaluator(q_veneer)
    assert verdict is not None, "Failed to detect L5 descriptive veneer 'Evaluate the role of'"
    assert verdict.reason_code == BLOOM_DEPTH_MISMATCH

    # Repair should elevate the cognitive demand
    repaired = repair_bloom_depth_mismatch(q_veneer, verdict.reason_code, verdict.detail)
    assert repaired is not None
    assert "Critically evaluate" in repaired.question_text
    assert bloom_content_evaluator(repaired) is None, f"Elevated question should pass: {repaired.question_text}"

    # Defect: L4 slot with subpart (b) starting with L2 'Describe'
    q_l4_mismatch = AgentGeneratedQuestion(
        slot_id="slot_q6",
        module_id="module_3",
        global_q_idx=6,
        question_text="(a) Analyze the cache coherence protocols. (b) Describe the bus snooping mechanism in SMP systems.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        bloom="L4",
        topic="Multiprocessors",
    )
    verdict_l4 = bloom_content_evaluator(q_l4_mismatch)
    assert verdict_l4 is not None
    assert verdict_l4.reason_code == BLOOM_DEPTH_MISMATCH

    repaired_l4 = repair_bloom_depth_mismatch(q_l4_mismatch, verdict_l4.reason_code, verdict_l4.detail)
    assert repaired_l4 is not None
    assert "(b) Analyze the operational mechanisms of" in repaired_l4.question_text
    assert bloom_content_evaluator(repaired_l4) is None


# =============================================================================
# Test 11: Scenario Consistency Evaluator
# =============================================================================

def test_scenario_consistency_evaluator():
    """Verify scenario_consistency_evaluator detects contradictory premises."""
    # Defect from audit: Q9(a) Round-robin + monopolize CPU
    q_rr_conflict = AgentGeneratedQuestion(
        slot_id="slot_q9a",
        module_id="module_5",
        global_q_idx=9,
        question_text="Consider a scenario where a high-priority process monopolizes the CPU for 10 seconds, leaving a low-priority process waiting. Calculate the wait time if the system uses a round-robin scheduling algorithm with a time quantum of 2 seconds.",
        solution="Model calculation.",
        marking_scheme=[{"criterion": "Calculation", "marks": 6}],
        marks=6,
        bloom="L3",
        topic="CPU Scheduling",
    )
    verdict_rr = scenario_consistency_evaluator(q_rr_conflict)
    assert verdict_rr is not None, "Failed to detect round-robin quantum vs monopolization contradiction"
    assert verdict_rr.reason_code == SCENARIO_CONTRADICTION

    repaired_rr = repair_scenario_contradiction(q_rr_conflict, verdict_rr.reason_code, verdict_rr.detail)
    assert repaired_rr is not None
    assert "priority scheduling algorithm without preemption" in repaired_rr.question_text
    assert scenario_consistency_evaluator(repaired_rr) is None

    # Defect from audit: Q5(a) 30 fps vs 1 sec per frame
    q_rate_conflict = AgentGeneratedQuestion(
        slot_id="slot_q5a",
        module_id="module_3",
        global_q_idx=5,
        question_text="Consider real-time video streaming at 30 frames per second, where each frame is 1 MB in size. Assume no compression and a transmission rate of 1 second per frame.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Bandwidth", "marks": 6}],
        marks=6,
        bloom="L3",
        topic="Multimedia Transmission",
    )
    verdict_rate = scenario_consistency_evaluator(q_rate_conflict)
    assert verdict_rate is not None, "Failed to detect 30 fps vs 1 second per frame rate contradiction"
    assert verdict_rate.reason_code == SCENARIO_CONTRADICTION

    repaired_rate = repair_scenario_contradiction(q_rate_conflict, verdict_rate.reason_code, verdict_rate.detail)
    assert repaired_rate is not None
    assert scenario_consistency_evaluator(repaired_rate) is None


# =============================================================================
# Test 12: Topic Source Sanitization & Sibling Overlap Containment
# =============================================================================

def test_stray_artifacts_and_topic_sanitization():
    """Verify topic sanitization, bullet stripping, and sibling overlap containment."""
    # Test planning agent topic cleaning
    raw_topic = "▪ Indefinite blocking, where low-priority processes"
    cleaned = PlanningAgent._clean_topic(raw_topic)
    assert cleaned == "Indefinite blocking"
    assert PlanningAgent._is_valid_topic(cleaned) is True

    # Test participle fragment rejection
    assert PlanningAgent._is_valid_topic("executed by multiple processors parallel") is False

    # Test sibling overlap containment (Q2a vs Q2b audit defect)
    s1 = "Describe the role of the OS in managing hardware resources efficiently."
    s2 = "Outline the methods by which the OS ensures efficient use of computer hardware."
    common, containment = _specific_overlap(s1, s2)
    assert containment >= 0.6, f"Containment should be >= 0.6 for duplicate asks: {containment}"
    assert len(common) >= 2, f"Common specific terms should be >= 2: {common}"

    # Test distinct questions have low containment
    s3 = "Explain the dual-mode operation and privileged instruction set in modern CPUs."
    _, containment_s3 = _specific_overlap(s1, s3)
    assert containment_s3 < 0.3


# =============================================================================
# Test 13: Sibling Independence Evaluator & Chaining Repair
# =============================================================================

def test_sibling_independence_evaluator_and_repair():
    """Verify sibling_independence_evaluator flags chained subquestions and repairs them."""
    # Defect from audit: Q8(b) explicitly chained via 'With reference to part (a)'
    q_chained_1 = AgentGeneratedQuestion(
        slot_id="slot_q8",
        module_id="module_4",
        global_q_idx=8,
        question_text="(a) Explain the role of the system call table. (b) With reference to part (a), explain how a user program invokes an interrupt.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        bloom="L2",
        topic="System Calls",
    )
    verdict_1 = sibling_independence_evaluator(q_chained_1)
    assert verdict_1 is not None, "Failed to flag 'With reference to part (a)' chaining"
    assert verdict_1.reason_code == SIBLING_INTERCONNECTED

    repaired_1 = repair_sibling_independence(q_chained_1, verdict_1.reason_code, verdict_1.detail)
    assert repaired_1 is not None
    assert "With reference to part (a)" not in repaired_1.question_text
    assert sibling_independence_evaluator(repaired_1) is None

    # Defect from audit: Q5(b) 'How does this compare...'
    q_chained_2 = AgentGeneratedQuestion(
        slot_id="slot_q5",
        module_id="module_3",
        global_q_idx=5,
        question_text="(a) Calculate the WAN transmission bandwidth required for uncompressed video. (b) How does this compare to typical consumer broadband speeds?",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        bloom="L3",
        topic="Network Transmission",
    )
    verdict_2 = sibling_independence_evaluator(q_chained_2)
    assert verdict_2 is not None, "Failed to flag 'How does this compare...' dependent comparison"
    assert verdict_2.reason_code == SIBLING_INTERCONNECTED

    # Defect: Subpart (b) starting with pronoun referring to (a)
    q_pronoun = AgentGeneratedQuestion(
        slot_id="slot_q3",
        module_id="module_2",
        global_q_idx=3,
        question_text="(a) Explain master-slave task assignment in asymmetric multiprocessing. (b) It is used to allocate CPU affinity dynamically.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        bloom="L2",
        topic="Multiprocessing",
    )
    verdict_3 = sibling_independence_evaluator(q_pronoun)
    assert verdict_3 is not None, "Failed to flag pronoun 'It' referring to previous part"
    assert verdict_3.reason_code == SIBLING_INTERCONNECTED

    # Valid independent multi-part question should pass cleanly
    q_independent = AgentGeneratedQuestion(
        slot_id="slot_q_ml",
        module_id="module_1",
        global_q_idx=1,
        question_text="(a) Explain the challenges of deploying machine learning models in production environments. (b) Apply the Find-S algorithm to determine the most specific hypothesis for the given training samples.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        bloom="L3",
        topic="Machine Learning",
    )
    assert sibling_independence_evaluator(q_independent) is None


# =============================================================================
# Standalone Runner
# =============================================================================

def run_all_tests() -> int:
    tests = [
        ("PaperSpec & Partition Allocation", test_paperspec_and_partition_allocation),
        ("Bloom Taxonomy & Opening Verbs", test_bloom_taxonomy_and_opening_verbs),
        ("Subquestion Diversity (Jaccard < 0.45)", test_subquestion_diversity_within_same_module),
        ("Multi-Part Prompt Instructions", test_multipart_prompt_instructions),
        ("Slot Filler & Module Grounding", test_slot_filler_module_grounding),
        ("Full VTU Paper Assembly (Q1..Q10)", test_full_vtu_paper_assembly),
        ("DOCX Exporter File Integrity", test_docx_export_generation),
        ("Fail-Closed Quality Gate", test_fail_closed_quality_gate),
        ("Topic Fragment Rejection & Repair", test_topic_fragment_evaluator_and_repair),
        ("Bloom Content Depth & Veneer Check", test_bloom_content_evaluator_and_depth),
        ("Scenario Premise Consistency Check", test_scenario_consistency_evaluator),
        ("Topic Sanitization & Overlap Containment", test_stray_artifacts_and_topic_sanitization),
        ("Sibling Independence & Unchaining Check", test_sibling_independence_evaluator_and_repair),
    ]

    print("\n" + "=" * 78)
    print("  AION MASTER MODEL TEST SUITE (CURRENT BUILD PARAMETERS)")
    print("=" * 78)

    passed_count = 0
    failed_count = 0
    total_start = time.time()

    for idx, (name, fn) in enumerate(tests, 1):
        t0 = time.time()
        sys.stdout.write(f"[{idx}/{len(tests)}] {name:<46} ... ")
        sys.stdout.flush()
        try:
            fn()
            dur = time.time() - t0
            sys.stdout.write(f"\033[92mPASS\033[0m ({dur * 1000.0:.1f}ms)\n")
            passed_count += 1
        except Exception as exc:
            dur = time.time() - t0
            sys.stdout.write(f"\033[91mFAIL\033[0m ({dur * 1000.0:.1f}ms)\n")
            print(f"      [ERROR] {exc}")
            failed_count += 1

    total_dur = time.time() - total_start
    print("-" * 78)
    print(f"Total: {len(tests)} | Passed: \033[92m{passed_count}\033[0m | Failed: \033[91m{failed_count}\033[0m | Time: {total_dur:.2f}s")
    print("=" * 78 + "\n")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
