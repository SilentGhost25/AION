"""
AION Evidence Validator & Tripartite Health Score Test Suite
============================================================
"""

import pytest
from core.validators.evidence_validator import EvidenceValidator, EvidenceValidationResult
from v0_1.module_alignment import ModuleAlignmentValidator, ModuleAlignmentResult
from v0_1.tripartite_health import TripartiteHealthScore
from v0_1.unified_pipeline import run_unified


def test_evidence_validator_grounding():
    question = "Explain Dijkstra's shortest path algorithm for weighted graphs."
    retrieved_chunks = [
        {"chunk_id": "chk_001", "text": "Dijkstra's algorithm finds single-source shortest paths in weighted graphs.", "page": 12},
    ]

    res = EvidenceValidator.validate(question, retrieved_chunks, target_module=3, target_bloom="L2")
    assert res.passed is True
    assert res.support_score >= 0.70
    assert len(res.evidence_refs) > 0
    assert res.evidence_refs[0]["chunk_id"] == "chk_001"


def test_module_alignment_validator():
    q_mod1 = "Explain linear stack array operations using push and pop."
    res_mod1 = ModuleAlignmentValidator.validate(q_mod1, target_module=1)
    assert res_mod1.passed is True
    assert "stack" in res_mod1.detected_concepts or "array" in res_mod1.detected_concepts


def test_tripartite_health_score_calculation():
    t_health = TripartiteHealthScore(
        structural_score=100,
        grounding_score=95,
        academic_score=90,
    )
    assert t_health.overall_health == 95


def test_pipeline_50_mark_completeness(tmp_path):
    f = tmp_path / "complete_textbook_sample.txt"
    f.write_text(
        "MODULE 1: Linear Data Structures. Arrays, Stacks, and Queues operate on LIFO and FIFO principles in data management. "
        "MODULE 2: Binary Search Trees. AVL trees maintain balance factor through LL, RR, LR, and RL rotations. "
        "MODULE 3: Graph Algorithms. Dijkstra's shortest path algorithm for weighted non-negative graphs. "
        "MODULE 4: Sorting Algorithms. Quick sort partitioning and merge sort recursive splitting. "
        "MODULE 5: Hashing and File Structures. Separate chaining and open addressing probing for collisions.",
        encoding="utf-8",
    )

    paper = run_unified(
        file_path=str(f),
        exam_type="IA",
        difficulty="Mixed",
        subject="Data Structures",
        max_questions=5,
    )

    assert paper is not None
    assert paper.total_marks == 50
    assert len(paper.modules) > 0
