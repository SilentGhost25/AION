"""
AION v2.1 PC Production Quality Evaluation & Failure Taxonomy Test
===================================================================
Executes quality evaluation report generation and classifies failures via FailureClassifier.
"""

import pytest
from v0_1.evaluator import ProductionQualityEvaluator, QualityEvaluationReport
from v0_1.failure_taxonomy import FailureCategory, FailureClassifier, FailureRecord
from v0_1.unified_pipeline import run_unified


@pytest.fixture
def eval_paper_file(tmp_path) -> str:
    f = tmp_path / "eval_sample_paper.txt"
    f.write_text(
        "MODULE 1: Linear Data Structures. Stacks and Queues operate on LIFO and FIFO principles. Array based stack implementations. "
        "MODULE 2: Binary Search Trees. AVL trees maintain balance factor through LL, RR, LR, and RL rotations for self balancing efficiency. "
        "MODULE 3: Graph Algorithms. Dijkstra's algorithm finds single-source shortest paths in weighted graphs with non-negative edge weights.",
        encoding="utf-8",
    )
    return str(f)


def test_quality_evaluator_report(eval_paper_file):
    paper = run_unified(
        file_path=eval_paper_file,
        exam_type="IA",
        difficulty="Mixed",
        subject="Data Structures",
        max_questions=4,
    )

    report = ProductionQualityEvaluator.evaluate_paper(paper)
    summary_str = report.format_summary()

    assert report.documents_tested == 1
    assert report.questions_generated > 0
    assert report.truncation_rate == 0.0
    assert report.duplicate_rate <= 100.0
    assert "AION v2.1 PC PRODUCTION QUALITY REPORT" in summary_str


def test_failure_taxonomy_classification():
    record_ext = FailureClassifier.classify("S1_EXTRACTION failed to read_text from PDF", stage="S1")
    assert record_ext.category == FailureCategory.EXTRACTION

    record_vre = FailureClassifier.classify("VKO validation failed for circuit diagram", stage="S6_VRE")
    assert record_vre.category == FailureCategory.VRE

    record_solver = FailureClassifier.classify("GraphSolver encountered unsolvable Dijkstra path", stage="S6_SOLVER")
    assert record_solver.category == FailureCategory.SOLVER

    record_structure = FailureClassifier.classify("Mark total mismatch: expected 10, got 15", stage="S8_ASSEMBLE")
    assert record_structure.category == FailureCategory.STRUCTURE
