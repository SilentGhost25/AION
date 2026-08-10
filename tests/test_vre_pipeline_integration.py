"""
End-to-end integration test verifying full pipeline execution:
Unified Pipeline -> VRE Engine -> Solvers -> QuestionPlan -> Visual Critic -> FinalPaper
"""

import pytest
from PIL import Image
from v0_1.unified_pipeline import run_unified
from v0_1.vre import PaperVisualValidator


@pytest.fixture
def sample_text_file(tmp_path) -> str:
    f = tmp_path / "sample_academic_text.txt"
    f.write_text(
        "Data Structures and Algorithms analysis. Dijkstra's algorithm finds the shortest path "
        "in a weighted graph with non-negative edge weights between source vertex A and destination vertex D. "
        "AVL tree rotations maintain the balance factor of every node between -1 and +1 efficiently. "
        "Resistive circuit analysis using Kirchhoff's Voltage Law (KVL) calculates the total voltage drop "
        "and equivalent resistance across all resistors in series and parallel loops. "
        "Simply supported beams under point loading undergo shear force and bending moment distributions along their span.",
        encoding="utf-8",
    )
    return str(f)


def test_unified_pipeline_end_to_end_with_vre(sample_text_file):
    paper = run_unified(
        file_path=sample_text_file,
        exam_type="IA",
        difficulty="Mixed",
        subject="Data Structures",
        max_questions=4,
    )

    assert paper is not None
    assert paper.doc_id is not None
    assert paper.health.score > 0
    assert len(paper.modules) > 0

    paper_dict = {
        "doc_id": paper.doc_id,
        "modules": paper.modules,
        "exam_type": paper.exam_type,
        "subject": paper.subject,
    }

    # Validate paper level visual QA rules
    valid, errors = PaperVisualValidator.validate_paper(paper_dict)
    assert valid is True
    assert len(errors) == 0
