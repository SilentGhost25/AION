"""
Real-Document Corpus Test: Civil Engineering
"""

import pytest
from v0_1.unified_pipeline import run_unified
from v0_1.question_completeness import QuestionCompletenessValidator


@pytest.fixture
def civil_corpus_file(tmp_path) -> str:
    f = tmp_path / "civil_structural_analysis.txt"
    f.write_text(
        "MODULE 1: Beams and Support Reactions. Simply supported beams subjected to concentrated point loads and uniformly distributed loads (UDL). "
        "Support reactions Ra and Rb calculated using static equilibrium equations sum Fy = 0 and sum M = 0. "
        "MODULE 2: Bending Moment and Shear Force Diagrams. Shear Force Diagram (SFD) and Bending Moment Diagram (BMD) along beam length under loading conditions. "
        "Maximum bending moment occurs at point of zero shear force in semplicemente supported beams. "
        "MODULE 3: Trusses and Frames. Method of joints and method of sections for statically determinate truss structures under external joint loads.",
        encoding="utf-8",
    )
    return str(f)


def test_civil_corpus_pipeline_run(civil_corpus_file):
    paper = run_unified(
        file_path=civil_corpus_file,
        exam_type="IA",
        difficulty="Mixed",
        subject="Structural Analysis",
        max_questions=4,
    )

    assert paper is not None
    assert paper.health.score > 0

    for module in paper.modules:
        for q in module.get("questions", []):
            for sub in q.get("subQuestions", []):
                valid, errors = QuestionCompletenessValidator.validate(sub["text"])
                assert valid is True, f"Civil question incomplete: {errors}"
