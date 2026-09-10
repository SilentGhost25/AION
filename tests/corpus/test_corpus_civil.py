"""
Real-Document Corpus Test: Civil Engineering
"""

import pytest
from v0_1.unified_pipeline import run_unified
from v0_1.question_completeness import QuestionCompletenessValidator


@pytest.fixture
def civil_corpus_file(tmp_path) -> str:
    import fitz
    pdf_path = tmp_path / "civil_structural_analysis.pdf"
    content = (
        "MODULE 1: Beams and Support Reactions\n\n"
        "Simply supported beams subjected to concentrated point loads and uniformly distributed loads (UDL). "
        "Support reactions Ra and Rb calculated using static equilibrium equations sum Fy = 0 and sum M = 0.\n\n"
        "MODULE 2: Bending Moment and Shear Force Diagrams\n\n"
        "Shear Force Diagram (SFD) and Bending Moment Diagram (BMD) along beam length under loading conditions. "
        "Maximum bending moment occurs at point of zero shear force in simply supported beams.\n\n"
        "MODULE 3: Trusses and Frames\n\n"
        "Method of joints and method of sections for statically determinate truss structures under external joint loads."
    )
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), content, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


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
