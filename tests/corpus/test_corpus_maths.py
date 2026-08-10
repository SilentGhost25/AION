"""
Real-Document Corpus Test: Engineering Mathematics
"""

import pytest
from v0_1.unified_pipeline import run_unified
from v0_1.question_completeness import QuestionCompletenessValidator


@pytest.fixture
def maths_corpus_file(tmp_path) -> str:
    f = tmp_path / "maths_linear_algebra.txt"
    f.write_text(
        "MODULE 1: Linear Algebra and Matrix Theory. Matrices, matrix rank, system of linear equations Ax = b, and Gauss elimination methods for solving simultaneous equations. "
        "MODULE 2: Eigenvalues and Eigenvectors. Characteristic polynomial equation det(A - lambda I) = 0 and Cayley-Hamilton theorem applications in finding matrix inverses. "
        "MODULE 3: Vector Calculus. Gradient of scalar fields, divergence and curl of vector fields, Green's theorem, and Stokes' theorem in line and surface vector integration.",
        encoding="utf-8",
    )
    return str(f)


def test_maths_corpus_pipeline_run(maths_corpus_file):
    paper = run_unified(
        file_path=maths_corpus_file,
        exam_type="IA",
        difficulty="Mixed",
        subject="Engineering Mathematics",
        max_questions=4,
    )

    assert paper is not None
    assert paper.health.score > 0

    for module in paper.modules:
        for q in module.get("questions", []):
            for sub in q.get("subQuestions", []):
                valid, errors = QuestionCompletenessValidator.validate(sub["text"])
                assert valid is True, f"Maths question incomplete: {errors}"
