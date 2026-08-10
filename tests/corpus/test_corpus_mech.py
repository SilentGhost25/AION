"""
Real-Document Corpus Test: Mechanical Engineering
"""

import pytest
from v0_1.unified_pipeline import run_unified
from v0_1.question_completeness import QuestionCompletenessValidator


@pytest.fixture
def mech_corpus_file(tmp_path) -> str:
    f = tmp_path / "mech_thermodynamics.txt"
    f.write_text(
        "MODULE 1: Basic Thermodynamics Concepts. First Law of Thermodynamics energy conservation for closed and open systems. "
        "Work transfer, heat transfer, and internal energy non-flow energy equation Q - W = delta U. "
        "MODULE 2: Second Law of Thermodynamics. Carnot cycle thermal efficiency, Clausius inequality statement, and entropy calculation. "
        "MODULE 3: Thermal Power Cycles. Otto, Diesel, Dual, and Rankine thermal power generation cycles.",
        encoding="utf-8",
    )
    return str(f)


def test_mech_corpus_pipeline_run(mech_corpus_file):
    paper = run_unified(
        file_path=mech_corpus_file,
        exam_type="IA",
        difficulty="Mixed",
        subject="Thermodynamics",
        max_questions=4,
    )

    assert paper is not None
    assert paper.health.score > 0

    for module in paper.modules:
        for q in module.get("questions", []):
            for sub in q.get("subQuestions", []):
                valid, errors = QuestionCompletenessValidator.validate(sub["text"])
                assert valid is True, f"Mech question incomplete: {errors}"
