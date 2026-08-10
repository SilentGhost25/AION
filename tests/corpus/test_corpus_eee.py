"""
Real-Document Corpus Test: Electrical & Electronics (EEE)
"""

import pytest
from v0_1.unified_pipeline import run_unified
from v0_1.question_completeness import QuestionCompletenessValidator


@pytest.fixture
def eee_corpus_file(tmp_path) -> str:
    f = tmp_path / "eee_electrical_machines.txt"
    f.write_text(
        "MODULE 1: DC Machines. DC Generators convert mechanical energy to electrical energy using Faraday's Law of Induction. "
        "EMF equation of DC Generator is E = (P * Phi * Z * N) / (60 * A). "
        "MODULE 2: Transformers. Single-phase transformers operate on mutual induction. Efficiency and voltage regulation under load. "
        "MODULE 3: Induction Motors. Three-phase induction motor rotating magnetic field, slip calculation, and torque-speed characteristics.",
        encoding="utf-8",
    )
    return str(f)


def test_eee_corpus_pipeline_run(eee_corpus_file):
    paper = run_unified(
        file_path=eee_corpus_file,
        exam_type="IA",
        difficulty="Mixed",
        subject="Electrical Machines",
        max_questions=4,
    )

    assert paper is not None
    assert paper.exportable is True

    for module in paper.modules:
        for q in module.get("questions", []):
            for sub in q.get("subQuestions", []):
                valid, errors = QuestionCompletenessValidator.validate(sub["text"])
                assert valid is True, f"EEE question incomplete: {errors}"
