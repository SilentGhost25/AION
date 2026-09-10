"""
Real-Document Corpus Test: Electrical & Electronics (EEE)
"""

import pytest
from v0_1.unified_pipeline import run_unified
from v0_1.question_completeness import QuestionCompletenessValidator


@pytest.fixture
def eee_corpus_file(tmp_path) -> str:
    import fitz
    pdf_path = tmp_path / "eee_electrical_machines.pdf"
    content = (
        "MODULE 1: DC Machines and Operating Principles\n\n"
        "DC Generators convert mechanical energy to electrical energy using Faraday's Law of Induction. "
        "EMF equation of DC Generator is E = (P * Phi * Z * N) / (60 * A). "
        "Armature reaction causes demagnetizing and cross-magnetizing effects on field flux.\n\n"
        "MODULE 2: Transformers and Voltage Regulation\n\n"
        "Single-phase transformers operate on mutual induction. Efficiency and voltage regulation under load conditions. "
        "Equivalent circuit parameters determine copper and core losses.\n\n"
        "MODULE 3: Induction Motors and Slip Characteristics\n\n"
        "Three-phase induction motor rotating magnetic field, synchronous speed, slip calculation, and torque-speed characteristics. "
        "Starting methods include star-delta and autotransformer starting."
    )
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), content, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


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
