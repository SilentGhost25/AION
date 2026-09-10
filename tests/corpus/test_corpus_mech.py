"""
Real-Document Corpus Test: Mechanical Engineering
"""

import pytest
from v0_1.unified_pipeline import run_unified
from v0_1.question_completeness import QuestionCompletenessValidator


@pytest.fixture
def mech_corpus_file(tmp_path) -> str:
    import fitz
    pdf_path = tmp_path / "mech_thermodynamics.pdf"
    content = (
        "MODULE 1: Basic Thermodynamics Concepts\n\n"
        "First Law of Thermodynamics energy conservation for closed and open systems. "
        "Work transfer, heat transfer, and internal energy non-flow energy equation Q - W = delta U. "
        "Enthalpy H = U + PV. Steady flow energy equation applies to nozzles, turbines, and compressors.\n\n"
        "MODULE 2: Second Law of Thermodynamics\n\n"
        "Carnot cycle thermal efficiency, Clausius inequality statement, and entropy calculation. "
        "Second law statements of Kelvin-Planck and Clausius. Reversible and irreversible processes. "
        "Entropy change in ideal gases and temperature-entropy T-s diagrams.\n\n"
        "MODULE 3: Thermal Power Cycles\n\n"
        "Otto, Diesel, Dual, and Rankine thermal power generation cycles. "
        "Air standard efficiency calculations, compression ratio influence, and mean effective pressure."
    )
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), content, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


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
