"""
Real-Document Corpus Test: Electronics & Communication (ECE)
"""

import pytest
from v0_1.unified_pipeline import run_unified
from v0_1.question_completeness import QuestionCompletenessValidator


@pytest.fixture
def ece_corpus_file(tmp_path) -> str:
    import fitz
    pdf_path = tmp_path / "ece_network_analysis.pdf"
    content = (
        "MODULE 1: Network Theorems\n\n"
        "Kirchhoff's Voltage Law (KVL) states that algebraic sum of voltages in a closed loop is zero. "
        "Kirchhoff's Current Law (KCL) states sum of currents entering a node equals sum leaving.\n\n"
        "MODULE 2: Equivalent Circuits\n\n"
        "Thevenin's theorem reduces complex linear networks to a single voltage source and series resistor. "
        "Norton's theorem reduces networks to an equivalent current source in parallel with equivalent resistance.\n\n"
        "MODULE 3: AC Circuit Analysis\n\n"
        "Sinusoidal steady-state response, phasor diagrams, and complex power in RLC circuits."
    )
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), content, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def test_ece_corpus_pipeline_run(ece_corpus_file):
    paper = run_unified(
        file_path=ece_corpus_file,
        exam_type="IA",
        difficulty="Mixed",
        subject="Network Analysis",
        max_questions=4,
    )

    assert paper is not None
    assert paper.exportable is True

    for module in paper.modules:
        for q in module.get("questions", []):
            for sub in q.get("subQuestions", []):
                valid, errors = QuestionCompletenessValidator.validate(sub["text"])
                assert valid is True, f"ECE question incomplete: {errors}"
