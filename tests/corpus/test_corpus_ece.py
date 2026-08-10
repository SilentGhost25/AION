"""
Real-Document Corpus Test: Electronics & Communication (ECE)
"""

import pytest
from v0_1.unified_pipeline import run_unified
from v0_1.question_completeness import QuestionCompletenessValidator


@pytest.fixture
def ece_corpus_file(tmp_path) -> str:
    f = tmp_path / "ece_network_analysis.txt"
    f.write_text(
        "MODULE 1: Network Theorems. Kirchhoff's Voltage Law (KVL) states that algebraic sum of voltages in a closed loop is zero. "
        "Kirchhoff's Current Law (KCL) states sum of currents entering a node equals sum leaving. "
        "MODULE 2: Equivalent Circuits. Thevenin's theorem reduces complex linear networks to a single voltage source and series resistor. "
        "Norton's theorem reduces networks to an equivalent current source in parallel with equivalent resistance. "
        "MODULE 3: AC Circuit Analysis. Sinusoidal steady-state response, phasor diagrams, and complex power in RLC circuits.",
        encoding="utf-8",
    )
    return str(f)


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
