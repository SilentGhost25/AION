"""
Real-Document Corpus Test: Corrupted Document Handling
"""

import pytest
from v0_1.contracts import ContractViolation
from v0_1.unified_pipeline import run_unified
from v0_1.question_completeness import QuestionCompletenessValidator


def test_corrupted_zero_text_rejection(tmp_path):
    empty_file = tmp_path / "corrupted_empty.txt"
    empty_file.write_text("   \n\t ", encoding="utf-8")

    with pytest.raises((ContractViolation, RuntimeError)):
        run_unified(
            file_path=str(empty_file),
            exam_type="IA",
            difficulty="Mixed",
            subject="Data Structures",
            max_questions=4,
        )


def test_truncated_question_validator_rejection():
    bad_questions = [
        "Calculate the equivalent resistance when R1 = 10Ω and R2 =",
        "Explain the process of Dijkstra's algorithm and",
        "Determine the balance factor of node",
        "Short text",
    ]

    for q in bad_questions:
        valid, errors = QuestionCompletenessValidator.validate(q)
        assert valid is False, f"Truncated question '{q}' passed validation unexpectedly!"
        assert len(errors) > 0
