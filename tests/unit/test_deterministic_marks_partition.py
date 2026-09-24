"""
Unit tests for deterministic sequential marks partition allocation across subjects.
Validates parameterized PaperSpec configurations, positive integer checks,
shape and sum constraints, and sequential index assignment.
"""

import pytest
from core.contracts.paper_spec import PaperSpec


@pytest.mark.parametrize("spec, split", [
    # IAT1: 5 modules × 2 questions × 10 marks = 10 questions
    (
        PaperSpec("IAT1", 5, 2, 10, 5),
        [[10], [10], [6, 4], [6, 4], [6, 4], [6, 4], [10], [10], [10], [10]],
    ),
    # SEE: 5 modules × 4 questions × 25 marks = 20 questions
    (
        PaperSpec("SEE_4Q", 5, 4, 25, 5),
        [[25]] * 20,
    ),
    # Elective: 3 modules × 3 questions × 10 marks = 9 questions
    (
        PaperSpec("ELECTIVE_3MOD", 3, 3, 10, 3),
        [[10]] * 9,
    ),
    # Mixed-marks paper: 4 modules × 2 questions × 10 marks = 8 questions
    (
        PaperSpec("MIXED", 4, 2, 10, 4),
        [[6, 4], [6, 4], [5, 5], [10], [10], [8, 2], [6, 4], [3, 3, 4]],
    ),
])
def test_allocate_partitions_generalized(spec, split):
    """Verify sequential partition mapping preserves order exactly for any subject."""
    assignments = spec.allocate_partitions(split)
    assert len(assignments) == spec.total_questions
    assert list(assignments.values()) == split

    # Check that (module 0, question 0) maps to index 0, etc.
    cumulative = 0
    for m in range(spec.module_count):
        for q in range(spec.questions_per_module):
            assert assignments[(m, q)] == split[cumulative]
            cumulative += 1


def test_allocate_partitions_length_mismatch():
    """Fail early if user partitions count != total planned questions."""
    spec = PaperSpec("IAT1", 5, 2, 10, 5)
    # Only 8 partitions provided for 10 planned questions
    invalid_split = [[10], [10], [6, 4], [6, 4], [6, 4], [6, 4], [10], [10]]
    with pytest.raises(ValueError, match="Expected 10 partitions"):
        spec.allocate_partitions(invalid_split)


def test_allocate_partitions_non_positive_integers():
    """Fail early if partition elements are zero, negative, or non-integer."""
    spec = PaperSpec("IAT1", 1, 1, 10, 1)

    with pytest.raises(ValueError, match="invalid element -5"):
        spec.allocate_partitions([[15, -5]])

    with pytest.raises(ValueError, match="invalid element 0"):
        spec.allocate_partitions([[10, 0]])


def test_allocate_partitions_sum_mismatch():
    """Fail early if partition does not sum to marks_per_question."""
    spec = PaperSpec("IAT1", 1, 1, 10, 1)
    with pytest.raises(ValueError, match="sums to 12, expected 10"):
        spec.allocate_partitions([[6, 6]])
