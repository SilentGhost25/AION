"""
Deterministic local evaluators.

Each evaluator takes a GeneratedQuestion and returns:
    - None if the question passes
    - SlotVerdict (with verdict="fail") if the question fails

Composability:
    LOCAL_EVALUATORS is a list of evaluators run in order.
    The first failure short-circuits further local evaluation for
    that slot, but other slots continue.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

from .evaluation_contracts import (
    BLOOM_MISMATCH,
    MARKING_SCHEME_MISMATCH,
    MISSING_FIGURE_REFERENCE,
    MULTI_PART_STRUCTURE_MISMATCH,
    QUESTION_TOO_LONG,
    QUESTION_TOO_SHORT,
    SOLUTION_MISSING,
    SOLUTION_TOO_SHORT,
    SlotVerdict,
)
from .prompts import BLOOM_VERBS


# -----------------------------------------------------------------------------
# Thresholds
# -----------------------------------------------------------------------------


MIN_QUESTION_CHARS = 20
MAX_QUESTION_CHARS = 1200
MIN_SOLUTION_CHARS = 20
MARKING_SUM_TOLERANCE = 0.5  # allow small rounding drift


# -----------------------------------------------------------------------------
# Evaluator type
# -----------------------------------------------------------------------------


LocalEvaluator = Callable[[object], Optional[SlotVerdict]]
# object is a GeneratedQuestion; typed loosely to avoid circular imports.


def _fail(question, code: str, detail: str, fix: Optional[str] = None) -> SlotVerdict:
    return SlotVerdict(
        slot_id=question.slot_id,
        verdict="fail",
        source="local",
        reason_code=code,
        detail=detail,
        suggested_fix=fix,
    )


# -----------------------------------------------------------------------------
# Individual evaluators
# -----------------------------------------------------------------------------


def bloom_verb_evaluator(question) -> Optional[SlotVerdict]:
    """First word must be an approved Bloom verb for the slot's Bloom level."""
    first = question.first_word
    if not first:
        return _fail(
            question, BLOOM_MISMATCH,
            "question text is empty",
        )
    allowed = BLOOM_VERBS.get(question.bloom, [])
    if first not in allowed:
        return _fail(
            question, BLOOM_MISMATCH,
            f"first word {first!r} not in approved verbs for {question.bloom}: {allowed}",
            fix=f"Start the question with one of: {', '.join(allowed)}",
        )
    return None


def marking_scheme_evaluator(question) -> Optional[SlotVerdict]:
    """Sum of marking scheme marks must equal the question's total marks."""
    if not question.marking_scheme:
        return _fail(
            question, MARKING_SCHEME_MISMATCH,
            "marking scheme is empty",
            fix="Provide a marking scheme whose entries sum to the total marks",
        )
    total = 0
    for entry in question.marking_scheme:
        if not isinstance(entry, dict):
            return _fail(
                question, MARKING_SCHEME_MISMATCH,
                f"marking scheme entry is not a dict: {entry!r}",
            )
        marks = entry.get("marks", 0)
        if not isinstance(marks, (int, float)):
            return _fail(
                question, MARKING_SCHEME_MISMATCH,
                f"marking scheme entry 'marks' not numeric: {marks!r}",
            )
        total += marks
    if abs(total - question.marks) > MARKING_SUM_TOLERANCE:
        return _fail(
            question, MARKING_SCHEME_MISMATCH,
            f"marking scheme sums to {total}, expected {question.marks}",
            fix=f"Adjust marking scheme to sum to {question.marks}",
        )
    return None


def figure_consistency_evaluator(question) -> Optional[SlotVerdict]:
    """If an image is attached, the question must reference it."""
    has_image = bool(question.image_path)
    references = bool(question.references_image)
    if has_image and not references:
        return _fail(
            question, MISSING_FIGURE_REFERENCE,
            "figure attached but question text does not reference it",
            fix="Add 'With the aid of the given figure...' or similar phrasing",
        )
    if references and not has_image:
        # Question claims a figure exists but none is attached. Also a failure.
        return _fail(
            question, MISSING_FIGURE_REFERENCE,
            "question references a figure but no image is attached",
            fix="Either remove the figure reference or attach the figure",
        )
    return None


def question_length_evaluator(question) -> Optional[SlotVerdict]:
    n = len(question.question_text.strip())
    if n < MIN_QUESTION_CHARS:
        return _fail(
            question, QUESTION_TOO_SHORT,
            f"question is {n} chars, minimum {MIN_QUESTION_CHARS}",
            fix="Expand the question to be self-contained",
        )
    if n > MAX_QUESTION_CHARS:
        return _fail(
            question, QUESTION_TOO_LONG,
            f"question is {n} chars, maximum {MAX_QUESTION_CHARS}",
            fix="Shorten the question text",
        )
    return None


def solution_evaluator(question) -> Optional[SlotVerdict]:
    solution = (question.solution or "").strip()
    if not solution:
        return _fail(
            question, SOLUTION_MISSING,
            "solution is empty",
            fix="Provide a complete model answer",
        )
    if len(solution) < MIN_SOLUTION_CHARS:
        return _fail(
            question, SOLUTION_TOO_SHORT,
            f"solution is {len(solution)} chars, minimum {MIN_SOLUTION_CHARS}",
            fix="Expand the model answer",
        )
    return None


# Multi-part label patterns
_PART_LABEL = re.compile(r"\(?([ivx]+|[a-z]|\d+)\)\s", re.IGNORECASE)


def multi_part_structure_evaluator(question) -> Optional[SlotVerdict]:
    """
    If the question is multi-part (partition has >1 elements), the text
    should contain sub-part labels. If single-part, it should not.
    """
    is_multi = question.is_multi_part
    has_labels = len(_PART_LABEL.findall(question.question_text)) >= 2
    if is_multi and not has_labels:
        return _fail(
            question, MULTI_PART_STRUCTURE_MISMATCH,
            f"partition {question.partition} implies sub-parts but none found in text",
            fix="Structure the question with explicit (i)/(ii) or (a)/(b) sub-parts",
        )
    if not is_multi and has_labels:
        return _fail(
            question, MULTI_PART_STRUCTURE_MISMATCH,
            f"single-part slot but text contains sub-part labels",
            fix="Remove sub-part labels",
        )
    return None


# -----------------------------------------------------------------------------
# Aggregated list
# -----------------------------------------------------------------------------


LOCAL_EVALUATORS: List[LocalEvaluator] = [
    bloom_verb_evaluator,
    marking_scheme_evaluator,
    figure_consistency_evaluator,
    question_length_evaluator,
    solution_evaluator,
    multi_part_structure_evaluator,
]


def run_local_evaluators(question) -> Optional[SlotVerdict]:
    """
    Run all local evaluators in order. Return the first failure, or None.
    """
    for evaluator in LOCAL_EVALUATORS:
        verdict = evaluator(question)
        if verdict is not None:
            return verdict
    return None
