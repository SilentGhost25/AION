"""
Deterministic repairs for flagged questions.

Each repair function:
    - Takes (question, reason_code, detail)
    - Returns a repaired GeneratedQuestion, or None if it can't help.

Design notes:
    - These repairs are conservative. Any repair that could make the
      question worse than the original returns None.
    - Every repair is pure: no side effects, no state, no LLM calls.
    - Repairs are attempted in the order listed in REPAIR_PIPELINE.
      The first successful repair wins.
"""

from __future__ import annotations

import copy
import re
from typing import Optional

from .evaluation_contracts import (
    BLOOM_MISMATCH,
    MARKING_SCHEME_MISMATCH,
    MISSING_FIGURE_REFERENCE,
    MULTI_PART_STRUCTURE_MISMATCH,
    QUESTION_TOO_SHORT,
    SOLUTION_TOO_SHORT,
    UNVERIFIED,
)
from .generated_question import GeneratedQuestion
from .prompts import BLOOM_VERBS


# -----------------------------------------------------------------------------
# Repairs
# -----------------------------------------------------------------------------


# Words that commonly precede a Bloom verb as a preamble clause. If the
# question starts with one of these, we strip the preamble to find the verb.
_PREAMBLE_PATTERN = re.compile(
    r"^(?:In\s+[^,]{2,40},\s*|Regarding\s+[^,]{2,40},\s*|"
    r"Concerning\s+[^,]{2,40},\s*|Given\s+[^,]{2,40},\s*)",
    re.IGNORECASE,
)


def _first_alpha_word(text: str) -> str:
    m = re.search(r"[A-Za-z]+", text)
    return m.group(0).lower() if m else ""


def repair_bloom_verb(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """
    Fix BLOOM_MISMATCH by stripping leading preambles and, if needed,
    prepending the correct Bloom verb.
    """
    if reason_code != BLOOM_MISMATCH:
        return None

    text = (question.question_text or "").strip()
    if not text:
        return None

    # Attempt 1: strip a preamble clause if present
    stripped = _PREAMBLE_PATTERN.sub("", text, count=1)
    if stripped != text:
        candidate_first = _first_alpha_word(stripped)
        allowed = BLOOM_VERBS.get(question.bloom, [])
        if candidate_first in allowed:
            repaired = _replace_text(question, stripped)
            return repaired

    # Attempt 2: prepend the appropriate Bloom verb
    allowed = BLOOM_VERBS.get(question.bloom, [])
    if not allowed:
        return None
    verb = allowed[0]
    # Only prepend if the current first word isn't already a Bloom verb
    current_first = _first_alpha_word(text)
    if current_first in allowed:
        return None  # already correct — shouldn't reach here

    # Lowercase the existing first letter to make a natural sentence
    prepend = f"{verb.capitalize()} the following: {text[0].lower() + text[1:]}"
    return _replace_text(question, prepend)


def repair_marking_scheme(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """
    Fix MARKING_SCHEME_MISMATCH conservatively:
        - Single-entry scheme: adjust to target.
        - Multi-entry with small delta: distribute the difference across entries.
        - Large delta: give up (return None).
    """
    if reason_code != MARKING_SCHEME_MISMATCH:
        return None

    scheme = list(question.marking_scheme or [])
    if not scheme:
        return None

    current = sum(
        e.get("marks", 0) for e in scheme
        if isinstance(e, dict) and isinstance(e.get("marks"), (int, float))
    )
    target = question.marks
    delta = target - current

    if abs(delta) < 1e-9:
        return None  # no change needed

    # Single entry: trivial fix
    if len(scheme) == 1:
        new_scheme = [dict(scheme[0], marks=target)]
        return _replace_marking_scheme(question, new_scheme)

    # Multi-entry: only fix if the delta is small relative to current
    if current > 0 and abs(delta) / current > 0.2:
        return None  # too risky to redistribute

    # Distribute delta proportionally
    new_scheme = []
    remaining = target
    for i, entry in enumerate(scheme):
        if not isinstance(entry, dict):
            return None
        if i == len(scheme) - 1:
            new_scheme.append(dict(entry, marks=remaining))
        else:
            old_marks = entry.get("marks", 0)
            share = round(old_marks / current * target) if current else 0
            new_scheme.append(dict(entry, marks=share))
            remaining -= share

    if remaining != target - sum(e["marks"] for e in new_scheme[:-1]):
        return None  # arithmetic drift; bail out

    return _replace_marking_scheme(question, new_scheme)


def repair_figure_reference(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """
    Fix MISSING_FIGURE_REFERENCE by prepending a natural figure-reference
    phrase when an image is attached but not referenced.
    """
    if reason_code != MISSING_FIGURE_REFERENCE:
        return None

    has_image = bool(question.image_path)
    references = bool(question.references_image)

    if not has_image:
        # Question references a figure we don't have — can't fix.
        return None

    if references:
        return None  # already references; nothing to do

    text = (question.question_text or "").strip()
    if not text:
        return None

    # Prepend the reference phrase and lowercase the original first letter.
    prefix = "With the aid of the given figure, "
    new_text = prefix + text[0].lower() + text[1:]

    repaired = _replace_text(question, new_text)
    repaired.references_image = True
    return repaired


def repair_short_question(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """
    Fix QUESTION_TOO_SHORT by appending a clarifying clause using the
    question's topic (from the plan).
    """
    if reason_code != QUESTION_TOO_SHORT:
        return None

    text = (question.question_text or "").strip()
    if len(text) >= 20:
        return None

    topic = (question.topic or "").strip()
    if not topic:
        return None

    # Append a topic-specific clarifier.
    # Only extend if the extension is semantically safe (no new claim).
    clarifier = f" Focus your answer on {topic}."
    new_text = text.rstrip(".") + "." + clarifier
    if len(new_text) < 20:
        return None
    return _replace_text(question, new_text)


def repair_short_solution(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """Do not attempt. Solution content should be regenerated, not padded."""
    return None


def repair_multi_part_structure(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """
    Multi-part repairs are hard to do safely. If the partition is [6, 4]
    and the text has no sub-parts, inserting them post-hoc risks
    semantically splitting a coherent question. Return None → LLM repair.
    """
    return None


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


# Ordered list of deterministic repair functions.
REPAIR_PIPELINE = [
    repair_bloom_verb,
    repair_figure_reference,
    repair_marking_scheme,
    repair_short_question,
    repair_short_solution,
    repair_multi_part_structure,
]


def try_deterministic_repair(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """
    Try each repair in order. Return the first successful repair,
    or None if no deterministic repair applies.
    """
    for repair in REPAIR_PIPELINE:
        try:
            result = repair(question, reason_code, detail)
        except Exception:
            result = None
        if result is not None:
            return result
    return None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _replace_text(
    question: GeneratedQuestion, new_text: str
) -> GeneratedQuestion:
    """Return a copy with the question_text replaced. Immutable discipline."""
    q = copy.deepcopy(question)
    q.question_text = new_text
    # Preserve references_image flag — caller may adjust after
    return q


def _replace_marking_scheme(
    question: GeneratedQuestion, new_scheme: list
) -> GeneratedQuestion:
    q = copy.deepcopy(question)
    q.marking_scheme = new_scheme
    return q
