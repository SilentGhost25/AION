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
    TOPIC_FRAGMENT,
    BLOOM_DEPTH_MISMATCH,
    SCENARIO_CONTRADICTION,
    SIBLING_DUPLICATE,
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


# -----------------------------------------------------------------------------
# Bloom Verb Rules & Normalization
# -----------------------------------------------------------------------------


SKIP_WORDS = frozenset({
    "a", "an", "the",
    "this", "that", "these", "those",
    "in", "on", "at", "for", "with", "using", "given",
    "from", "by", "during", "considering",
})

ALL_BLOOM_VERBS = frozenset(
    v.lower() for verbs in BLOOM_VERBS.values() for v in verbs
)

BLOOM_ADJACENCY = {
    "L1": {"L1"},
    "L2": {"L2"},
    "L3": {"L3", "L4"},
    "L4": {"L4", "L5"},
    "L5": {"L5", "L6"},
    "L6": {"L5", "L6"},
}

BLOOM_LEVEL_EXTRAS = {
    # Engineering practical design tasks are accepted at L3 application
    "L3": {"design"},
}


def _first_meaningful_word(text: str) -> str:
    """Return the first Bloom verb if present, or first non-stopword."""
    if not text:
        return ""
    words = [w.rstrip(".,:;!?").lower() for w in text.split()]
    clean_words = [w for w in words if w and w not in SKIP_WORDS]
    if not clean_words:
        return ""
    for w in clean_words:
        if w in ALL_BLOOM_VERBS:
            return w
    return clean_words[0]


def bloom_verb_evaluator(question) -> Optional[SlotVerdict]:
    """First meaningful word must be a Bloom verb from this level or an adjacent one."""
    first = _first_meaningful_word(question.question_text or "")
    if not first:
        return _fail(
            question, BLOOM_MISMATCH,
            "question text is empty or contains only stopwords",
        )

    allowed_levels = BLOOM_ADJACENCY.get(question.bloom, {question.bloom})
    allowed_verbs = set()
    for level in allowed_levels:
        allowed_verbs.update(BLOOM_VERBS.get(level, []))
    allowed_verbs.update(BLOOM_LEVEL_EXTRAS.get(question.bloom, set()))

    if first not in allowed_verbs:
        canonical = BLOOM_VERBS.get(question.bloom, [])
        return _fail(
            question, BLOOM_MISMATCH,
            f"first meaningful word {first!r} not in approved verbs for "
            f"{question.bloom} (or adjacent): {sorted(allowed_verbs)}",
            fix=f"Start the question with one of: {', '.join(canonical)}",
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


def _extract_subparts(text: str) -> List[tuple]:
    """Extract (label, subpart_text) from text."""
    pattern = (
        r"(?:^|\n|(?<=[.:;?!]\s)|(?<=\s))"
        r"(?:"
        r"\(([a-eA-E])\)"
        r"|\(([ivxIVX]{1,4})\)"
        r"|\(([1-9])\)"
        r"|([a-eA-E])\)"
        r"|(?<=\s)([a-eA-E])\.)"
        r"\s+"
    )
    matches = list(re.finditer(pattern, text))
    if not matches:
        return []
    parts = []
    for i in range(len(matches)):
        label = matches[i].group(0).strip()
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        part_text = text[start:end].strip()
        parts.append((label, part_text))
    return parts


def topic_fragment_evaluator(question) -> Optional[SlotVerdict]:
    """Reject questions with garbled topic insertion or template artifacts."""
    text = (getattr(question, "question_text", None) or getattr(question, "text", "") or "").strip()
    if not text:
        return None

    # Check for raw bullet or template markers anywhere in question text
    if any(marker in text for marker in ["▪", "•", "→", "◆", "■"]):
        return _fail(
            question, TOPIC_FRAGMENT,
            f"Question contains template or bullet artifacts: {text[:60]!r}",
            fix="Remove bullet markers and unparsed artifact symbols",
        )

    # Pattern: "... of {TOPIC} in detail" or "... of {TOPIC}."
    match = re.search(r"\bof\s+(.+?)(?:\s+in\s+detail|\s*[.?!]|\s*$)", text, re.IGNORECASE)
    if match:
        topic = match.group(1).strip()
        topic_words = topic.split()
        if topic_words and topic_words[0].lower() in {"the", "a", "an"}:
            topic_sub = " ".join(topic_words[1:])
        else:
            topic_sub = topic

        if topic.endswith(",") or topic.endswith(";"):
            return _fail(
                question, TOPIC_FRAGMENT,
                f"Topic ends mid-clause: {topic!r}",
                fix="Use a complete noun phrase as topic",
            )

        if re.search(r"\b(where|which|that|who|when)\b", topic, re.IGNORECASE):
            return _fail(
                question, TOPIC_FRAGMENT,
                f"Topic contains incomplete relative clause: {topic!r}",
                fix="Use a concise concept noun phrase without relative clauses",
            )

        fragment_starters = {
            "executed", "executing", "where", "which", "that", "by", "when",
            "and", "or", "if", "because", "while", "during", "given", "using",
        }
        sub_words = topic_sub.split()
        first_word = sub_words[0].lower().rstrip(".,;:!?") if sub_words else ""
        if first_word in fragment_starters:
            return _fail(
                question, TOPIC_FRAGMENT,
                f"Topic starts with fragment word: {first_word!r}",
                fix="Use a noun phrase rather than a participle or conjunction clause",
            )

        if len(sub_words) < 2 and not (len(sub_words) == 1 and sub_words[0].isupper() and len(sub_words[0]) >= 2):
            if len(topic.strip()) < 3:
                return _fail(
                    question, TOPIC_FRAGMENT,
                    f"Topic phrase is too short or empty: {topic!r}",
                    fix="Provide a substantive concept name",
                )

    return None


L1_L2_VERBS = frozenset({
    "describe", "explain", "state", "list", "define", "outline",
    "identify", "summarize", "mention", "what", "name",
})

L3_VERBS = frozenset({
    "calculate", "compute", "apply", "solve", "determine", "derive",
    "illustrate", "demonstrate", "provide", "show", "construct",
})


def bloom_content_evaluator(question) -> Optional[SlotVerdict]:
    """Ensure cognitive demand matches the assigned Bloom level."""
    bloom = getattr(question, "bloom", "L2")
    text = (getattr(question, "question_text", None) or getattr(question, "text", "") or "").strip()
    if not text:
        return None

    veneer_patterns = [
        (r"\bevaluate\s+the\s+role\s+of\b", "Evaluate the role of is descriptive (L2) rather than an evaluation with criteria (L5)."),
        (r"\banalyze\s+what\s+is\b", "Analyze what is is definitional (L1/L2) rather than analytical (L4)."),
        (r"\bevaluate\s+the\s+definition\b", "Evaluate the definition is definitional (L1) rather than evaluative (L5)."),
    ]
    if bloom in ("L4", "L5", "L6"):
        for pattern, msg in veneer_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return _fail(
                    question, BLOOM_DEPTH_MISMATCH,
                    f"Cognitive depth mismatch for {bloom}: {msg}",
                    fix="Frame the question to require comparative analysis, trade-off assessment, or critical judgment against explicit criteria.",
                )

    sub_parts = _extract_subparts(text)
    if len(sub_parts) > 1 and bloom in ("L4", "L5", "L6"):
        for label, sub_text in sub_parts:
            first = _first_meaningful_word(sub_text)
            if first in L1_L2_VERBS:
                return _fail(
                    question, BLOOM_DEPTH_MISMATCH,
                    f"Subpart {label} uses lower-order verb {first!r} (L1/L2) in a slot marked {bloom}.",
                    fix=f"Align subpart {label} to {bloom} cognitive level (e.g., analyze, compare, evaluate, critique).",
                )
            if bloom in ("L5", "L6") and first in L3_VERBS:
                return _fail(
                    question, BLOOM_DEPTH_MISMATCH,
                    f"Subpart {label} uses procedural verb {first!r} (L3) in a slot marked {bloom}.",
                    fix=f"Elevate subpart {label} to require evaluation, critique, or synthesis.",
                )

    return None


def scenario_consistency_evaluator(question) -> Optional[SlotVerdict]:
    """Detect factual and numerical premise contradictions in question scenarios."""
    text = (getattr(question, "question_text", None) or getattr(question, "text", "") or "").lower()
    if not text:
        return None

    # Contradiction 1: Round-robin quantum vs CPU monopolization
    if ("round-robin" in text or "round robin" in text or "time quantum" in text) and (
        "monopoliz" in text or "monopolises" in text
    ):
        return _fail(
            question, SCENARIO_CONTRADICTION,
            "Contradictory premises: Round-robin scheduling with time quantum precludes CPU monopolization.",
            fix="Align scheduling algorithm with the scenario (e.g., priority scheduling without preemption).",
        )

    # Contradiction 2: Conflicting frame rates / transmission rates
    fps_match = re.search(r"(\d+)\s*(?:fps|frames?\s+per\s+second)", text)
    spf_match = re.search(r"(\d+)\s*(?:seconds?|secs?)\s+per\s+frame", text)
    if fps_match and spf_match:
        fps_val = float(fps_match.group(1))
        spf_val = float(spf_match.group(1))
        if fps_val > 1.0 and spf_val >= 1.0:
            return _fail(
                question, SCENARIO_CONTRADICTION,
                f"Contradictory rates: Scenario specifies both {fps_val} frames per second and {spf_val} second(s) per frame.",
                fix="Specify consistent frame rate and transmission parameters.",
            )

    return None


# -----------------------------------------------------------------------------
# Aggregated list
# -----------------------------------------------------------------------------


LOCAL_EVALUATORS: List[LocalEvaluator] = [
    bloom_verb_evaluator,
    topic_fragment_evaluator,
    scenario_consistency_evaluator,
    bloom_content_evaluator,
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
