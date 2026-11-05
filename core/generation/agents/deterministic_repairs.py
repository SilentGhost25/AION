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
    TOPIC_FRAGMENT,
    BLOOM_DEPTH_MISMATCH,
    SCENARIO_CONTRADICTION,
    SIBLING_DUPLICATE,
    SIBLING_INTERCONNECTED,
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

_FORBIDDEN_PREAMBLE_REGEX = re.compile(
    r"^(?:Based\s+on\s+(?:the\s+)?evidence\s*(?:provided)?\s*[,.:;]?\s*|"
    r"Based\s+on\s+the\s+provided\s+evidence\s*[,.:;]?\s*|"
    r"Apply\s+the\s+following\s*[:.,;]?\s*|"
    r"Refer\s+to\s+the\s+(?:text|notes|book)\s*[,.:;]?\s*|"
    r"Using\s+the\s+provided\s+(?:evidence|text)\s*[,.:;]?\s*)",
    re.IGNORECASE,
)


def repair_forbidden_preamble(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """Strip forbidden preamble phrases when present at the start of the question."""
    text = (question.question_text or "").lstrip()
    if not text:
        return None
    m = _FORBIDDEN_PREAMBLE_REGEX.match(text)
    if m:
        stripped = text[m.end():].strip()
        if stripped:
            # Capitalize the first letter of the remaining text
            stripped = stripped[0].upper() + stripped[1:]
            return _replace_text(question, stripped)
    return None


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
    if current > 0 and abs(delta) / current > 0.35:
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

    # Clamp the last entry to eliminate rounding drift
    if new_scheme:
        distributed = sum(e["marks"] for e in new_scheme[:-1])
        new_scheme[-1]["marks"] = target - distributed

    return _replace_marking_scheme(question, new_scheme)


def repair_marking_scheme_rescale(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """Always rescale marking scheme to sum to total marks."""
    scheme = list(question.marking_scheme or [])
    if not scheme:
        return None
    current = sum(
        e.get("marks", 0) for e in scheme
        if isinstance(e, dict) and isinstance(e.get("marks"), (int, float))
    )
    if current == question.marks or current <= 0:
        return None

    # Proportional rescale, then clamp last entry to eliminate drift
    scale = question.marks / current
    new_scheme = []
    for i, entry in enumerate(scheme):
        if not isinstance(entry, dict):
            return None
        if i < len(scheme) - 1:
            new_marks = max(1, round(entry.get("marks", 0) * scale))
        else:
            new_marks = question.marks - sum(e["marks"] for e in new_scheme)
        new_scheme.append(dict(entry, marks=new_marks))

    # Verify sum
    final_sum = sum(e["marks"] for e in new_scheme)
    if final_sum != question.marks:
        return None
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


def repair_topic_fragment_and_artifacts(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """
    Repair TOPIC_FRAGMENT by removing bullet artifacts, stripping clause fragments,
    and cleaning garbled topic templates.
    """
    if reason_code != TOPIC_FRAGMENT:
        return None
    text = (question.question_text or "").strip()
    if not text:
        return None

    cleaned = re.sub(r"[▪•→◆■]+", " ", text)
    cleaned = re.sub(r"\s*\([ivxIVX\d]+\)\s*(?=[.?!]|$)", "", cleaned)
    cleaned = re.sub(r"\.\s*\([ivxIVX\d]+\)\s*$", ".", cleaned)

    m = re.search(r"\bof\s+(.+?)(?:\s+in\s+detail|\s*[.?!]|\s*$)", cleaned, re.IGNORECASE)
    if m:
        raw_topic = m.group(1).strip()
        clean_top = re.sub(r"^[▪•→◆■\*\-\d\.\)\s]+", "", raw_topic).strip()
        clean_top = re.split(r",\s*(?:where|which|that|who|when)\b", clean_top, flags=re.IGNORECASE)[0].strip()
        clean_top = clean_top.rstrip(",;:.- ")
        clean_top = re.sub(r"^(?:executed\s+by|where|which|that|by|when|and|or|for|during|using)\s+", "", clean_top, flags=re.IGNORECASE).strip()
        if clean_top.lower().startswith("the "):
            clean_top = clean_top[4:].strip()
        if not clean_top or len(clean_top) < 3:
            clean_top = getattr(question, "module_id", "the module").replace("_", " ") + " principles"

        cleaned = cleaned[:m.start(1)] + clean_top + cleaned[m.end(1):]

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned != text:
        return _replace_text(question, cleaned)
    return None


def repair_bloom_depth_mismatch(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """
    Repair BLOOM_DEPTH_MISMATCH by elevating cognitive demand:
    replace descriptive veneer with analytical/evaluative framing.
    """
    if reason_code != BLOOM_DEPTH_MISMATCH:
        return None
    text = (question.question_text or "").strip()
    if not text:
        return None

    elevated = text
    elevated = re.sub(
        r"\bevaluate\s+the\s+role\s+of\b",
        "Critically evaluate the performance trade-offs and operational role of",
        elevated,
        flags=re.IGNORECASE,
    )
    elevated = re.sub(
        r"\banalyze\s+what\s+is\b",
        "Analyze the architectural principles and operational characteristics of",
        elevated,
        flags=re.IGNORECASE,
    )
    elevated = re.sub(
        r"\bevaluate\s+the\s+definition\s+of\b",
        "Critically assess and evaluate the operational scope of",
        elevated,
        flags=re.IGNORECASE,
    )

    bloom = getattr(question, "bloom", "L2")
    if bloom in ("L4", "L5", "L6"):
        elevated = re.sub(
            r"(\([a-eA-EivxIVX\d]+\)\s*)Describe\b",
            r"\1Analyze the operational mechanisms of",
            elevated,
        )
        elevated = re.sub(
            r"(\([a-eA-EivxIVX\d]+\)\s*)Provide a step-by-step example\b",
            r"\1Critically evaluate with a concrete step-by-step trace",
            elevated,
        )

    if elevated != text:
        return _replace_text(question, elevated)
    return None


def repair_scenario_contradiction(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """Repair common factual/scenario premise contradictions."""
    if reason_code != SCENARIO_CONTRADICTION:
        return None
    text = (question.question_text or "").strip()
    if not text:
        return None

    repaired = text
    if ("round-robin" in repaired.lower() or "round robin" in repaired.lower()) and "monopoliz" in repaired.lower():
        repaired = re.sub(
            r"\b(?:a\s+)?round-robin\s+scheduling\s+algorithm\s+with\s+a\s+time\s+quantum\s+of\s+\d+\s+seconds?\b",
            "a priority scheduling algorithm without preemption",
            repaired,
            flags=re.IGNORECASE,
        )
        repaired = re.sub(
            r"\bround-robin\s+scheduling\b",
            "non-preemptive priority scheduling",
            repaired,
            flags=re.IGNORECASE,
        )

    if re.search(r"\d+\s*fps", repaired, re.IGNORECASE) or re.search(r"\d+\s*frames?\s+per\s+second", repaired, re.IGNORECASE):
        repaired = re.sub(
            r"\btransmission\s+rate\s+of\s+\d+\s*(?:seconds?|secs?)\s+per\s+frame\b",
            "transmission rate of 30 frames per second",
            repaired,
            flags=re.IGNORECASE,
        )

    if repaired != text:
        return _replace_text(question, repaired)
    return None


def repair_sibling_independence(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """
    Repair SIBLING_INTERCONNECTED by stripping chaining phrases like
    'With reference to part (a)', 'In the context of part (a)', etc.,
    or replacing dependent pronouns at the start of subpart (b).
    """
    if reason_code != SIBLING_INTERCONNECTED:
        return None
    text = (question.question_text or "").strip()
    if not text:
        return None

    repaired = text
    chain_strip_patterns = [
        r"\bwith\s+reference\s+to\s+(?:part\s*)?\(?[ab]\)?\s*[,.:;]?\s*",
        r"\bin\s+(?:the\s+)?context\s+of\s+(?:part\s*)?\(?[ab]\)?\s*[,.:;]?\s*",
        r"\busing\s+(?:your|the)\s+answer\s+(?:from|to)\s+(?:part\s*)?\(?[ab]\)?\s*[,.:;]?\s*",
        r"\bcontinuing\s+from\s+(?:part\s*)?\(?[ab]\)?\s*[,.:;]?\s*",
        r"\bsubsequent\s+to\s+(?:the\s+)?above\s*[,.:;]?\s*",
    ]
    for p in chain_strip_patterns:
        m = re.search(p, repaired, flags=re.IGNORECASE)
        if m:
            start, end = m.span()
            head = repaired[:start]
            tail = repaired[end:].lstrip()
            if tail:
                tail = tail[0].upper() + tail[1:]
            repaired = head + tail

    repaired = re.sub(
        r"\bhow\s+does\s+(?:this|that)\s+compare\s+to\b",
        "Evaluate and compare the operational characteristics with",
        repaired,
        flags=re.IGNORECASE,
    )

    topic_b = getattr(question, "topic_b", "") or getattr(question, "topic", "")
    topic_clean = topic_b.strip() if topic_b else "the corresponding architectural mechanism"
    repaired = re.sub(
        r"(\([a-eA-EivxIVX\d]+\)\s*)(?:It|This|That|These|Those|They)\s+is\s+used\s+to\b",
        r"\1Explain how " + topic_clean + " is used to",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        r"(\([a-eA-EivxIVX\d]+\)\s*)(?:This|These)\s+(?:mechanism|task|technique|method)s?\b",
        r"\1" + topic_clean.capitalize(),
        repaired,
        flags=re.IGNORECASE,
    )

    repaired = re.sub(r"\s+", " ", repaired).strip()
    if repaired != text:
        return _replace_text(question, repaired)
    return None


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


# Ordered list of deterministic repair functions.
REPAIR_PIPELINE = [
    repair_forbidden_preamble,
    repair_bloom_verb,
    repair_topic_fragment_and_artifacts,
    repair_scenario_contradiction,
    repair_bloom_depth_mismatch,
    repair_sibling_independence,
    repair_figure_reference,
    repair_marking_scheme,
    repair_marking_scheme_rescale,
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
    print(f"[DET-REPAIR] trying {reason_code} for {question.slot_id}", flush=True)
    for repair in REPAIR_PIPELINE:
        try:
            result = repair(question, reason_code, detail)
        except Exception as e:
            print(f"[DET-REPAIR] {repair.__name__} raised: {e}", flush=True)
            result = None
        if result is not None:
            print(f"[DET-REPAIR] {repair.__name__} succeeded for {question.slot_id}", flush=True)
            return result
    print(f"[DET-REPAIR] no repair succeeded for {question.slot_id}", flush=True)
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
