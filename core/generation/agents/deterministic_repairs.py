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
from typing import Optional, Tuple

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
from .generated_question import GeneratedQuestion, SubPart
from .local_evaluators import evaluate_cognitive_demand, SemanticDemandOutcome
from .prompts import BLOOM_VERBS, IMAGE_REFERENCE_PHRASES
from core.validation.bloom_validator import BLOOM_VERB_LEVEL_MAP, get_canonical_verbs


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

# Canonical instructional verbs recognized across Bloom taxonomy levels
KNOWN_INSTRUCTIONAL_VERBS = frozenset(
    verb.lower() for verbs in BLOOM_VERB_LEVEL_MAP.values() for verb in verbs
)

# Multiword or qualified instruction patterns where a single-verb replacement
# would produce ungrammatical or semantically broken phrasing (e.g. "Illustrate with a neat sketch")
_MULTIWORD_MODIFIER_PATTERN = re.compile(
    r"^(?:with\s+a\s+|with\s+the\s+|with\s+neat\s+|with\s+|using\s+|by\s+|through\s+|in\s+terms\s+of\b)",
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
    if question.sub_parts:
        return None

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
    Fix BLOOM_MISMATCH safely and conservatively:
      1. If question.sub_parts exists, identify affected subpart(s) from detail or audit,
         swap leading verb with canonical target Bloom verb from BLOOM_VERB_LEVEL_MAP,
         verify against _MULTIWORD_MODIFIER_PATTERN and evaluate_cognitive_demand,
         update subparts, synchronize materialized projections, and verify consistency.
      2. If single-part, strip preamble or swap leading verb, re-validate cognitive demand,
         and update question_text.
      3. Never replace verbs inside multiword modifiers (e.g. 'Illustrate with a neat sketch').
      4. Always fail closed (return None) if no meaning-preserving repair is possible.
    """
    if reason_code != BLOOM_MISMATCH:
        return None

    if question.sub_parts:
        candidate = copy.deepcopy(question)
        # Parse affected subpart labels from detail if specified, e.g. "SubPart (b)"
        affected_labels = re.findall(r"SubPart\s*\(([a-z])\)", detail, re.IGNORECASE)
        affected_set = {l.lower() for l in affected_labels}

        changed = False
        for sp in candidate.sub_parts:
            if affected_set and sp.label.lower() not in affected_set:
                continue

            # Determine target bloom for this subpart
            target_bloom = candidate.bloom or "L2"
            # If detail contains subpart-specific target, e.g. "for Bloom level L4"
            m_sp_bloom = re.search(
                rf"SubPart\s*\({re.escape(sp.label)}\)[^;]*?(?:Bloom\s+level|expected)\s+(L[1-6])",
                detail,
                re.IGNORECASE,
            )
            if m_sp_bloom:
                target_bloom = m_sp_bloom.group(1).upper()
            else:
                m_gen_bloom = re.search(r"\b(?:Bloom\s+level|expected)\s+(L[1-6])\b", detail, re.IGNORECASE)
                if m_gen_bloom:
                    target_bloom = m_gen_bloom.group(1).upper()

            allowed = BLOOM_VERBS.get(target_bloom, [])
            if not allowed:
                canonical = BLOOM_VERB_LEVEL_MAP.get(target_bloom, ())
                if canonical:
                    allowed = list(canonical)
                else:
                    return None
            target_verb = allowed[0]

            sp_text = (sp.text or "").strip()
            if not sp_text:
                return None

            # Attempt 1: check if stripping preamble clause reveals valid verb
            stripped = _PREAMBLE_PATTERN.sub("", sp_text, count=1).strip()
            if stripped != sp_text:
                cand_first = _first_alpha_word(stripped)
                if cand_first in allowed:
                    sp.text = stripped
                    valid, _ = sp.validate()
                    if not valid:
                        return None
                    changed = True
                    continue

            # Strip leading redundant subpart marker if present in text
            sp_text_clean = re.sub(
                rf"^\s*\(?\s*{re.escape(sp.label)}\s*[\)\.\:\-]\s*",
                "",
                sp_text,
                count=1,
                flags=re.IGNORECASE,
            ).strip()

            m = re.match(r"^([A-Za-z]+)(\b.*)$", sp_text_clean, re.DOTALL)
            if not m:
                return None

            leading_word = m.group(1)
            remainder = m.group(2)
            leading_lower = leading_word.lower()

            if leading_lower in allowed:
                continue

            if leading_lower not in KNOWN_INSTRUCTIONAL_VERBS:
                return None

            remainder_clean = remainder.lstrip()
            if _MULTIWORD_MODIFIER_PATTERN.match(remainder_clean):
                return None

            replacement_verb = target_verb.capitalize() if leading_word[0].isupper() else target_verb.lower()
            repaired_sp_text = replacement_verb + remainder

            # Re-validate cognitive demand
            demand_outcome, _ = evaluate_cognitive_demand(repaired_sp_text, target_bloom)
            if demand_outcome == SemanticDemandOutcome.FAIL:
                return None

            sp.text = repaired_sp_text
            valid, err = sp.validate()
            if not valid:
                return None
            changed = True

        if changed:
            candidate.sync_materialized_projections()
            consistent, err = candidate.verify_projection_consistency()
            if consistent:
                return candidate
        return None

    # Single-part legacy path
    text = (question.question_text or "").strip()
    if not text:
        return None

    target_bloom = question.bloom or "L2"
    m_bloom = re.search(r"\b(?:Bloom\s+level|expected)\s+(L[1-6])\b", detail, re.IGNORECASE)
    if m_bloom:
        target_bloom = m_bloom.group(1).upper()

    allowed = BLOOM_VERBS.get(target_bloom, [])
    if not allowed:
        canonical = BLOOM_VERB_LEVEL_MAP.get(target_bloom, ())
        if canonical:
            allowed = list(canonical)
        else:
            return None
    target_verb = allowed[0]

    # Attempt 1: check if stripping a recognized preamble clause reveals a valid verb
    stripped = _PREAMBLE_PATTERN.sub("", text, count=1).strip()
    if stripped != text:
        candidate_first = _first_alpha_word(stripped)
        if candidate_first in allowed:
            return _replace_text(question, stripped)

    # Inspect the opening token of the question
    m = re.match(r"^([A-Za-z]+)(\b.*)$", text, re.DOTALL)
    if not m:
        return None

    leading_word = m.group(1)
    remainder = m.group(2)
    leading_lower = leading_word.lower()

    # Already starts with an allowed Bloom verb for this level -> nothing to repair
    if leading_lower in allowed:
        return None

    # Check if the leading word is a recognized instructional verb
    if leading_lower not in KNOWN_INSTRUCTIONAL_VERBS:
        return None

    # Check for multiword instructional modifiers following the verb
    remainder_clean = remainder.lstrip()
    if _MULTIWORD_MODIFIER_PATTERN.match(remainder_clean):
        return None

    # Safe replacement: swap the single leading instructional verb with target Bloom verb
    replacement_verb = target_verb.capitalize() if leading_word[0].isupper() else target_verb.lower()
    repaired_text = replacement_verb + remainder

    # Re-validate cognitive demand
    demand_outcome, _ = evaluate_cognitive_demand(repaired_text, target_bloom)
    if demand_outcome == SemanticDemandOutcome.FAIL:
        return None

    return _replace_text(question, repaired_text)


def _rescale_criteria(scheme: list, target: int) -> Optional[list]:
    if not scheme or target <= 0:
        return None
    current = sum(
        e.get("marks", 0) for e in scheme
        if isinstance(e, dict) and isinstance(e.get("marks"), (int, float))
    )
    if current == target or current <= 0:
        return None
    scale = target / current
    new_scheme = []
    for i, entry in enumerate(scheme):
        if not isinstance(entry, dict):
            return None
        if i < len(scheme) - 1:
            new_marks = max(1, round(entry.get("marks", 0) * scale))
        else:
            new_marks = target - sum(e["marks"] for e in new_scheme)
        new_scheme.append(dict(entry, marks=new_marks))
    if sum(e["marks"] for e in new_scheme) != target:
        return None
    return new_scheme


def repair_marking_scheme(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """
    Fix MARKING_SCHEME_MISMATCH conservatively for single-part questions:
        - Single-entry scheme: adjust to target.
        - Multi-entry with small delta: distribute the difference across entries.
        - Large delta: give up (return None).
    Multi-part questions fail closed to avoid unauthorized criteria mutation.
    """
    if reason_code != MARKING_SCHEME_MISMATCH:
        return None

    if question.sub_parts:
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
    """Always rescale marking scheme to sum to total marks for single-part questions."""
    if question.sub_parts:
        return None

    scheme = list(question.marking_scheme or [])
    if not scheme:
        return None
    current = sum(
        e.get("marks", 0) for e in scheme
        if isinstance(e, dict) and isinstance(e.get("marks"), (int, float))
    )
    if current == question.marks or current <= 0:
        return None

    new_scheme = _rescale_criteria(scheme, question.marks)
    if new_scheme is None:
        return None
    return _replace_marking_scheme(question, new_scheme)


def repair_figure_reference(
    question: GeneratedQuestion,
    reason_code: str,
    detail: str,
) -> Optional[GeneratedQuestion]:
    """
    Fix MISSING_FIGURE_REFERENCE idempotently and safely.
    Distinguishes three separate conditions:
      1. A figure exists in the extracted document.
      2. A figure is actually attached and available (question.image_path or subpart.image_path).
      3. The question text correctly refers to that figure.

    Idempotent behavior:
      - If no figure asset is attached, returns None (cannot invent an image).
      - If text already contains any valid figure reference phrase, do NOT prepend another reference!
        Simply ensure question.references_image = True.
      - If already correctly referenced and flagged, returns None (no-op).
      - Only prepends 'With the aid of the given figure, ' when an asset is attached AND the text
        completely lacks any figure reference phrase.
    """
    if reason_code != MISSING_FIGURE_REFERENCE:
        return None

    has_image = bool(question.image_path) or any(bool(sp.image_path) for sp in question.sub_parts)
    if not has_image:
        return None

    prefix = "With the aid of the given figure, "

    if question.sub_parts:
        return None

    text = (question.question_text or "").strip()
    if not text:
        return None

    lower_text = text.lower()
    already_has_reference_phrase = any(phrase in lower_text for phrase in IMAGE_REFERENCE_PHRASES)

    if already_has_reference_phrase or question.references_image:
        if not question.references_image:
            repaired = copy.deepcopy(question)
            repaired.references_image = True
            return repaired
        return None

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

    if question.sub_parts:
        return None

    text = (question.question_text or "").strip()
    if len(text) >= 20:
        return None

    topic = (question.topic or "").strip()
    if not topic:
        return None

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
    Repair MULTI_PART_STRUCTURE_MISMATCH deterministically when:
    1. Case normalization: subpart labels are uppercase (e.g. ['A', 'B']) or whitespace-padded,
       and unambiguously map to canonical sequence ('a', 'b', ...).
    2. Redundant leading labels in SubPart.text: e.g. text starts with "(a) " or "a. " when label is already "a".
    3. Materialization desynchronization: question.sub_parts is valid and complete, but question_text
       is desynchronized or missing subpart markers.

    NEVER attempt to repair:
    - Missing subparts (e.g. partition [6, 4] but only 1 subpart exists).
    - Extra subparts (e.g. partition [6, 4] but 3 subparts exist).
    - Duplicate labels that cannot be disambiguated (e.g. ['a', 'a']).
    - Mark sum mismatches across subparts.
    For these cases, return None so refinement or regeneration handles them.
    """
    if reason_code != MULTI_PART_STRUCTURE_MISMATCH:
        return None

    if not question.sub_parts or len(question.sub_parts) < 2:
        return None

    expected_count = len(question.partition) if question.partition else len(question.sub_parts)
    if len(question.sub_parts) != expected_count:
        return None

    candidate = copy.deepcopy(question)
    expected_labels = [chr(ord('a') + i) for i in range(len(candidate.sub_parts))]

    # Inspect current labels
    raw_labels = [sp.label for sp in candidate.sub_parts]
    normalized_labels = [str(l).strip().lower() for l in raw_labels]

    if normalized_labels != expected_labels:
        # Not a simple case/whitespace deviation; duplicate labels or non-canonical sequence
        return None

    # Check marks conservation across subparts
    subpart_marks = [sp.marks for sp in candidate.sub_parts]
    if sum(subpart_marks) != candidate.marks:
        return None
    if candidate.partition and subpart_marks != candidate.partition:
        return None

    changed = False
    for i, sp in enumerate(candidate.sub_parts):
        target_label = expected_labels[i]
        if sp.label != target_label:
            sp.label = target_label
            changed = True

        # Strip redundant leading label from sp.text
        # e.g. "(a) Describe...", "(A) Describe...", "a. Describe...", "a) Describe..."
        old_raw = raw_labels[i]
        redundant_pattern = rf"^\s*\(?\s*(?:{re.escape(target_label)}|{re.escape(str(old_raw).strip())})\s*[\)\.\:\-]\s*|^\s*\(\s*(?:{re.escape(target_label)}|{re.escape(str(old_raw).strip())})\s*\)\s*"
        cleaned_text = re.sub(redundant_pattern, "", sp.text, count=1, flags=re.IGNORECASE).strip()
        if cleaned_text and cleaned_text != sp.text:
            sp.text = cleaned_text
            changed = True

        valid, err = sp.validate()
        if not valid:
            return None

    # Check projection consistency or sync if changed or desynchronized
    consistent, _ = candidate.verify_projection_consistency()
    if not consistent or changed:
        candidate.sync_materialized_projections()
        consistent, err = candidate.verify_projection_consistency()
        if not consistent:
            return None
        return candidate

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

    def _clean_text(text: str, topic_default: str) -> str:
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
                clean_top = topic_default
            cleaned = cleaned[:m.start(1)] + clean_top + cleaned[m.end(1):]

        return re.sub(r"\s+", " ", cleaned).strip()

    topic_fallback = getattr(question, "module_id", "the module").replace("_", " ") + " principles"

    if question.sub_parts:
        return None

    text = (question.question_text or "").strip()
    if not text:
        return None
    cleaned = _clean_text(text, topic_fallback)
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

    def _elevate(text: str, bloom: str) -> str:
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
        return elevated

    if question.sub_parts:
        return None

    text = (question.question_text or "").strip()
    if not text:
        return None
    elevated = _elevate(text, getattr(question, "bloom", "L2"))
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

    def _fix_contradiction(text: str) -> str:
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
        return repaired

    if question.sub_parts:
        return None

    text = (question.question_text or "").strip()
    if not text:
        return None
    repaired = _fix_contradiction(text)
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

    chain_strip_patterns = [
        r"\bwith\s+reference\s+to\s+(?:part\s*)?\(?[ab]\)?\s*[,.:;]?\s*",
        r"\bin\s+(?:the\s+)?context\s+of\s+(?:part\s*)?\(?[ab]\)?\s*[,.:;]?\s*",
        r"\busing\s+(?:your|the)\s+answer\s+(?:from|to)\s+(?:part\s*)?\(?[ab]\)?\s*[,.:;]?\s*",
        r"\bcontinuing\s+from\s+(?:part\s*)?\(?[ab]\)?\s*[,.:;]?\s*",
        r"\bsubsequent\s+to\s+(?:the\s+)?above\s*[,.:;]?\s*",
    ]

    def _strip_chaining(text: str) -> str:
        repaired = text
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
        return repaired

    if question.sub_parts:
        return None

    # Single-part legacy path
    text = (question.question_text or "").strip()
    if not text:
        return None

    repaired = _strip_chaining(text)
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
    repair_multi_part_structure,
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
    if q.sub_parts and len(q.sub_parts) == 1 and not q.is_multi_part:
        q.sub_parts[0].text = new_text
        q.sync_materialized_projections()
    return q


def _replace_marking_scheme(
    question: GeneratedQuestion, new_scheme: list
) -> GeneratedQuestion:
    q = copy.deepcopy(question)
    q.marking_scheme = new_scheme
    if q.sub_parts and len(q.sub_parts) == 1 and not q.is_multi_part:
        q.sub_parts[0].marking_scheme = new_scheme
        q.sync_materialized_projections()
    return q
