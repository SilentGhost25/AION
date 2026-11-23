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
from typing import Callable, List, Optional, Tuple

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
    SIBLING_INTERCONNECTED,
    UNVERIFIED,
    SlotVerdict,
)
from core.validation.bloom_validator import (
    BLOOM_VERB_LEVEL_MAP,
    classify_verb,
    get_all_canonical_verbs,
    get_canonical_verbs,
    is_exact_slot_compliant,
)
from .prompts import BLOOM_VERBS, IMAGE_REFERENCE_PHRASES


# -----------------------------------------------------------------------------
# Thresholds
# -----------------------------------------------------------------------------



class CorrespondenceOutcome:
    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"
    LEGACY = "LEGACY"


MIN_QUESTION_CHARS = 20
MAX_QUESTION_CHARS = 1200
MIN_SOLUTION_CHARS = 20
MARKING_SUM_TOLERANCE = 0.5  # allow small rounding drift


# -----------------------------------------------------------------------------
# Evaluator type
# -----------------------------------------------------------------------------


LocalEvaluator = Callable[[object], Optional[SlotVerdict]]
# object is a GeneratedQuestion; typed loosely to avoid circular imports.


def _fail(
    question,
    code: str,
    detail: str,
    fix: Optional[str] = None,
    affected_subparts: Tuple[str, ...] = (),
) -> SlotVerdict:
    return SlotVerdict(
        slot_id=question.slot_id,
        verdict="fail",
        source="local",
        reason_code=code,
        detail=detail,
        suggested_fix=fix,
        affected_subparts=tuple(s.strip("(). ").lower() for s in affected_subparts),
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

ALL_BLOOM_VERBS = get_all_canonical_verbs()

# Exact-level mapping: Adjacency tolerance is explicitly zero for authoritative VTU slot compliance
BLOOM_ADJACENCY = {
    "L1": {"L1"},
    "L2": {"L2"},
    "L3": {"L3"},
    "L4": {"L4"},
    "L5": {"L5"},
    "L6": {"L6"},
}

BLOOM_LEVEL_EXTRAS = {
    # Adjacency and level extras removed for strict VTU compliance
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


def resolve_subpart_plan_correspondence(
    question: Any, plan: Any
) -> Tuple[str, Dict[str, Tuple[Any, Any]], Optional[SlotVerdict]]:
    """
    Establish authoritative correspondence between question.sub_parts and plan.sub_part_plans.
    Handles:
      - Duplicate labels in either collection.
      - Missing plan entries.
      - Extra generated subparts.
      - Case normalization.
      - Missing or ambiguous slot IDs.
    Returns:
      (status, mapping, error_verdict)
      status: "PASS" | "FAIL" | "INDETERMINATE" | "LEGACY"
      mapping: {norm_label: (sp, sp_plan)}
      error_verdict: SlotVerdict if status in ("FAIL", "INDETERMINATE"), else None
    """
    if plan is None or not getattr(plan, "sub_part_plans", None):
        return "LEGACY", {}, None

    # Check slot_id correspondence
    q_slot = getattr(question, "slot_id", None)
    p_slot = getattr(plan, "slot_id", None)
    if not q_slot or not p_slot or q_slot != p_slot:
        return "INDETERMINATE", {}, SlotVerdict(
            slot_id=q_slot or "unknown",
            verdict="unverified",
            source="local",
            reason_code=UNVERIFIED,
            detail=f"slot_id correspondence ambiguous: question slot {q_slot!r} vs plan slot {p_slot!r}",
        )

    q_subparts = getattr(question, "sub_parts", None) or []
    p_subpart_plans = getattr(plan, "sub_part_plans", None) or []

    # Check for duplicate labels in plan
    p_labels = [getattr(sp_p, "label", "").strip().lower() for sp_p in p_subpart_plans]
    if len(p_labels) != len(set(p_labels)):
        dups = sorted(list(set(l for l in p_labels if p_labels.count(l) > 1)))
        return "INDETERMINATE", {}, SlotVerdict(
            slot_id=q_slot,
            verdict="unverified",
            source="local",
            reason_code=UNVERIFIED,
            detail=f"duplicate subpart labels in plan for slot {p_slot}: {dups}",
        )

    # Check for duplicate labels in question
    q_labels = [getattr(sp, "label", "").strip().lower() for sp in q_subparts]
    if len(q_labels) != len(set(q_labels)):
        dups = sorted(list(set(l for l in q_labels if q_labels.count(l) > 1)))
        return "FAIL", {}, _fail(
            question,
            MULTI_PART_STRUCTURE_MISMATCH,
            f"duplicate subpart labels in generated question: {dups}",
            affected_subparts=tuple(dups),
        )

    # Check for missing or extra subparts
    set_q = set(q_labels)
    set_p = set(p_labels)
    if set_q != set_p:
        missing_in_q = sorted(list(set_p - set_q))
        extra_in_q = sorted(list(set_q - set_p))
        err_msg = []
        if missing_in_q:
            err_msg.append(f"missing planned subparts {missing_in_q}")
        if extra_in_q:
            err_msg.append(f"unexpected extra subparts {extra_in_q}")
        return "FAIL", {}, _fail(
            question,
            MULTI_PART_STRUCTURE_MISMATCH,
            f"Subpart correspondence violation: {'; '.join(err_msg)}",
            affected_subparts=tuple(missing_in_q or extra_in_q),
        )

    # Map by normalized label
    p_by_label = {getattr(sp_p, "label", "").strip().lower(): sp_p for sp_p in p_subpart_plans}
    mapping = {}
    for sp in q_subparts:
        lbl = getattr(sp, "label", "").strip().lower()
        mapping[lbl] = (sp, p_by_label[lbl])

    return "PASS", mapping, None


def bloom_verb_evaluator(question, plan=None) -> Optional[SlotVerdict]:
    """First meaningful word must be a canonical Bloom action verb exactly matching target Bloom level."""
    # SubPart-governed path
    if getattr(question, "sub_parts", None) and plan is not None:
        status, mapping, err_verdict = resolve_subpart_plan_correspondence(question, plan)
        if err_verdict is not None:
            return err_verdict
        if status == "PASS":
            for label, (sp, sp_plan) in mapping.items():
                first = _first_meaningful_word(getattr(sp, "text", "") or "")
                if not first:
                    return _fail(
                        question, BLOOM_MISMATCH,
                        f"SubPart ({label}) text is empty or contains only stopwords",
                        affected_subparts=(label,),
                    )
                target_bloom = getattr(sp_plan, "bloom", None) or getattr(question, "bloom", "L2")
                compliant, detected_level, reason = is_exact_slot_compliant(first, target_bloom)
                if not compliant:
                    canonical = sorted(BLOOM_VERB_LEVEL_MAP.get(target_bloom, set()))
                    return _fail(
                        question, BLOOM_MISMATCH,
                        f"SubPart ({label}) opening verb {first!r} does not match target {target_bloom}: {reason}",
                        fix=f"Start SubPart ({label}) with one of: {', '.join(canonical)}",
                        affected_subparts=(label,),
                    )
            return None

    # Legacy / Single-part fallback (when plan is None or question has no subparts)
    first = _first_meaningful_word(getattr(question, "question_text", "") or "")
    if not first:
        return _fail(
            question, BLOOM_MISMATCH,
            "question text is empty or contains only stopwords",
        )

    exp_level = getattr(question, "bloom", "L2")
    compliant, detected_level, reason = is_exact_slot_compliant(first, exp_level)
    if not compliant:
        canonical = sorted(BLOOM_VERB_LEVEL_MAP.get(exp_level, set()))
        return _fail(
            question, BLOOM_MISMATCH,
            reason or f"first meaningful word {first!r} does not match {exp_level}",
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
    """
    Distinguish three separate conditions:
      1. A figure exists in the extracted document.
      2. A figure is actually attached and available (question.image_path).
      3. The question text correctly refers to that figure.
    """
    has_image = bool(question.image_path)
    text = (getattr(question, "question_text", "") or "").lower()
    text_references = any(phrase in text for phrase in IMAGE_REFERENCE_PHRASES)
    is_referenced = bool(text_references or getattr(question, "references_image", False))

    if has_image and not is_referenced:
        return _fail(
            question, MISSING_FIGURE_REFERENCE,
            "figure attached but question text does not reference it",
            fix="Add 'With the aid of the given figure...' or similar phrasing",
        )
    if is_referenced and not has_image:
        return _fail(
            question, MISSING_FIGURE_REFERENCE,
            "question references a figure but no image asset is attached or available",
            fix="Either remove the figure reference or attach the figure",
        )
    return None


def question_length_evaluator(question, plan=None) -> Optional[SlotVerdict]:
    """
    Validate question length under explicit VTU assessment length policy.
    - SubParts: <= 600 chars (for marks <= 6), <= 800 chars (for marks > 6).
    - Composite: max(1200, sum(subpart_limits)).
    - Legacy / Single-part: MIN = 20, MAX = 1200 chars.
    """
    sub_parts = getattr(question, "sub_parts", None) or []
    if sub_parts and len(sub_parts) > 1:
        comp_max = max(MAX_QUESTION_CHARS, sum(600 if getattr(sp, "marks", 5) <= 6 else 800 for sp in sub_parts))
        comp_len = len((getattr(question, "question_text", "") or "").strip())
        if comp_len < MIN_QUESTION_CHARS:
            return _fail(
                question, QUESTION_TOO_SHORT,
                f"composite question is {comp_len} chars, minimum {MIN_QUESTION_CHARS}",
                fix="Expand the question to be self-contained",
            )
        if comp_len > comp_max:
            return _fail(
                question, QUESTION_TOO_LONG,
                f"composite question is {comp_len} chars, maximum {comp_max}",
                fix="Shorten the question text",
            )
        for sp in sub_parts:
            lbl = getattr(sp, "label", "unknown")
            sp_marks = getattr(sp, "marks", 5)
            sp_max = 600 if sp_marks <= 6 else 800
            # Strip outer label markers and trailing marks annotations to measure actual intended content consistently
            sp_raw = getattr(sp, "text", "") or ""
            sp_clean = re.sub(r"^\s*\(?[a-zA-Z0-9ivxIVX]+\)?[.:)]?\s*", "", sp_raw)
            sp_clean = re.sub(r"\s*\[\s*\d+\s*(?:marks?|m)?\s*\]\s*$|\s*\(\s*\d+\s*(?:marks?|m)?\s*\)\s*$", "", sp_clean, flags=re.IGNORECASE).strip()
            if len(sp_clean) < MIN_QUESTION_CHARS:
                return _fail(
                    question, QUESTION_TOO_SHORT,
                    f"SubPart ({lbl}) content is {len(sp_clean)} chars, minimum {MIN_QUESTION_CHARS}",
                    fix=f"Expand SubPart ({lbl}) to be self-contained",
                    affected_subparts=(lbl,),
                )
            if len(sp_clean) > sp_max:
                return _fail(
                    question, QUESTION_TOO_LONG,
                    f"SubPart ({lbl}) content is {len(sp_clean)} chars, maximum {sp_max} for {sp_marks} marks",
                    fix=f"Shorten SubPart ({lbl}) text",
                    affected_subparts=(lbl,),
                )
        return None

    # Single-part / Legacy evaluation
    n = len((getattr(question, "question_text", "") or "").strip())
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


def multi_part_structure_evaluator(question, plan=None) -> Optional[SlotVerdict]:
    """
    Validate multi-part structural integrity:
    - Multi-part question must have >= 2 subparts with matching labels and conserved marks.
    - Single-part question must not have subpart labels.
    """
    sub_parts = getattr(question, "sub_parts", None) or []
    is_multi = getattr(question, "is_multi_part", False)

    if sub_parts:
        if is_multi:
            if len(sub_parts) < 2:
                return _fail(
                    question, MULTI_PART_STRUCTURE_MISMATCH,
                    f"partition {getattr(question, 'partition', [])} implies sub-parts but only {len(sub_parts)} found",
                    fix="Structure the question with explicit (a)/(b) sub-parts",
                )
            labels = [getattr(sp, "label", "").strip().lower() for sp in sub_parts]
            # Check duplicate labels
            if len(labels) != len(set(labels)):
                dups = sorted(list(set(l for l in labels if labels.count(l) > 1)))
                return _fail(
                    question, MULTI_PART_STRUCTURE_MISMATCH,
                    f"duplicate subpart labels: {dups}",
                    affected_subparts=tuple(dups),
                )
            # Check marks conservation across subparts
            subpart_marks_sum = sum(getattr(sp, "marks", 0) for sp in sub_parts)
            q_marks = getattr(question, "marks", 0)
            if subpart_marks_sum != q_marks:
                return _fail(
                    question, MULTI_PART_STRUCTURE_MISMATCH,
                    f"subpart marks sum ({subpart_marks_sum}) != question marks ({q_marks})",
                    fix="Ensure subpart marks sum to question marks",
                )
        else:
            if len(sub_parts) > 1:
                return _fail(
                    question, MULTI_PART_STRUCTURE_MISMATCH,
                    "single-part slot contains multiple subparts",
                    fix="Consolidate into single question",
                )
        return None

    # Legacy regex-based evaluation when sub_parts list is absent
    q_text = getattr(question, "question_text", "") or ""
    has_labels = len(_PART_LABEL.findall(q_text)) >= 2
    if is_multi and not has_labels:
        return _fail(
            question, MULTI_PART_STRUCTURE_MISMATCH,
            f"partition {getattr(question, 'partition', [])} implies sub-parts but none found in text",
            fix="Structure the question with explicit (i)/(ii) or (a)/(b) sub-parts",
        )
    if not is_multi and has_labels:
        return _fail(
            question, MULTI_PART_STRUCTURE_MISMATCH,
            "single-part slot but text contains sub-part labels",
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


class SemanticDemandOutcome:
    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


L1_L2_VERBS = frozenset(
    get_canonical_verbs("L1") | get_canonical_verbs("L2") | {"what", "mention"}
)

L3_VERBS = frozenset(
    get_canonical_verbs("L3") | {"provide", "construct"}
)


def evaluate_cognitive_demand(text: str, target_bloom: str) -> Tuple[str, Optional[str]]:
    """
    Evaluate the intellectual/cognitive demand of a question or subpart text against target_bloom.
    Returns (outcome, detail) where outcome is 'PASS', 'FAIL', or 'INDETERMINATE'.

    Layer 2 Semantic Cognitive Demand:
      - PASS: No detectable violation; structural/contextual evidence confirms target demand.
      - FAIL: Concrete mismatch detected (definitional/procedural veneer, lower-order demand, etc.).
      - INDETERMINATE: Opening verb belongs to target level, but question lacks explicit
        criteria, context, or parameters to confirm target demand. Strictly fail-closed.
    """
    if not text or not text.strip():
        return SemanticDemandOutcome.FAIL, "Question text is empty."

    stripped = text.strip()
    words = stripped.split()
    if len(words) < 2:
        return SemanticDemandOutcome.FAIL, "Question text is too short or incomplete."

    target_bloom = target_bloom.upper() if target_bloom else "L2"
    first = _first_meaningful_word(stripped)

    # 1. Definitional Veneer Check across L2, L3, L4, L5, L6
    if target_bloom in ("L2", "L3", "L4", "L5", "L6"):
        veneer_patterns = [
            (r"\b(?:analyze|analyse|evaluate|design|create|apply|calculate|determine|solve|explain|describe)\s+(?:the\s+)?definition\s+of\b",
             "Definitional veneer: asking for a definition is recall (L1), not high-order cognitive demand."),
            (r"\b(?:analyze|analyse|evaluate|design|create|apply)\s+what\s+(?:is|are)\b",
             "Definitional veneer: 'what is' poses a definitional/recall task (L1/L2)."),
            (r"\b(?:analyze|analyse|evaluate|design|create|apply)\s+what\s+\w+\s+means\b",
             "Definitional veneer: asking what something means is conceptual recall (L1/L2)."),
            (r"\b(?:analyze|analyse|evaluate|design|create|apply)\s+(?:the\s+)?meaning\s+of\b",
             "Definitional veneer: asking for meaning is conceptual recall (L1/L2)."),
            (r"\b(?:evaluate|assess)\s+the\s+role\s+of\b",
             "Descriptive veneer: 'evaluate the role of' is descriptive (L2) rather than evaluative with criteria (L5)."),
        ]
        for pat, msg in veneer_patterns:
            if re.search(pat, stripped, re.IGNORECASE):
                return SemanticDemandOutcome.FAIL, f"Cognitive depth mismatch for {target_bloom}: {msg}"

    # 2. Lower-order verb usage in higher-order slots (L4+)
    if target_bloom in ("L4", "L5", "L6"):
        if first in L1_L2_VERBS:
            return SemanticDemandOutcome.FAIL, f"Question uses lower-order verb {first!r} (L1/L2) in a slot marked {target_bloom}."

    # 3. Procedural calculation verb in evaluative/creative slots (L5+)
    if target_bloom in ("L5", "L6"):
        if first in L3_VERBS:
            return SemanticDemandOutcome.FAIL, f"Question uses procedural verb {first!r} (L3) in slot marked {target_bloom}."
        # Procedural veneer check: evaluating purely by calculating without judgment criteria
        if re.search(r"\b(?:evaluate|assess|critique)\s+.*?\b(?:by\s+calculating|using\s+calculation)\b", stripped, re.IGNORECASE):
            return SemanticDemandOutcome.FAIL, f"Procedural veneer: Question requests routine calculation rather than evaluative judgment for {target_bloom}."

    lower_text = stripped.lower()

    # 4. Level-specific positive indicator vs INDETERMINATE checks
    if target_bloom == "L1":
        # Check if question improperly embeds analytical comparison
        if re.search(r"\b(?:trade-offs?|advantages\s+and\s+disadvantages|compare\s+and\s+contrast)\b", lower_text):
            return SemanticDemandOutcome.FAIL, "Question requires analytical comparison (L4), which exceeds L1 recall demand."
        return SemanticDemandOutcome.PASS, None

    elif target_bloom == "L2":
        # Check if question improperly embeds system design from scratch
        if re.search(r"\b(?:design\s+a\s+(?:custom|complete)|implement\s+an\s+allocator)\b", lower_text):
            return SemanticDemandOutcome.FAIL, "Question requires constructive design (L6), which exceeds L2 explanation demand."
        return SemanticDemandOutcome.PASS, None

    elif target_bloom == "L3":
        # L3 Apply: requires concrete application context, numerical problem, algorithm, or data
        has_numbers = bool(re.search(r"\d+", stripped))
        has_givens = bool(re.search(
            r"\b(?:given|using|for\s+the\s+(?:following|given)|formula\w*|algorithms?|ratio\w*|burst\w*|arrival\w*|frame\w*|quantum\w*|workload\w*|reference\s+string|schedul\w*|parameters?|values?|dataset|matrices|matrix|trace|hit\s+ratio|page\s+faults?|effective\s+(?:memory\s+)?access\s+time|turnaround|waiting\s+times?)\b",
            lower_text
        ))
        if len(words) < 5 or not (has_numbers or has_givens):
            return SemanticDemandOutcome.INDETERMINATE, "Question lacks concrete scenario givens, numerical parameters, or algorithm context required to verify L3 application."
        return SemanticDemandOutcome.PASS, None

    elif target_bloom == "L4":
        # L4 Analyze: requires comparative, trade-off, structural decomposition, or relationship context
        has_analysis_context = bool(re.search(
            r"\b(?:between|compare|contrast|versus|vs\.?|trade-offs?|relationship|impacts?|degradation|difference|hazards?|conflicts?|internal|external|advantages\s+and\s+disadvantages|effects?|breakdown|policies|strategies|mechanisms?|algorithms?|protocols?|snooping|directory|penalt(?:y|ies)|hit\s+rates?|miss\s+rates?|latency|throughput|overhead|performance|efficiency|behavior|scaling|complexity|bounds?|race\s+conditions?|deadlocks?|concurrency|contention|synchronization|preemption|anomal(?:y|ies))\b",
            lower_text
        ))
        if len(words) < 5 or not has_analysis_context:
            return SemanticDemandOutcome.INDETERMINATE, "Question lacks explicit comparative dimensions, trade-offs, or structural relationships required for L4 analysis."
        return SemanticDemandOutcome.PASS, None

    elif target_bloom == "L5":
        # L5 Evaluate: requires explicit criteria, standards, justification context, or trade-offs
        has_eval_criteria = bool(re.search(
            r"\b(?:against|criteria|trade-offs?|suitability|justify\s+why|assess\s+whether|merits|limitations|requirements?|when|under\s+(?:high|low|the)|safety-critical|effectiveness|architectural)\b",
            lower_text
        ))
        if len(words) < 5 or not has_eval_criteria:
            return SemanticDemandOutcome.INDETERMINATE, "Question lacks explicit evaluation criteria, trade-off context, or standard against which to judge for L5 evaluation."
        return SemanticDemandOutcome.PASS, None

    elif target_bloom == "L6":
        # L6 Create: requires constraints, specifications, custom architecture, or requirements
        has_create_context = bool(re.search(
            r"\b(?:satisfying|constraints?|specifications?|requirements?|to\s+eliminate|within\s+o\(|custom|protocol|architecture|algorithm\s+to|scheme\s+to|model\s+for|system\s+for)\b",
            lower_text
        ))
        if len(words) < 5 or not has_create_context:
            return SemanticDemandOutcome.INDETERMINATE, "Question lacks explicit constraints, specifications, or system requirements required for L6 creation/design."
        return SemanticDemandOutcome.PASS, None

    return SemanticDemandOutcome.PASS, None


def bloom_content_evaluator(question, plan=None) -> Optional[SlotVerdict]:
    """Ensure cognitive demand matches the assigned Bloom level across all 6 levels."""
    sub_parts = getattr(question, "sub_parts", None) or []
    if sub_parts and plan is not None:
        status, mapping, err_verdict = resolve_subpart_plan_correspondence(question, plan)
        if err_verdict is not None:
            return err_verdict
        if status == "PASS":
            for label, (sp, sp_plan) in mapping.items():
                sp_text = getattr(sp, "text", "") or ""
                target_bloom = getattr(sp_plan, "bloom", None) or getattr(question, "bloom", "L2")
                sp_outcome, sp_reason = evaluate_cognitive_demand(sp_text, target_bloom)
                if sp_outcome != SemanticDemandOutcome.PASS:
                    return _fail(
                        question, BLOOM_DEPTH_MISMATCH,
                        f"SubPart ({label}): {sp_reason}",
                        fix=f"Align SubPart ({label}) to {target_bloom} cognitive requirements.",
                        affected_subparts=(label,),
                    )
            return None

    # Legacy fallback
    bloom = getattr(question, "bloom", "L2")
    text = (getattr(question, "question_text", None) or getattr(question, "text", "") or "").strip()
    if not text:
        return None

    # Evaluate composite question text
    outcome, reason = evaluate_cognitive_demand(text, bloom)
    if outcome != SemanticDemandOutcome.PASS:
        return _fail(
            question, BLOOM_DEPTH_MISMATCH,
            reason or f"Cognitive demand validation failed with outcome {outcome}",
            fix=f"Align question framing to {bloom} cognitive requirements.",
        )

    # For multi-part questions without plan, evaluate each subpart against question.bloom
    extracted_parts = _extract_subparts(text)
    if len(extracted_parts) > 1:
        for label, sub_text in extracted_parts:
            sp_outcome, sp_reason = evaluate_cognitive_demand(sub_text, bloom)
            if sp_outcome != SemanticDemandOutcome.PASS:
                clean_subpart = label.strip("(). ").lower()
                return _fail(
                    question, BLOOM_DEPTH_MISMATCH,
                    f"Subpart {label}: {sp_reason}",
                    fix=f"Align subpart {label} to {bloom} cognitive requirements.",
                    affected_subparts=(clean_subpart,),
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


def sibling_independence_evaluator(question) -> Optional[SlotVerdict]:
    """
    Reject multi-part questions where subpart (b) references (a) explicitly,
    extends (a), compares against (a)'s answer, or starts with pronouns referring to (a).
    """
    text = (getattr(question, "question_text", None) or getattr(question, "text", "") or "").strip()
    if not text:
        return None

    chain_patterns = [
        (r"\bwith\s+reference\s+to\s+(?:part\s*)?\(?[ab]\)?", "explicit chaining to previous part"),
        (r"\bin\s+(?:the\s+)?context\s+of\s+(?:part\s*)?\(?[ab]\)?", "contextual dependency on previous part"),
        (r"\busing\s+(?:your|the)\s+answer\s+(?:from|to)\s+(?:part\s*)?\(?[ab]\)?", "dependency on previous part's answer"),
        (r"\bextend(?:ing)?\s+(?:the\s+)?(?:above|part\s*\(?[ab]\)?)", "extends previous part instead of testing distinct topic"),
        (r"\bcontinuing\s+from\s+(?:part\s*)?\(?[ab]\)?", "continuity chaining from previous part"),
        (r"\bcompare\s+(?:this\s+)?to\s+(?:your|the)\s+answer\b", "references previous answer"),
        (r"\bhow\s+does\s+(?:this|that)\s+compare\b", "relies on comparison with previous part"),
        (r"\bsubsequent\s+to\s+(?:the\s+)?above\b", "sequential dependency on previous part"),
        (r"\bbased\s+on\s+the\s+result\s+of\b", "result dependency on previous part"),
    ]

    for pattern, desc in chain_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return _fail(
                question, SIBLING_INTERCONNECTED,
                f"Sub-question chaining detected: {desc}",
                fix="Rewrite sub-part (b) as an independent question testing a distinct topic without referencing part (a).",
                affected_subparts=("a", "b"),
            )

    subparts = _extract_subparts(text)
    if len(subparts) > 1:
        for label, sub_text in subparts[1:]:
            first_w = _first_meaningful_word(sub_text)
            if first_w in {"it", "this", "that", "these", "those", "they", "them", "such"}:
                clean_subpart = label.strip("(). ").lower()
                return _fail(
                    question, SIBLING_INTERCONNECTED,
                    f"Sub-part {label} starts with referring pronoun {first_w!r} rather than an independent concept/verb.",
                    fix=f"Make sub-part {label} self-contained with its own noun subject or Bloom verb.",
                    affected_subparts=("a", clean_subpart),
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
    sibling_independence_evaluator,
    marking_scheme_evaluator,
    figure_consistency_evaluator,
    question_length_evaluator,
    solution_evaluator,
    multi_part_structure_evaluator,
]


def run_local_evaluators(question, plan=None) -> Optional[SlotVerdict]:
    """
    Run all local evaluators in order. Return the first failure, or None.
    Passes plan to subpart-aware evaluators while preserving single-arg signature for legacy callers.
    """
    for evaluator in LOCAL_EVALUATORS:
        try:
            verdict = evaluator(question, plan=plan)
        except TypeError:
            verdict = evaluator(question)
        if verdict is not None:
            return verdict
    return None
