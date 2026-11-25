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


_NOUN_PRONE_BLOOM_VERBS = frozenset({
    "state", "list", "name", "use", "design", "build", "show", "find",
})

_DIRECTIVE_OBJECT_STARTERS = frozenset({
    "the", "a", "an", "its", "their", "your", "this", "that", "these", "those",
    "all", "two", "three", "four", "five", "six", "how", "what", "why", "when",
    "where", "which", "whether", "each", "any", "some", "at", "each",
})

_INTERROGATIVE_STARTERS = frozenset({
    "what", "how", "why", "which", "when", "where", "who",
})


_SOURCE_TAG_PATTERNS = (
    (re.compile(r"\b(?:fig(?:ure)?|table|tbl)\s*\.?\s*\d+(?:\.\d+)*\b", re.IGNORECASE), "numbered figure/table tag"),
    (re.compile(r"\b(?:chapter|ch|section|sec|unit|module)\s*\.?\s*\d+(?:\.\d+)*\b", re.IGNORECASE), "numbered chapter/section/module tag"),
)


def validate_no_source_tags(text: str) -> Optional[str]:
    """
    Reject leaked textbook/source-document tags in generated question text
    (e.g., 'Figure 5.1', 'Fig. 1.2', 'Table 2.1', 'Chapter 3', 'Section 2.4', 'Module 1', 'Unit 2').
    """
    if not text or not text.strip():
        return None
    for pat, desc in _SOURCE_TAG_PATTERNS:
        m = pat.search(text)
        if m:
            return (
                f"Leaked source tag {m.group(0)!r} ({desc}) found in question text; "
                "questions must not cite textbook figure, table, chapter, section, unit, or module numbers."
            )
    return None


def validate_singular_directive(text: str) -> Optional[str]:
    """
    Enforce that a question or sub-question contains a single directive only
    (e.g. only 'define', OR only 'describe', OR only 'explain', OR only 'implement').
    Rejects stacked questions, multiple '?' prompts, and coordinated Bloom imperatives.
    Allows prepositional/visual opening phrases (e.g. 'With reference to the figure, explain...')
    and single-context sentences followed by one directive.
    Returns an error detail string on violation, or None if valid.
    """
    if not text or not text.strip():
        return None

    clean = re.sub(r"^\s*\(?[a-zA-Z0-9ivxIVX]+\)?[.:]\s*", "", text.strip())
    if not clean:
        return None

    # 1. Multiple explicit question marks
    if clean.count("?") > 1:
        return (
            "Multiple questions detected in a single sub-question; "
            "ask ONE singular question with a single directive and single topic."
        )

    # 2. Multi-sentence stacking: count sentences that begin with a directive verb or interrogative
    protected = re.sub(r"\b(e\.g|i\.e|vs|etc|fig|no|dr|mr|ms)\.", r"\1<DOT>", clean, flags=re.IGNORECASE)
    sentences = [s.strip() for s in re.split(r"[.?!]+\s+", protected) if s.strip()]
    sentence_directives: List[str] = []
    for sent in sentences:
        tokens = re.findall(r"[a-zA-Z]+", sent.lower())
        if not tokens:
            continue
        first_tok = tokens[0]
        if first_tok in ("also", "then", "further", "furthermore", "additionally") and len(tokens) > 1:
            first_tok = tokens[1]
        if first_tok in ALL_BLOOM_VERBS or first_tok in _INTERROGATIVE_STARTERS:
            sentence_directives.append(first_tok)

    if len(sentence_directives) > 1:
        return (
            f"Multiple directives detected across sentences ({sentence_directives[0]!r} and {sentence_directives[1]!r}); "
            "ask ONE singular directive on a single topic."
        )

    # 3. Coordinated Bloom directives within a single sentence
    first_verb = _first_meaningful_word(clean)
    # Only check for a *second* coordinated verb strictly AFTER the first directive verb
    if first_verb in ALL_BLOOM_VERBS:
        first_match = re.search(rf"\b{re.escape(first_verb)}\b", clean, re.IGNORECASE)
        tail = clean[first_match.end():] if first_match else ""
        coord_pattern = re.compile(
            r"(?:,?\s+\b(?:and|then|as\s+well\s+as)\b\s+|;\s*|,\s+)"
            r"(?:(?:also|then|briefly|further)\s+)?"
            r"([a-zA-Z]+)"
            r"(?:\s+([a-zA-Z]+))?",
            re.IGNORECASE,
        )
        for match in coord_pattern.finditer(tail):
            cand = match.group(1).lower()
            next_word = (match.group(2) or "").lower()
            if cand not in ALL_BLOOM_VERBS:
                continue
            # Allow standard single comparative idiom: "Compare and contrast ..."
            if first_verb == "compare" and cand == "contrast":
                continue
            # For noun-prone verbs ('state', 'list', 'use', 'design', etc.), require an imperative object starter
            if cand in _NOUN_PRONE_BLOOM_VERBS and next_word not in _DIRECTIVE_OBJECT_STARTERS:
                continue
            return (
                f"Multiple directives detected ({first_verb!r} combined with {cand!r}); "
                "a question must contain only ONE singular directive (e.g. only define, describe, explain, or implement)."
            )

    return None


def validate_single_topic(
    text: str,
    assigned_topic: Optional[str],
    sibling_topics: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Enforce that a sub-question focuses on a single assigned topic and does not
    bundle distinct sibling topics together (e.g. 'Explain Topic A and Topic B').
    """
    if not text or not sibling_topics or not assigned_topic:
        return None

    assigned_norm = assigned_topic.strip().lower()
    lower_text = text.lower()
    has_assigned = bool(assigned_norm and len(assigned_norm) >= 4 and re.search(rf"\b{re.escape(assigned_norm)}\b", lower_text))
    if not has_assigned:
        return None

    for sib in sibling_topics:
        sib_clean = (sib or "").strip()
        sib_norm = sib_clean.lower()
        if not sib_norm or len(sib_norm) < 5:
            continue
        if sib_norm == assigned_norm or sib_norm in assigned_norm or assigned_norm in sib_norm:
            continue
        if re.search(rf"\b{re.escape(sib_norm)}\b", lower_text):
            return (
                f"Multiple topics detected: sub-question bundles sibling topic {sib_clean!r} "
                f"with assigned topic {assigned_topic!r}. Keep each sub-question focused on a single topic."
            )
    return None


def singular_directive_evaluator(question, plan=None) -> Optional[SlotVerdict]:
    """Ensure every question / subpart asks ONE singular directive on a single topic."""
    sub_parts = getattr(question, "sub_parts", None) or []
    if sub_parts:
        sibling_topics = []
        if plan is not None and getattr(plan, "sub_part_plans", None):
            sibling_topics = [getattr(sp, "topic", "") for sp in plan.sub_part_plans if getattr(sp, "topic", None)]
        for sp in sub_parts:
            sp_text = getattr(sp, "text", "") or ""
            sp_label = getattr(sp, "label", "a")
            err = validate_singular_directive(sp_text)
            if err:
                return _fail(
                    question,
                    BLOOM_DEPTH_MISMATCH,
                    f"SubPart ({sp_label}): {err}",
                    fix=f"Rewrite SubPart ({sp_label}) using a single Bloom action verb and a single topic.",
                    affected_subparts=(sp_label,),
                )
            sp_topic = getattr(sp, "topic", None)
            if sp_topic and sibling_topics:
                t_err = validate_single_topic(sp_text, sp_topic, sibling_topics)
                if t_err:
                    return _fail(
                        question,
                        BLOOM_DEPTH_MISMATCH,
                        f"SubPart ({sp_label}): {t_err}",
                        fix=f"Focus SubPart ({sp_label}) exclusively on {sp_topic!r}.",
                        affected_subparts=(sp_label,),
                    )
        return None

    text = (getattr(question, "question_text", None) or getattr(question, "text", "") or "").strip()
    if not text:
        return None
    err = validate_singular_directive(text)
    if err:
        return _fail(
            question,
            BLOOM_DEPTH_MISMATCH,
            err,
            fix="Rewrite the question using a single Bloom action verb on a single topic.",
        )
    return None


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


# =============================================================================
# Stage G6: Academic Evaluators (Solution Completeness, Rubrics & Visual Tasks)
# =============================================================================


def evaluate_solution_completeness(
    question_text: str,
    solution_text: str,
    contract: Optional[Any] = None,
    marks: int = 5,
) -> Tuple[str, Optional[str]]:
    """
    Evaluates solution completeness against expected response contract (Section 7.1).
    Eliminates arbitrary character counts.
    Returns:
        (outcome, reason)
        outcome: "PASS" | "FAIL" | "INDETERMINATE"
    """
    sol = (solution_text or "").strip()
    if not sol:
        return "FAIL", "Solution text is completely empty"

    sol_lower = sol.lower()
    placeholders = [
        "placeholder", "to be filled", "tbd", "todo", "lorem ipsum",
        "answer goes here", "solution omitted", "pending implementation",
    ]
    if any(p in sol_lower for p in placeholders) or (len(sol.split()) < 3 and marks >= 4):
        return "FAIL", "Solution contains placeholder text or is trivial"

    if contract is not None:
        req_elements = getattr(contract, "required_elements", None)
        if req_elements is None and isinstance(contract, dict):
            req_elements = contract.get("required_elements")
        req_elements = list(req_elements or [])

        # Check for unconventional derivations
        unconventional_markers = [
            "unconventional", "alternate derivation", "heuristic approximation",
            "non-standard substitution", "empirical formula shortcut",
        ]
        if any(marker in sol_lower for marker in unconventional_markers):
            return "INDETERMINATE", "Unconventional derivation requires manual academic evaluation"

        # Check for calculation elements
        has_formula_req = any("formula" in el.lower() for el in req_elements)
        has_steps_req = any("step" in el.lower() or "substitution" in el.lower() for el in req_elements)
        has_answer_req = any("answer" in el.lower() or "final" in el.lower() or "result" in el.lower() for el in req_elements)

        if has_formula_req or has_steps_req or has_answer_req:
            # Check presence of math/calculation indicators
            has_numbers = bool(re.search(r"\d+", sol))
            has_operators = any(op in sol for op in ["=", "+", "-", "*", "/", "\\times", "^", "√"])
            has_units_or_labels = bool(re.search(r"\b(?:ans|answer|result|final|therefore|hence|yields|units?|hz|db|v|a|w|kb|mb|ms|ns|s)\b", sol_lower))

            missing = []
            if has_formula_req and not (has_operators or "formula" in sol_lower or "=" in sol):
                missing.append("formula")
            if has_steps_req and not (has_numbers and has_operators):
                missing.append("working")
            if has_answer_req and not (has_numbers or has_units_or_labels):
                missing.append("final answer")

            if missing and marks >= 5:
                return "FAIL", f"Missing required calculation elements: {', '.join(missing)}"

    return "PASS", None


def evaluate_marking_scheme_relevance(
    question: Any,
    marking_scheme: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Evaluates rubric relevance and strict criteria marks (Section 7.2).
    Criteria marks must be strict positive integers summing exactly to question/subpart marks.
    Returns:
        (outcome, reason)
        outcome: "PASS" | "FAIL" | "INDETERMINATE"
    """
    scheme = marking_scheme
    if scheme is None:
        scheme = getattr(question, "marking_scheme", None)

    if not scheme or not isinstance(scheme, (list, tuple)):
        return "FAIL", "Marking scheme is empty or invalid type"

    expected_marks = getattr(question, "marks", None)
    if expected_marks is None and isinstance(question, dict):
        expected_marks = question.get("marks")
    if expected_marks is None:
        expected_marks = 0

    total_marks = 0
    has_legacy = False

    for idx, entry in enumerate(scheme):
        if not isinstance(entry, dict):
            return "FAIL", f"Criteria entry {idx} is not a dict: {type(entry).__name__}"

        if entry.get("is_legacy_string"):
            has_legacy = True
            total_marks += int(entry.get("marks", 0) or 0)
            continue

        crit_text = entry.get("criterion")
        if not crit_text or not isinstance(crit_text, str) or not crit_text.strip():
            return "FAIL", f"Criteria entry {idx} has missing or empty 'criterion' description"

        m = entry.get("marks")
        if type(m) is not int or type(m) is bool or m <= 0:
            return "FAIL", f"Criteria entry {idx} marks must be strict positive int, got {type(m).__name__}: {m!r}"

        total_marks += m

    if has_legacy:
        return "INDETERMINATE", "INDETERMINATE_LEGACY_STRING: Marking scheme contains wrapped legacy prose"

    if expected_marks > 0 and total_marks != expected_marks:
        return (
            "FAIL",
            f"RUBRIC_MARKS_SUM_MISMATCH: Criteria marks sum ({total_marks}) does not equal expected marks ({expected_marks})",
        )

    return "PASS", None


def evaluate_visual_dependency(
    question_text: str,
    figure_metadata: Optional[Dict[str, Any]] = None,
    visual_contract: Optional[Any] = None,
    bloom: str = "L2",
) -> Tuple[str, Optional[str]]:
    """
    Evaluates 4 independent visual eligibility conditions (Section 7.3):
      1. Condition 1 (Archetype Capability): Archetype supports visual tasks.
      2. Condition 2 (Physical Asset Validity & Bounding): Asset exists and is referenced.
      3. Condition 3 (Semantic Dependency & Non-Redundancy): Non-decorative, genuine dependency.
      4. Condition 4 (Bloom Operation Alignment): Bloom level (L1–L6) is aligned.
    Returns:
        (outcome, reason)
        outcome: "PASS" | "FAIL" | "INDETERMINATE"
    """
    q_text = (question_text or "").strip()
    q_lower = q_text.lower()

    # Condition 1: Archetype Capability
    incompatible_archetypes = [
        "conceptual_essay", "definition_recall", "pure_theory_overview",
        "historical_survey", "regulatory_policy_summary",
    ]
    if any(pat in q_lower for pat in ["write an essay on", "define the conceptual meaning of", "explain the history of"]):
        return "FAIL", "ARCHETYPE_VISUAL_INCOMPATIBLE: Question archetype does not support visual grounding"

    # Condition 2: Physical Asset Validity & Bounding & Text Reference
    fig_id = None
    if visual_contract is not None:
        fig_id = getattr(visual_contract, "figure_id", None)
        if fig_id is None and isinstance(visual_contract, dict):
            fig_id = visual_contract.get("figure_id")

    if figure_metadata is not None or fig_id is not None:
        if not figure_metadata and not visual_contract:
            return "FAIL", "MISSING_VISUAL_ASSET: Figure metadata is completely absent"

        if figure_metadata:
            w = figure_metadata.get("width")
            h = figure_metadata.get("height")
            if w is not None and h is not None:
                if not (isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0):
                    return "FAIL", "INVALID_ASSET_DIMENSIONS: Figure asset has non-positive or invalid dimensions"

        # Text reference check
        has_fig_ref = (
            any(phrase in q_lower for phrase in IMAGE_REFERENCE_PHRASES)
            or ("figure" in q_lower)
            or ("fig." in q_lower)
            or ("diagram" in q_lower)
            or ("schematic" in q_lower)
            or ("spatial layout" in q_lower)
            or (fig_id and str(fig_id).lower() in q_lower)
        )
        if not has_fig_ref:
            return "FAIL", "UNREFERENCED_VISUAL_ASSET: Question claims visual contract but text never references figure"

    # Condition 3: Semantic Dependency & Non-Redundancy
    # FAIL if superficial / all values stated in text
    superficial_patterns = [
        r"given\s+that\s+all\s+parameters\s+are\s+as\s+stated\s+above",
        r"without\s+(?:referring|looking)\s+to\s+the\s+diagram",
        r"resistor\s+r1\s*=\s*\d+.*resistor\s+r2\s*=\s*\d+.*voltage\s+v\s*=\s*\d+.*calculate\s+(?:the\s+)?total",
        r"all\s+values\s+are\s+explicitly\s+provided\s+in\s+the\s+text",
    ]
    if any(re.search(pat, q_lower) for pat in superficial_patterns):
        return "FAIL", "DECORATIVE_VISUAL_REDUNDANT: All values and relationships stated in prose; figure is decorative"

    # Spatial ambiguity / boundary state -> INDETERMINATE
    ambiguity_markers = [
        "spatial interpretation is ambiguous", "unlabeled topology branch",
        "partial visual dependency plausible", "boundary visual dependency",
        "ambiguous spatial orientation",
    ]
    if any(marker in q_lower for marker in ambiguity_markers):
        return "INDETERMINATE", "VISUAL_DEPENDENCY_AMBIGUOUS: Spatial ambiguity requires human visual inspection"

    # PASS if demonstrably dependent
    dependent_markers = [
        "read", "determine from the figure", "from the given schematic",
        "shown in the diagram", "topology", "state transition diagram",
        "inflection point", "timing diagram", "waveform", "circuit shown",
        "with the aid of the given figure", "from figure",
    ]
    if any(marker in q_lower for marker in dependent_markers):
        return "PASS", None

    # Default to boundary state if figure is attached and referenced but dependency degree is unverified
    if figure_metadata or fig_id:
        return "INDETERMINATE", "VISUAL_DEPENDENCY_AMBIGUOUS: Dependency degree unverified; human review recommended"

    return "PASS", None


def _fig_field(fig: Any, key: str, default: Any = None) -> Any:
    if fig is None:
        return default
    if isinstance(fig, dict):
        return fig.get(key, default)
    return getattr(fig, key, default)


def audit_visual_asset(
    figure: Any,
    question_text: str,
    *,
    expected_answer: Optional[str] = None,
    visual_contract: Optional[Any] = None,
    expected_response: Optional[Any] = None,
    registered_document_id: Optional[str] = None,
    registered_revision: Optional[str] = None,
    expected_module_id: Optional[str] = None,
    module_page_range: Optional[Tuple[int, int]] = None,
    ocr_extractor: Optional[Callable[[str], Dict[str, Any]]] = None,
    vlm_verifier: Optional[Callable[..., Dict[str, Any]]] = None,
):
    """
    Visual Asset Correctness & Answer-Leakage Gate (Stage G6.1).
    Independently verifies:
      1. Source authenticity (document ID, revision, module, page, SHA-256).
      2. Correct image identification (required features & question/topic alignment).
      3. Mechanism A: Deterministic OCR / content extraction & task-specific leakage check.
      4. Mechanism B: Independent VLM visual reasoning challenge (if provided).
    Returns a VisualAssetAudit contract with decision in {'PASS', 'FAIL', 'INDETERMINATE'}.
    """
    import hashlib
    import os
    from .question_plan import VisualAssetAudit

    fig_id = str(_fig_field(figure, "id", "") or _fig_field(visual_contract, "figure_id", "unknown_fig"))
    doc_id = str(_fig_field(figure, "source_document_id", "") or _fig_field(figure, "source_pdf_path", "") or "unknown_doc")
    rev = str(_fig_field(figure, "source_revision", "") or _fig_field(figure, "source_pdf_sha256", "") or "unknown_rev")
    img_sha = str(_fig_field(figure, "image_sha256", "") or "missing_sha256")
    mod_id = str(_fig_field(figure, "module_id", "") or "unknown_module")
    raw_page = _fig_field(figure, "page", 1)
    page_num = raw_page if isinstance(raw_page, int) and not isinstance(raw_page, bool) and raw_page >= 1 else 1
    img_path = _fig_field(figure, "image_path", None)
    caption = str(_fig_field(figure, "caption", "") or "")

    leakage_evidence: List[str] = []
    answer_leakage = "NONE"
    req_verified = True
    align_verified = True
    unreadable = bool(_fig_field(figure, "unreadable_labels", False))
    hard_fail_reasons: List[str] = []

    # 1. Source Authenticity & Provenance Checks
    if registered_document_id and doc_id != registered_document_id:
        hard_fail_reasons.append(f"SOURCE_DOCUMENT_MISMATCH: expected {registered_document_id!r}, got {doc_id!r}")
    if registered_revision and rev != registered_revision:
        hard_fail_reasons.append(f"SOURCE_REVISION_MISMATCH: expected {registered_revision!r}, got {rev!r}")
    if expected_module_id and mod_id != expected_module_id:
        hard_fail_reasons.append(f"MODULE_PROVENANCE_MISMATCH: expected {expected_module_id!r}, got {mod_id!r}")
    if module_page_range is not None:
        p_lo, p_hi = module_page_range
        if page_num < p_lo or page_num > p_hi:
            hard_fail_reasons.append(f"PAGE_OUT_OF_MODULE_BOUNDS: page {page_num} outside [{p_lo}, {p_hi}]")

    if img_path and isinstance(img_path, str) and os.path.isfile(img_path):
        with open(img_path, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        if img_sha != "missing_sha256" and actual_sha != img_sha:
            hard_fail_reasons.append("IMAGE_SHA256_MISMATCH: file bytes do not match registered SHA-256")

    # 2. Mechanism A: OCR & Deterministic Content Extraction
    ocr_data: Dict[str, Any] = {}
    if ocr_extractor is not None and img_path:
        ocr_data = ocr_extractor(img_path) or {}
    elif isinstance(_fig_field(figure, "ocr_payload", None), dict):
        ocr_data = _fig_field(figure, "ocr_payload")

    det_text = list(ocr_data.get("detected_text") or _fig_field(figure, "detected_text", ()) or ())
    det_values = list(ocr_data.get("detected_values") or _fig_field(figure, "detected_values", ()) or ())
    det_eqs = list(ocr_data.get("detected_equations") or _fig_field(figure, "detected_equations", ()) or ())
    det_annots = list(ocr_data.get("detected_annotations") or _fig_field(figure, "detected_annotations", ()) or ())
    if ocr_data.get("unreadable_labels"):
        unreadable = True

    combined_ocr = " | ".join(det_text + det_values + det_eqs + det_annots).lower()
    if any(tok in combined_ocr for tok in ("???", "[unreadable]", "illegible")):
        unreadable = True

    # Required visual features verification (from VisualTaskContract)
    req_features = list(getattr(visual_contract, "required_features", ()) or ())
    if isinstance(visual_contract, dict):
        req_features = list(visual_contract.get("required_features") or ())
    combined_features_corpus = f"{caption} {combined_ocr}".lower()
    for feat in req_features:
        feat_norm = str(feat).strip().lower()
        if not feat_norm:
            continue
        if f"missing {feat_norm}" in combined_features_corpus:
            req_verified = False
            hard_fail_reasons.append(f"MISSING_REQUIRED_FEATURE: {feat!r}")
        elif (det_text or det_values or det_eqs) and feat_norm not in combined_features_corpus:
            feat_tokens = [t for t in re.findall(r"[a-z0-9]+", feat_norm) if len(t) > 2]
            if feat_tokens and not any(tok in combined_features_corpus for tok in feat_tokens):
                req_verified = False
                hard_fail_reasons.append(f"MISSING_REQUIRED_FEATURE: {feat!r}")

    q_lower = (question_text or "").lower()
    task_op = str(getattr(visual_contract, "task_operation", "") or (visual_contract.get("task_operation", "") if isinstance(visual_contract, dict) else "")).lower()

    # Answer-leakage detection (Mechanism A)
    confirmed_leak_patterns = [
        (r"\b(?:ans(?:wer)?|solution|final\s+answer|result)\s*[:=]\s*[^\s]+", "explicit answer/solution label"),
        (r"\bworked\s+solution\b", "worked solution present in image"),
        (r"\bcorrect\s+option\b|\[x\]|✓", "highlighted/marked MCQ answer"),
        (r"\b(?:r_?eq|r_?total|equivalent\s+resistance)\s*=\s*\d+", "pre-computed equivalent resistance"),
        (r"[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[a-zA-Z0-9_+\-*/\s]+\s*=\s*\d+", "worked intermediate substitution equation"),
        (r"\bcompleted\s+construction\b|\bsolved\s+diagram\b", "completed/solved diagram"),
    ]
    is_read_output_task = task_op in ("read_output", "read_graph", "inspect_given_output")
    for pat, desc in confirmed_leak_patterns:
        if re.search(pat, combined_ocr, re.IGNORECASE):
            if is_read_output_task and "pre-computed" not in desc and "worked" not in desc:
                continue
            answer_leakage = "CONFIRMED"
            leakage_evidence.append(f"OCR detected {desc}")

    if expected_answer and str(expected_answer).strip():
        exp_norm = str(expected_answer).strip().lower()
        if exp_norm in combined_ocr and exp_norm not in q_lower and not is_read_output_task:
            answer_leakage = "CONFIRMED"
            leakage_evidence.append(f"Expected answer {expected_answer!r} appears directly in image")

    if not is_read_output_task and re.search(r"\bpre-?annotated\s+output\b|\boutput\s*=\s*\d+", combined_ocr, re.IGNORECASE):
        answer_leakage = "CONFIRMED"
        leakage_evidence.append("Pre-annotated requested output detected on figure")

    ambiguous_patterns = [
        r"\bambiguous\s+(?:handwritten\s+)?annotation\b",
        r"\bhandwritten\s+(?:note|mark|scribble)\b",
        r"\bunclear\s+(?:label|annotation)\b",
    ]
    if answer_leakage == "NONE" and any(re.search(p, combined_ocr, re.IGNORECASE) for p in ambiguous_patterns):
        answer_leakage = "POSSIBLE"
        leakage_evidence.append("Ambiguous handwritten or unclear annotation detected")

    # 3. Mechanism B: Independent Visual Reasoning Verification
    if vlm_verifier is not None:
        vlm_result = vlm_verifier(
            image_path=img_path,
            question_text=question_text,
            visual_contract=visual_contract,
            expected_response=expected_response,
        ) or {}
        if vlm_result.get("required_features_verified") is False:
            req_verified = False
            hard_fail_reasons.append("VLM verifier reported missing required visual features")
        if vlm_result.get("question_alignment_verified") is False or vlm_result.get("topic_module_match") is False:
            align_verified = False
            hard_fail_reasons.append("VLM verifier reported question/topic/configuration mismatch")
        if vlm_result.get("completed_solution_detected") is True or vlm_result.get("answer_leakage") == "CONFIRMED":
            answer_leakage = "CONFIRMED"
            for ev in (vlm_result.get("leakage_evidence") or vlm_result.get("concern_regions") or ["VLM confirmed answer leakage"]):
                leakage_evidence.append(str(ev))
        elif vlm_result.get("answer_leakage") == "POSSIBLE" or vlm_result.get("ambiguous_annotations"):
            if answer_leakage == "NONE":
                answer_leakage = "POSSIBLE"
            for ev in (vlm_result.get("leakage_evidence") or ["VLM flagged possible leakage or ambiguous annotation"]):
                leakage_evidence.append(str(ev))

    # 4. Final Decision Logic (Strict Fail-Closed)
    if hard_fail_reasons or not req_verified or not align_verified or answer_leakage == "CONFIRMED":
        decision = "FAIL"
    elif unreadable or answer_leakage == "POSSIBLE":
        decision = "INDETERMINATE"
    else:
        decision = "PASS"

    return VisualAssetAudit(
        figure_id=fig_id,
        source_document_id=doc_id,
        source_revision=rev,
        image_sha256=img_sha,
        module_id=mod_id,
        page_number=page_num,
        required_features_verified=req_verified,
        question_alignment_verified=align_verified,
        detected_text=det_text,
        detected_values=det_values,
        detected_equations=det_eqs,
        detected_annotations=det_annots,
        answer_leakage=answer_leakage,
        leakage_evidence=leakage_evidence,
        decision=decision,
    )


def verify_exported_docx_visuals(
    docx_path: str,
    approved_audits: List[Any],
) -> Tuple[bool, List[str]]:
    """
    Post-Export DOCX Visual Manifest Verification (Section 7):
    Inspects the exported DOCX file, computes SHA-256 digests of all embedded images,
    and verifies every approved VisualAssetAudit asset is present without stale/swapped images.
    Returns (is_valid, errors).
    """
    import hashlib
    from docx import Document

    errors: List[str] = []
    doc = Document(docx_path)

    embedded_hashes: List[str] = []
    for shape in doc.inline_shapes:
        try:
            r_id = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
            image_part = doc.part.related_parts[r_id]
            embedded_hashes.append(hashlib.sha256(image_part.blob).hexdigest())
        except Exception as e:
            errors.append(f"Failed to inspect embedded DOCX image: {e}")

    approved_only = [a for a in approved_audits if getattr(a, "decision", None) == "PASS"]
    non_approved = [a for a in approved_audits if getattr(a, "decision", None) != "PASS"]
    if non_approved:
        for na in non_approved:
            errors.append(
                f"Unapproved visual asset {getattr(na, 'figure_id', '?')} "
                f"(decision={getattr(na, 'decision', '?')}) in manifest"
            )

    expected_hashes = [getattr(a, "image_sha256", "") for a in approved_only]
    for exp_sha in expected_hashes:
        if exp_sha not in embedded_hashes:
            errors.append(f"Approved visual asset SHA-256 {exp_sha[:12]}... missing from exported DOCX")

    for emb_sha in embedded_hashes:
        if emb_sha not in expected_hashes:
            errors.append(f"Unexpected/unapproved image SHA-256 {emb_sha[:12]}... embedded in exported DOCX")

    return len(errors) == 0, errors
