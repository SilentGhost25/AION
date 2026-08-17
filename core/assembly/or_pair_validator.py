# core/assembly/or_pair_validator.py

import re
import logging
from enum import Enum
from typing import Any, List, Tuple, Set, FrozenSet
from core.contracts.question_slot import QuestionSlot
from core.contracts.question import GeneratedQuestion

LOG = logging.getLogger(__name__)
OR_SIMILARITY_THRESHOLD = 0.50
MAX_OR_ATTEMPTS = 3


class Policy(str, Enum):
    REQUIRED = "REQUIRED"
    FORBIDDEN = "FORBIDDEN"


class ORRegenerationContractViolation(Exception):
    """Raised when OR regeneration changes contract fields."""
    pass


class ORPairDeduplicationFailed(Exception):
    """Raised when OR pair remains too similar after attempts."""
    pass


def _derive_math_policy(slot: QuestionSlot) -> Policy:
    return Policy.REQUIRED if slot.math_required else Policy.FORBIDDEN


def _derive_visual_policy(slot: QuestionSlot) -> Policy:
    return Policy.REQUIRED if slot.visual_required else Policy.FORBIDDEN


def assert_contract_unchanged(
    original_slot : QuestionSlot,
    result_question: GeneratedQuestion
) -> None:
    """
    H5 — Validates the ENTIRE contract, not just CO.
    Called after OR alternative regeneration.
    """
    mismatches = []

    checks = [
        ("module_id",     str(result_question.module_id),  str(original_slot.module_id)),
        ("co",            result_question.co,               original_slot.co),
        ("marks",         str(result_question.marks),       str(original_slot.marks)),
        ("bloom_level",   result_question.bloom_level,      original_slot.bloom_level),
        ("bloom_verb",    result_question.bloom_verb,       original_slot.bloom_verb),
        ("sub_label",     result_question.sub_label,        original_slot.sub_label),
        ("question_type", result_question.question_type,    original_slot.question_type),
        ("difficulty",    result_question.difficulty,       original_slot.difficulty),
        ("math_policy",   result_question.math_policy,      _derive_math_policy(original_slot).value),
        ("visual_policy", result_question.visual_policy,    _derive_visual_policy(original_slot).value),
    ]

    for field, actual, expected in checks:
        if actual != expected:
            mismatches.append(
                f"  {field}: expected={expected!r}, got={actual!r}"
            )

    if mismatches:
        raise ORRegenerationContractViolation(
            f"OR regeneration changed contract fields for {original_slot.slot_id}:\n"
            + "\n".join(mismatches) + "\n"
            f"This is a SlotBuilder or Orchestrator bug, not a Qwen error."
        )


def compute_or_similarity(q_a: List[GeneratedQuestion], q_b: List[GeneratedQuestion]) -> float:
    """Computes Jaccard word similarity between alternatives A and B."""
    text_a = " ".join(q.question_text.lower() for q in q_a)
    text_b = " ".join(q.question_text.lower() for q in q_b)
    words_a = set(re.findall(r'\b\w+\b', text_a))
    words_b = set(re.findall(r'\b\w+\b', text_b))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def validate_and_repair(
    q_a          : List[GeneratedQuestion],
    q_b          : List[GeneratedQuestion],
    slots_b      : List[QuestionSlot],
    excluded     : FrozenSet[str],
    orchestrator : Any,
) -> Tuple[List[GeneratedQuestion], List[GeneratedQuestion]]:
    """Enforces semantic deduplication between alternative sets."""
    similarity = compute_or_similarity(q_a, q_b)
    if similarity < OR_SIMILARITY_THRESHOLD:
        return q_a, q_b

    LOG.info(
        f"[OR] Similarity {similarity:.3f} >= {OR_SIMILARITY_THRESHOLD} "
        f"— regenerating B"
    )

    for attempt in range(MAX_OR_ATTEMPTS):
        # Collect concepts/topics from A to exclude in B
        topics_a  = frozenset(q.topic for q in q_a)
        b_excluded = excluded | topics_a

        new_q_b = []
        for slot_b in slots_b:
            # We try to reload or build a new evidence pack if possible
            # In slot orchestrator it will reload evidence based on b_excluded
            evidence_pack = getattr(orchestrator, "artifact", None)
            if hasattr(orchestrator, "_reload_evidence"):
                evidence_pack = orchestrator._reload_evidence(slot_b, b_excluded)

            new_q = orchestrator.generate(
                slot              = slot_b,
                evidence_pack     = evidence_pack,
                excluded_concepts = b_excluded,
            )

            # H5 — Full contract validation after regeneration
            assert_contract_unchanged(slot_b, new_q)
            new_q_b.append(new_q)

        new_sim = compute_or_similarity(q_a, new_q_b)
        LOG.info(f"[OR] attempt {attempt+1}: similarity={new_sim:.3f}")

        if new_sim < OR_SIMILARITY_THRESHOLD:
            return q_a, new_q_b

    raise ORPairDeduplicationFailed(
        f"OR pair still similar ({new_sim:.3f}) after {MAX_OR_ATTEMPTS} "
        f"attempts. Wider evidence pool or different topics needed."
    )
