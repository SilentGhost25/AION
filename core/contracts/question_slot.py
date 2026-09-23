# core/contracts/question_slot.py

from dataclasses import dataclass, field, replace
from enum import Enum
import logging
from typing import Tuple
from core.contracts.budgets import AnswerBudget, QuestionBudget
from core.contracts.task_signature import TaskSignature

LOG = logging.getLogger("aion.question_slot")


class SlotStatus(str, Enum):
    """
    Lifecycle status of a QuestionSlot during generation.
    - PENDING: Slot initialized, generation not yet attempted.
    - GENERATED: LLM produced a candidate, awaiting validation.
    - PASS: Validated and accepted against all authoritative contracts.
    - UNRESOLVED: Failed validation or exhausted retries without recovery; fail-closed.
    - FAILED: Fatal pipeline failure or unrecoverable error for this slot.
    """
    PENDING = "PENDING"
    GENERATED = "GENERATED"
    PASS = "PASS"
    UNRESOLVED = "UNRESOLVED"
    FAILED = "FAILED"


class UnresolvedSlotException(Exception):
    """
    Raised when a QuestionSlot exhausts retry attempts on content, teacher-suitability,
    or grounding defects without resolving, triggering fail-closed pipeline halting.
    """
    def __init__(self, slot: "QuestionSlot", failure_code: str = "EXHAUSTION_CRITICAL"):
        slot_id = getattr(slot, "slot_id", "unknown_slot")
        super().__init__(f"Slot {slot_id} unresolved after retries (failure_code={failure_code}). Generation fail-closed.")
        self.slot = slot
        self.failure_code = failure_code


@dataclass(frozen=True)
class QuestionContract:
    """
    Immutable, validated structural blueprint for a single question slot.
    Passed to validators to enforce truth.
    """
    slot_id              : str
    question_no          : int
    sub_label            : str
    module_id            : int
    marks                : int
    bloom_level          : str
    bloom_verb           : str
    bloom_operation      : str
    co                   : str
    difficulty           : str
    question_type        : str
    topic                : str
    evidence_ids         : Tuple[str, ...]
    task_signature       : TaskSignature
    math_required        : bool = False
    visual_required      : bool = False
    keywords             : Tuple[str, ...] = field(default_factory=tuple)
    co_assignment_mode   : str = "marks-based"
    status               : str = SlotStatus.PENDING.value


@dataclass(frozen=True)
class QuestionSlot:
    """
    A single addressable question slot in the paper.
    Contains structural specifications, budgets, and evidence refs.
    """
    slot_id              : str
    question_no          : int
    sub_label            : str
    or_pair_id           : str
    is_alternative       : bool
    module_id            : int
    marks                : int
    bloom_level          : str
    bloom_verb           : str
    bloom_operation      : str
    co                   : str
    difficulty           : str
    question_type        : str
    topic                : str
    evidence_ids         : Tuple[str, ...]
    answer_budget        : AnswerBudget
    question_budget      : QuestionBudget
    task_signature       : TaskSignature
    math_required        : bool = False
    visual_required      : bool = False
    generation_seed      : int = 0  # 0 = random
    keywords             : Tuple[str, ...] = field(default_factory=tuple)
    co_assignment_mode   : str = "marks-based"
    status               : str = SlotStatus.PENDING.value

    def __post_init__(self):
        if self.marks <= 0:
            raise ValueError(f"marks must be > 0, got {self.marks}")
        valid_blooms = {"L1", "L2", "L3", "L4", "L5", "L6"}
        if self.bloom_level not in valid_blooms:
            raise ValueError(f"invalid bloom_level: {self.bloom_level}. Must be one of {valid_blooms}")
        if not self.co.startswith("CO"):
            raise ValueError(f"CO label must start with 'CO', got '{self.co}'")
        if not self.keywords:
            LOG.debug(f"Slot {self.slot_id} has no keywords assigned. Generation quality may degrade.")

    def make_attempt_slot(self, attempt: int) -> "QuestionSlot":
        """Return a new QuestionSlot instance for a retry attempt with updated seed."""
        return replace(self, generation_seed=self.generation_seed + attempt)

    def to_contract(self) -> QuestionContract:
        """Map slot to validated immutable contract."""
        return QuestionContract(
            slot_id              = self.slot_id,
            question_no          = self.question_no,
            sub_label            = self.sub_label,
            module_id            = self.module_id,
            marks                = self.marks,
            bloom_level          = self.bloom_level,
            bloom_verb           = self.bloom_verb,
            bloom_operation      = self.bloom_operation,
            co                   = self.co,
            difficulty           = self.difficulty,
            question_type        = self.question_type,
            topic                = self.topic,
            evidence_ids         = self.evidence_ids,
            task_signature       = self.task_signature,
            math_required        = self.math_required,
            visual_required      = self.visual_required,
            keywords             = self.keywords,
            co_assignment_mode   = self.co_assignment_mode,
            status               = self.status,
        )
