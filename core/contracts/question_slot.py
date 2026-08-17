# core/contracts/question_slot.py

from dataclasses import dataclass, replace
from typing import Tuple
from core.contracts.budgets import AnswerBudget, QuestionBudget
from core.contracts.task_signature import TaskSignature


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
    math_required        : bool
    visual_required      : bool


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
    generation_seed      : int = 12345

    def __post_init__(self):
        if self.marks <= 0:
            raise ValueError(f"marks must be > 0, got {self.marks}")
        valid_blooms = {"L1", "L2", "L3", "L4", "L5", "L6"}
        if self.bloom_level not in valid_blooms:
            raise ValueError(f"invalid bloom_level: {self.bloom_level}. Must be one of {valid_blooms}")
        if not self.co.startswith("CO"):
            raise ValueError(f"CO label must start with 'CO', got '{self.co}'")

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
        )
