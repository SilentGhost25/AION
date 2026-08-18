# core/contracts/question.py

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.contracts.question_slot import QuestionSlot
    from core.generation.output_schema import QuestionOutput


@dataclass(frozen=True)
class QuestionProvenance:
    """
    Immutable provenance record attached to every GeneratedQuestion.

    The chain this represents:
        PDF chunks -> evidence_ids -> QuestionSlot -> GeneratedQuestion

    All fields are sourced exclusively from the immutable QuestionSlot
    (never from the LLM output). The ExportGate uses this to verify that
    CO, Bloom, marks, module, and evidence are all self-consistent.

    source_hash is derived from slot.evidence_ids so it is stable and
    reproducible, not tied to the volatile _raw_evidence field.
    """
    slot_id      : str
    module_id    : int
    co           : str
    bloom_level  : str
    marks        : int
    evidence_ids : Tuple[str, ...]   # chunk IDs from the generating module
    source_hash  : str               # sha256 of sorted evidence_ids (16 hex chars)

    @classmethod
    def from_slot(cls, slot: "QuestionSlot") -> "QuestionProvenance":
        """
        Build provenance deterministically from the immutable QuestionSlot.
        The source hash is sha256 of the sorted, joined evidence_ids.
        This is stable across runs and reproducible for audit.
        """
        eid_key = "|".join(sorted(slot.evidence_ids)) if slot.evidence_ids else ""
        source_hash = hashlib.sha256(eid_key.encode("utf-8")).hexdigest()[:16]
        return cls(
            slot_id      = slot.slot_id,
            module_id    = slot.module_id,
            co           = slot.co,
            bloom_level  = slot.bloom_level,
            marks        = slot.marks,
            evidence_ids = tuple(slot.evidence_ids),
            source_hash  = source_hash,
        )


@dataclass
class QuestionIntent:
    """
    The complete specification for one sub-question slot.
    Qwen receives this. Qwen does not make decisions about anything in this object.
    """
    slot_id          : str
    question_no      : int
    sub_label        : str
    module_id        : int
    marks            : int          # LOCKED
    bloom            : str          # LOCKED e.g. "L3"
    bloom_verb       : str          # LOCKED e.g. "Calculate"
    bloom_operation  : str          # LOCKED e.g. "APPLY"
    co               : str          # LOCKED
    question_type    : str          # LOCKED e.g. "NUMERICAL"
    difficulty_band  : str          # LOCKED e.g. "MEDIUM"
    visual_decision  : str          # LOCKED "IMAGE_REQUIRED"|"IMAGE_NOT_NEEDED"

    # Evidence
    concept          : str
    evidence_text    : List[str]
    evidence_pages   : List[int]
    grounding_score  : float

    # Math artifacts (references only — M4 invariant)
    math_artifacts   : Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Solver (for numerical/graph questions)
    solver_context   : Optional[Dict[str, Any]] = None

    # Constraints for Qwen
    expected_answer_type : str = "descriptive"
    forbidden_phrases    : List[str] = field(default_factory=list)
    required_entities    : List[str] = field(default_factory=list)

    # Generation seed for reproducibility
    generation_seed  : int = 42


class GeneratedQuestion:
    """
    Canonical v5 GeneratedQuestion representation.
    Metadata fields are strictly assigned from the immutable QuestionSlot.
    """
    def __init__(self, output: QuestionOutput, slot: QuestionSlot):
        self.output = output
        self.slot = slot

        # Strictly map metadata from Slot (no Qwen override)
        self.slot_id = slot.slot_id
        self.question_text = output.question_text
        self.marks = slot.marks
        self.bloom = slot.bloom_level
        self.bloom_level = slot.bloom_level
        self.bloom_verb = slot.bloom_verb
        self.bloom_operation = slot.bloom_operation
        self.co = slot.co
        self.module_id = slot.module_id
        self.sub_label = slot.sub_label
        self.question_type = slot.question_type
        self.difficulty = slot.difficulty
        self.math_policy = "REQUIRED" if slot.math_required else "FORBIDDEN"
        self.visual_policy = "REQUIRED" if slot.visual_required else "FORBIDDEN"
        self.evidence_ids = slot.evidence_ids
        self.topic = slot.topic
        self.question_no = slot.question_no
        self.math_placeholders = [b.block_id for b in output.math_blocks]
        self.math_blocks = output.math_blocks

        # Set default audit/rendering fields
        self.visual_asset = None
        self.diagram_request = output.diagram_request if hasattr(output, "diagram_request") else None
        self.integrity_score = 100.0
        self.warnings: List[str] = []
        self.status = "PASS"

        # Immutable provenance record — sourced entirely from slot, never from LLM output
        self.provenance: QuestionProvenance = QuestionProvenance.from_slot(slot)


class LegacyGeneratedQuestionAdapter:
    """Adapts legacy inputs to canonical v5 GeneratedQuestion structure at boundaries."""
    @classmethod
    def adapt(
        cls,
        slot_id: str,
        question_text: str,
        marks: int,
        bloom: str,
        co: str,
        topic: str = "",
        module_id: int = 1,
        question_no: int = 1,
        sub_label: str = "a",
        evidence_ids: Tuple[str, ...] = ()
    ) -> GeneratedQuestion:
        from core.contracts.question_slot import QuestionSlot
        from core.contracts.task_signature import TaskSignature
        from core.contracts.budgets import AnswerBudget, QuestionBudget
        from core.generation.output_schema import QuestionOutput

        # Extract a basic instruction clause
        sentences = question_text.split(".")
        instruction = sentences[0] if sentences else question_text

        output = QuestionOutput(
            instruction=instruction,
            question_text=question_text,
            math_blocks=[]
        )

        slot = QuestionSlot(
            slot_id=slot_id,
            question_no=question_no,
            sub_label=sub_label,
            or_pair_id="OR_legacy",
            is_alternative=False,
            module_id=module_id,
            marks=marks,
            bloom_level=bloom,
            bloom_verb="Explain",
            bloom_operation="UNDERSTAND",
            co=co,
            difficulty="MEDIUM",
            question_type="descriptive",
            topic=topic,
            evidence_ids=evidence_ids,
            answer_budget=AnswerBudget.from_marks_and_bloom(marks, bloom),
            question_budget=QuestionBudget.from_bloom(bloom, marks),
            task_signature=TaskSignature.from_bloom_marks_type(bloom, marks, "descriptive"),
            math_required=False,
            visual_required=False,
            generation_seed=42
        )

        return GeneratedQuestion(output, slot)
