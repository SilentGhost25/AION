"""
AION Structural Architecture v2 — Data Structures & Contracts
===============================================================
Core immutable types, structural signatures, equivalence profiles,
and locked question slot contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union


class BloomLevel(IntEnum):
    L1 = 1   # Remember
    L2 = 2   # Understand
    L3 = 3   # Apply
    L4 = 4   # Analyze
    L5 = 5   # Evaluate
    L6 = 6   # Create


class DistributionPolicy(str, Enum):
    BALANCED       = "BALANCED"
    PRIMARY_HEAVY  = "PRIMARY_HEAVY"
    PROGRESSIVE    = "PROGRESSIVE"
    CUSTOM         = "CUSTOM"


class VisualPrior(str, Enum):
    FORBIDDEN    = "FORBIDDEN"
    DISCOURAGED  = "DISCOURAGED"
    OPTIONAL     = "OPTIONAL"
    PREFERRED    = "PREFERRED"


class VisualDecision(str, Enum):
    IMAGE_REQUIRED     = "IMAGE_REQUIRED"
    IMAGE_OPTIONAL     = "IMAGE_OPTIONAL"
    IMAGE_NOT_NEEDED   = "IMAGE_NOT_NEEDED"


class SlotStatus(str, Enum):
    EMPTY            = "EMPTY"
    STRUCTURE_LOCKED = "STRUCTURE_LOCKED"
    GROUNDING_READY  = "GROUNDING_READY"
    CONTENT_READY    = "CONTENT_READY"
    GENERATED        = "GENERATED"
    VALIDATED        = "VALIDATED"
    FAILED           = "FAILED"


class DifficultyBand(str, Enum):
    EASY   = "EASY"
    MEDIUM = "MEDIUM"
    HARD   = "HARD"


class RecoveryAction(str, Enum):
    NEW_EVIDENCE  = "NEW_EVIDENCE"
    NEW_CONCEPT   = "NEW_CONCEPT"
    SLOT_FAILED   = "SLOT_FAILED"
    PAPER_FAILED  = "PAPER_FAILED"


@dataclass(frozen=True)
class StructuralSignature:
    """
    Immutable structural identity shared by ALL questions in a paper.
    Frozen dataclass — fields cannot be modified after creation.
    """
    total_marks         : int
    sub_question_count  : int
    mark_distribution   : Tuple[int, ...]   # tuple for hashability
    distribution_policy : DistributionPolicy = DistributionPolicy.BALANCED

    def __post_init__(self):
        assert sum(self.mark_distribution) == self.total_marks, (
            f"Distribution sum {sum(self.mark_distribution)} "
            f"!= total_marks {self.total_marks}"
        )
        assert len(self.mark_distribution) == self.sub_question_count, (
            f"Distribution length {len(self.mark_distribution)} "
            f"!= sub_question_count {self.sub_question_count}"
        )
        assert all(m >= 1 for m in self.mark_distribution), (
            "All slot marks must be >= 1"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructuralSignature):
            return False
        # Policy is NOT compared — only numerical outcome matters
        return (
            self.total_marks       == other.total_marks       and
            self.sub_question_count == other.sub_question_count and
            self.mark_distribution  == other.mark_distribution
        )

    def __hash__(self):
        return hash((
            self.total_marks,
            self.sub_question_count,
            self.mark_distribution
        ))

    def __repr__(self):
        return (
            f"σ(marks={self.total_marks}, "
            f"n={self.sub_question_count}, "
            f"D={list(self.mark_distribution)})"
        )


@dataclass(frozen=True)
class AlternativeEquivalenceProfile:
    """
    Locks academic equivalence constraints for an OR pair.
    Both alternatives MUST satisfy this profile.
    Topics may differ; cognitive workload must be comparable.
    """
    bloom_profile         : Tuple[BloomLevel, ...]       # per sub-question
    difficulty_profile    : Tuple[DifficultyBand, ...]
    question_type_profile : Tuple[str, ...]            # domain-specific types
    cognitive_weights     : Tuple[float, ...]            # sums to 1.0

    def __post_init__(self):
        n = len(self.bloom_profile)
        assert len(self.difficulty_profile)   == n, f"Difficulty profile len != {n}"
        assert len(self.question_type_profile) == n, f"Question type profile len != {n}"
        assert len(self.cognitive_weights)     == n, f"Cognitive weights len != {n}"
        assert abs(sum(self.cognitive_weights) - 1.0) < 1e-4, (
            f"Cognitive weights must sum to 1.0, got {sum(self.cognitive_weights)}"
        )


@dataclass
class QuestionSlot:
    """
    A single scorable unit within a question paper.
    Fields are partitioned into STRUCTURAL (locked early)
    and CONTENT (filled later).
    """

    # ── STRUCTURAL FIELDS (locked at S3) ─────────────────────────────────────
    slot_id          : str                 # e.g. "Q1a"
    question_number  : int                 # e.g. 1
    sub_label        : str                 # e.g. "a"
    module_id        : int
    marks            : int                 # IMMUTABLE after S3
    bloom            : BloomLevel          # IMMUTABLE after S3
    co               : str                 # IMMUTABLE after S3
    question_type    : str                 # IMMUTABLE after S3
    difficulty_band  : DifficultyBand      # IMMUTABLE after S3
    visual_prior     : VisualPrior         # IMMUTABLE after S3
    status           : SlotStatus = SlotStatus.EMPTY

    # ── CONTENT FIELDS (filled at content stage) ──────────────────────────────
    concept          : Optional[Any] = None
    evidence_chunks  : Optional[List[Dict[str, Any]]] = None
    evidence_pages   : Optional[List[int]]  = None
    grounding_score  : Optional[float] = None

    # ── GENERATION FIELDS (filled by Qwen) ───────────────────────────────────
    question_text    : Optional[str] = None
    visual_decision  : Optional[VisualDecision] = None
    visual_asset     : Optional[Dict[str, Any]] = None
    solver_context   : Optional[Dict[str, Any]] = None   # from VRE/solver
    generation_context: Optional[Dict[str, Any]] = None

    # ── PROVENANCE (audit trail) ──────────────────────────────────────────────
    generation_seed  : Optional[int] = None
    model_used       : Optional[str] = None
    source_chunk_ids : List[str] = field(default_factory=list)

    def lock_structure(self):
        """Called exactly once. Prevents structural field mutation."""
        self.status = SlotStatus.STRUCTURE_LOCKED
        object.__setattr__(self, "_structure_locked", True)

    def assert_structure_locked(self):
        assert getattr(self, "_structure_locked", False), (
            f"Slot {self.slot_id}: structure not locked before content fill"
        )


@dataclass
class Alternative:
    question_id  : str
    slots        : List[QuestionSlot]
    signature    : StructuralSignature
    profile      : AlternativeEquivalenceProfile

    def mark_sum(self) -> int:
        return sum(s.marks for s in self.slots)

    def bloom_profile(self) -> Tuple[BloomLevel, ...]:
        return tuple(s.bloom for s in self.slots)

    def type_profile(self) -> Tuple[str, ...]:
        return tuple(s.question_type for s in self.slots)


@dataclass
class ORPair:
    module_id    : int
    signature    : StructuralSignature
    profile      : AlternativeEquivalenceProfile
    alternatives : List[Alternative]   # always exactly 2

    def __post_init__(self):
        assert len(self.alternatives) == 2, (
            "An OR pair must have exactly 2 alternatives"
        )


@dataclass
class EvidenceLedgerEntry:
    """Provenance record for every generated question."""
    question_id     : str
    slot_id         : str
    module_id       : int
    concept         : str
    source_pages    : List[int]
    chunk_ids       : List[str]
    grounding_score : float
    operation       : str
    bloom           : BloomLevel
    solver_answer   : Optional[Dict[str, Any]]
    visual_used     : bool
    model           : str
    seed            : int
    timestamp       : str
