"""
AION Structural Architecture v2
===============================
Deterministic Structural Signatures, Mark Distribution Engine (MDE),
Joint Bloom x Mark Constraint Solver (JBMCS), OR Pair Equivalence Builder,
SeedManager, ContentRandomizer, QwenAdapter, and 38-Check PaperContractVerifier.
"""

from .contracts import (
    BloomLevel,
    DistributionPolicy,
    VisualPrior,
    VisualDecision,
    SlotStatus,
    DifficultyBand,
    RecoveryAction,
    StructuralSignature,
    AlternativeEquivalenceProfile,
    QuestionSlot,
    Alternative,
    ORPair,
    EvidenceLedgerEntry,
)
from .mde import MarkDistributionEngine
from .jbmcs import JointBloomMarkConstraintSolver
from .equivalence import ORPairEquivalenceBuilder, DifficultyResolver, DomainQuestionTypeMatrix
from .visual_pipeline import VisualDecisionPipeline
from .seed_manager import SeedManager
from .content_randomizer import ContentRandomizer
from .qwen_adapter import QwenAdapter, build_context
from .verifier import PaperContractVerifier

__all__ = [
    "BloomLevel",
    "DistributionPolicy",
    "VisualPrior",
    "VisualDecision",
    "SlotStatus",
    "DifficultyBand",
    "RecoveryAction",
    "StructuralSignature",
    "AlternativeEquivalenceProfile",
    "QuestionSlot",
    "Alternative",
    "ORPair",
    "EvidenceLedgerEntry",
    "MarkDistributionEngine",
    "JointBloomMarkConstraintSolver",
    "ORPairEquivalenceBuilder",
    "DifficultyResolver",
    "DomainQuestionTypeMatrix",
    "VisualDecisionPipeline",
    "SeedManager",
    "ContentRandomizer",
    "QwenAdapter",
    "build_context",
    "PaperContractVerifier",
]
