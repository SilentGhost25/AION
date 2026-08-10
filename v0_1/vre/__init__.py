"""
AION Visual Reasoning Engine (VRE) Package
==========================================
Clean, typed, guarded Visual Reasoning Engine for AION.
"""

from .contracts import (
    BloomLevel, ConfidenceMetrics, FigureClassification, FigureExtractionResult,
    FigureInput, MutationRule, OperationChain, OperationStep, ProvenanceRecord,
    QuantityType, QuestionPlan, RenderMode, VKO, VQG, VREDecision, VREDecisionState, VREOutput,
    VRERequest
)
from .engine import VREEngine
from .errors import (
    CriticRejectionError, FSCClassificationError, ProvenanceError, QualityGateFailure,
    RenderQAError, SolverError, VKOValidationError, VREError
)
from .figure_quality import FigureQualityGate
from .fsc import FSC
from .gg import GG
from .npe import NPE
from .paper_visual_validator import PaperVisualValidator
from .provenance import ProvenanceTracker
from .qpvde import QPVDE
from .quantity_parser import QuantityParser
from .retry_policy import VRERetryController
from .semantic_validator import SemanticQuestionValidator
from .solvers import (
    BeamSolver, CircuitSolver, GenericSolver, GraphSolver, TreeSolver, get_solver, solve_vko
)
from .taxonomy import (
    BLOOM_LEVEL_VISUAL_TENDENCY, CONCEPT_VISUAL_DEPENDENCY, GROUNDING_VOCABULARY,
    HOT_TAXONOMY, SUPPORTED_FIGURE_CLASSES
)
from .vc import VisualCritic
from .vkoc import VKOC
from .vko_validator import VKOValidator
from .vqg import VQGBuilder
from .vqgr import VQGR

__all__ = [
    "VREEngine",
    "FSC",
    "VKOC",
    "VQGBuilder",
    "QPVDE",
    "NPE",
    "GG",
    "VisualCritic",
    "VQGR",
    "FigureQualityGate",
    "VKOValidator",
    "QuantityParser",
    "ProvenanceTracker",
    "SemanticQuestionValidator",
    "PaperVisualValidator",
    "VRERetryController",
    "VREDecisionState",
    "RenderMode",
    "VRERequest",
    "VREOutput",
    "VREDecision",
    "FigureInput",
    "FigureExtractionResult",
    "FigureClassification",
    "VKO",
    "OperationChain",
    "QuestionPlan",
    "ProvenanceRecord",
    "ConfidenceMetrics",
    "HOT_TAXONOMY",
    "CONCEPT_VISUAL_DEPENDENCY",
    "GROUNDING_VOCABULARY",
    "SUPPORTED_FIGURE_CLASSES",
    "GraphSolver",
    "TreeSolver",
    "CircuitSolver",
    "BeamSolver",
    "GenericSolver",
    "get_solver",
    "solve_vko",
    "VREError",
    "QualityGateFailure",
    "FSCClassificationError",
    "VKOValidationError",
    "SolverError",
    "CriticRejectionError",
    "RenderQAError",
    "ProvenanceError",
]
