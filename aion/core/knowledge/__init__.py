"""
AION v2 Knowledge Compilation Subsystem
=======================================
Transforms the immutable multimodal DocumentDOM into a queryable semantic index
and coverage ledger for syllabus-faithful examination generation.
"""

from aion.core.knowledge.contracts import (
    CognitiveDemand,
    SemanticType,
    SemanticCandidate,
    BindingType,
    FigureContextBinding,
    ImportanceScore,
    SyllabusScopeMask,
    KnowledgeUnit,
)
from aion.core.knowledge.topology import (
    DocumentTopology,
    DocumentTopologyClassifier,
    TopologyAnalysis,
)
from aion.core.knowledge.compiler import (
    KnowledgeCompiler,
    KnowledgeCompilationResult,
    SemanticPatternDetector,
)
from aion.core.knowledge.ledger import (
    CoverageLedger,
    AssessmentRecord,
    KUAssessmentState,
)

__all__ = [
    "CognitiveDemand",
    "SemanticType",
    "SemanticCandidate",
    "BindingType",
    "FigureContextBinding",
    "ImportanceScore",
    "SyllabusScopeMask",
    "KnowledgeUnit",
    "DocumentTopology",
    "DocumentTopologyClassifier",
    "TopologyAnalysis",
    "KnowledgeCompiler",
    "KnowledgeCompilationResult",
    "SemanticPatternDetector",
    "CoverageLedger",
    "AssessmentRecord",
    "KUAssessmentState",
]
