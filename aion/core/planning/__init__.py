"""
AION v2 Evidence Planning & Grounding Package
=============================================
Provides deterministic evidence selection, grounding verification, and model serialization:
- EvidencePlanner: Assembles archetype-constrained EvidenceBundles from compiled KnowledgeUnits.
- GroundingVerifier: Enforces the 6-Level Grounding Firewall and source-bound draft entity checks.
- CapabilitySerializer: Adapts EvidenceBundles for text-only vs multimodal generative models.
"""

from aion.core.planning.contracts import (
    EvidenceBundle,
    EvidenceItemRef,
    EvidenceRelationStrength,
    EvidenceRequirements,
    EvidenceRole,
    GroundingReport,
    GroundingViolation,
    GroundingViolationType,
    InsufficientEvidenceError,
)
from aion.core.planning.evidence_planner import (
    ARCHETYPE_REQUIREMENTS,
    EvidencePlanner,
)
from aion.core.planning.grounding_verifier import GroundingVerifier
from aion.core.planning.serializer import CapabilitySerializer

__all__ = [
    "EvidenceRelationStrength",
    "EvidenceRole",
    "EvidenceItemRef",
    "EvidenceRequirements",
    "EvidenceBundle",
    "GroundingViolationType",
    "GroundingViolation",
    "GroundingReport",
    "InsufficientEvidenceError",
    "ARCHETYPE_REQUIREMENTS",
    "EvidencePlanner",
    "GroundingVerifier",
    "CapabilitySerializer",
]
