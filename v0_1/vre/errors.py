"""
AION VRE Exceptions
===================
Hierarchical error types for the Visual Reasoning Engine.
"""

class VREError(Exception):
    """Base exception for all VRE errors."""
    pass


class QualityGateFailure(VREError):
    """Raised when candidate figure fails quality gate checks."""
    pass


class FSCClassificationError(VREError):
    """Raised when figure semantic classification fails."""
    pass


class VKOValidationError(VREError):
    """Raised when VKO fails structural integrity verification."""
    pass


class SolverError(VREError):
    """Raised when deterministic domain solver fails or finds no solution."""
    pass


class CriticRejectionError(VREError):
    """Raised when question fails Visual Critic (MCRS) criteria."""
    pass


class RenderQAError(VREError):
    """Raised when SVG render QA validation fails."""
    pass


class ProvenanceError(VREError):
    """Raised when provenance record is missing or corrupt."""
    pass
