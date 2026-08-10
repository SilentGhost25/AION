"""
AION Math Integrity Architecture
================================
Guarantees foundational invariants M1-M5:
  M1: Math is an object, not a string (MathArtifact).
  M2: LaTeX is the canonical representation (Unicode -> LaTeX -> AST).
  M3: Unicode replacement character (U+FFFD) is a fatal violation.
  M4: Qwen references equations via placeholders [MATH:eq_...], never serializes them.
  M5: Every equation must pass round-trip validation.
"""

from .contracts import (
    MathRepresentation,
    MathSourceType,
    MathValidationStatus,
    EquationType,
    HealerAction,
    MathSymbol,
    MathNode,
    ExpressionAST,
    MathArtifact,
    ProtectedTextEnvelope,
    MathIntegrityViolation,
)
from .registry import MATH_SYMBOL_REGISTRY, unicode_to_latex, latex_to_unicode
from .encoding_guard import EncodingInvariantGuard
from .boundary_guard import MathBoundaryGuard
from .normalizer import MathNormalizer
from .validator import MathValidator, MathValidationReport
from .healer import MathHealer, HealingFailure
from .equivalence import SymbolicEquivalenceChecker
from .renderer import MathRenderer, RenderedMath, RenderFormat
from .qwen_math import QwenMathInterface

__all__ = [
    "MathRepresentation",
    "MathSourceType",
    "MathValidationStatus",
    "EquationType",
    "HealerAction",
    "MathSymbol",
    "MathNode",
    "ExpressionAST",
    "MathArtifact",
    "ProtectedTextEnvelope",
    "MathIntegrityViolation",
    "MATH_SYMBOL_REGISTRY",
    "unicode_to_latex",
    "latex_to_unicode",
    "EncodingInvariantGuard",
    "MathBoundaryGuard",
    "MathNormalizer",
    "MathValidator",
    "MathValidationReport",
    "MathHealer",
    "HealingFailure",
    "SymbolicEquivalenceChecker",
    "MathRenderer",
    "RenderedMath",
    "RenderFormat",
    "QwenMathInterface",
]
