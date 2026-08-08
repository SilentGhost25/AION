"""
AION v2 Validators Package
==========================
Content and Academic quality validation gates.
"""

from .academic_validator import validate_academic_quality, AcademicValidationResult

__all__ = ["validate_academic_quality", "AcademicValidationResult"]
