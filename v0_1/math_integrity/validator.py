"""
AION Math Integrity Architecture — Math Validator
===================================================
Executes V01-V07 validation checks enforcing canonical hash integrity,
delimiter balance, UTF-8 clean encoding, and M3 replacement character checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from .contracts import MathArtifact, MathIntegrityViolation, MathValidationStatus
from .normalizer import MathNormalizer


@dataclass
class MathValidationReport:
    status: MathValidationStatus
    score: float
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == MathValidationStatus.VALID and self.score >= 0.70


class MathValidator:
    """Math Integrity Validator executing V01-V07 checks."""

    @classmethod
    def validate(cls, artifact: MathArtifact) -> MathValidationReport:
        errors: List[str] = []

        # V01 — M3 Invariant (U+FFFD Replacement Character)
        for val in (artifact.latex, artifact.normalized_latex, artifact.source_text):
            if val and "\ufffd" in val:
                return MathValidationReport(
                    status=MathValidationStatus.CORRUPT,
                    score=0.0,
                    errors=["V01: Unicode replacement character (U+FFFD) detected (M3 Violation)"],
                )

        # V06 — Non-Empty Check
        if not artifact.latex or not artifact.latex.strip():
            return MathValidationReport(
                status=MathValidationStatus.CORRUPT,
                score=0.0,
                errors=["V06: LaTeX string is empty"],
            )

        # V07 — UTF-8 Encoding Check
        try:
            artifact.latex.encode("utf-8")
        except UnicodeEncodeError as e:
            return MathValidationReport(
                status=MathValidationStatus.CORRUPT,
                score=0.0,
                errors=[f"V07: UTF-8 encoding failure: {e}"],
            )

        # V02 — Canonical Hash Integrity Check
        if not artifact.verify_canonical_hash():
            return MathValidationReport(
                status=MathValidationStatus.CORRUPT,
                score=0.0,
                errors=["V02: Canonical hash verification failed"],
            )

        # V03 — Delimiter Balance Check
        balanced = MathNormalizer.check_delimiter_balance(artifact.normalized_latex)
        if not balanced:
            errors.append("V03: Unbalanced LaTeX delimiters")

        # Score calculation
        score = 1.0 if not errors else 0.60
        status = MathValidationStatus.VALID if score >= 0.70 else MathValidationStatus.TRUNCATED

        return MathValidationReport(status=status, score=score, errors=errors)
