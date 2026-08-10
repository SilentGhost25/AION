"""
AION Math Integrity Architecture — Math Healer
===============================================
Deterministic repair decision tree for LaTeX symbols, encoding, and delimiters.
Strictly forbids guessing content or repairing U+FFFD replacement characters.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Union
from .contracts import HealerAction, MathArtifact, MathIntegrityViolation, MathValidationStatus
from .normalizer import MathNormalizer
from .validator import MathValidator


@dataclass
class HealingFailure:
    action: HealerAction
    math_id: str
    message: str


class MathHealer:
    """Math Healer implementing guarded deterministic repairs."""

    @classmethod
    def heal(cls, artifact: MathArtifact) -> Union[MathArtifact, HealingFailure]:
        """Attempt deterministic healing of a damaged MathArtifact."""
        # Hard Stop: M3 Replacement characters CANNOT be healed by guessing!
        for val in (artifact.latex, artifact.normalized_latex, artifact.source_text):
            if val and "\ufffd" in val:
                return HealingFailure(
                    action=HealerAction.BLOCKED,
                    math_id=artifact.math_id,
                    message="U+FFFD replacement character detected (M3 Violation) — guessing is strictly forbidden.",
                )

        current_latex = artifact.normalized_latex or artifact.latex

        # Repair 1 — Symbol Replacement
        repaired_latex = MathNormalizer.convert_unicode_to_latex(current_latex)

        # Repair 3 — Delimiter Balancing
        if not MathNormalizer.check_delimiter_balance(repaired_latex):
            if repaired_latex.startswith(r"\[") and not repaired_latex.endswith(r"\]"):
                repaired_latex = repaired_latex + r"\]"
            elif repaired_latex.startswith(r"\(") and not repaired_latex.endswith(r"\)"):
                repaired_latex = repaired_latex + r"\)"
            elif repaired_latex.count("$") % 2 != 0:
                repaired_latex = repaired_latex + "$"
            elif repaired_latex.count("{") > repaired_latex.count("}"):
                repaired_latex = repaired_latex + ("}" * (repaired_latex.count("{") - repaired_latex.count("}")))

        # Re-validate
        artifact.normalized_latex = repaired_latex
        artifact.canonical_hash = hashlib.sha256(repaired_latex.encode("utf-8")).hexdigest()

        report = MathValidator.validate(artifact)
        if report.is_valid:
            artifact.validation_status = MathValidationStatus.VALID
            return artifact
        else:
            return HealingFailure(
                action=HealerAction.BLOCKED,
                math_id=artifact.math_id,
                message=f"Healing failed: {report.errors}",
            )
