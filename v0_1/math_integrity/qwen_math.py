"""
AION Math Integrity Architecture — Qwen Math Interface
======================================================
Enforces invariant M4: Qwen receives and outputs [MATH:eq_...] placeholders only;
it never serializes or mutates raw mathematical expressions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from .contracts import MathArtifact, ProtectedTextEnvelope


class QwenMathInterface:
    """Qwen Math Interface maintaining M4 Placeholder Invariant."""

    SYSTEM_MATH_INSTRUCTIONS = (
        "Mathematical equations are provided as artifact references.\n"
        "Use [MATH:eq_...] placeholders exactly as shown.\n"
        "Do NOT rewrite equations in Unicode or ASCII approximations.\n"
        "Do NOT modify variable names, operators, or constants.\n"
        "Insert the placeholder [MATH:eq_...] exactly where the equation should appear."
    )

    @classmethod
    def prepare_context(cls, envelope: ProtectedTextEnvelope) -> Dict[str, Any]:
        """Prepare LLM generation context with protected math placeholders."""
        artifacts_meta = {}
        for placeholder, artifact in envelope.artifacts.items():
            artifacts_meta[artifact.math_id] = {
                "math_id": artifact.math_id,
                "placeholder": placeholder,
                "latex": artifact.latex,
                "unicode": artifact.unicode_text,
                "description": f"Math equation artifact {artifact.math_id}",
            }

        return {
            "text": envelope.text,
            "math_instructions": cls.SYSTEM_MATH_INSTRUCTIONS,
            "math_artifacts": artifacts_meta,
        }

    @classmethod
    def validate_response(cls, response_text: str, expected_placeholders: List[str]) -> bool:
        """Verify that Qwen preserved all required [MATH:eq_...] placeholders."""
        for ph in expected_placeholders:
            if ph not in response_text:
                return False
        return True

    @classmethod
    def restore_math(cls, response_text: str, envelope: ProtectedTextEnvelope) -> str:
        """Replace placeholders in Qwen response with canonical LaTeX math for rendering."""
        result = response_text
        for placeholder, artifact in envelope.artifacts.items():
            result = result.replace(placeholder, artifact.best_for_display())
        return result
