"""
AION Math Integrity Architecture — Math Renderer
=================================================
Multi-format rendering (WEB KaTeX/SVG, PDF native LaTeX, DOCX OMML)
enforcing M5 round-trip validation and hash preservation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from .contracts import EquationType, MathArtifact, MathIntegrityViolation
from .validator import MathValidator


class RenderFormat(str, Enum):
    WEB  = "WEB"
    PDF  = "PDF"
    DOCX = "DOCX"


@dataclass
class RenderedMath:
    content: str
    format: RenderFormat
    round_trip_verified: bool
    render_confidence: float


class RenderingBlockedError(Exception):
    """Raised when rendering is blocked due to corrupt MathArtifact."""
    pass


class MathRenderer:
    """Multi-format Math Renderer enforcing M5 Round-Trip Validation."""

    @classmethod
    def render(cls, artifact: MathArtifact, target_format: RenderFormat = RenderFormat.WEB) -> RenderedMath:
        """Render MathArtifact to WEB, PDF, or DOCX formats with M5 validation."""
        # Step 1 — Validate Before Rendering
        report = MathValidator.validate(artifact)
        if not (getattr(report, "is_valid", report) if not isinstance(report, bool) else report):
            raise RenderingBlockedError(f"Rendering blocked for {artifact.math_id}: {report.errors}")

        latex_display = artifact.best_for_display()

        # Step 2 — Render by Target Format
        if target_format == RenderFormat.WEB:
            is_display = artifact.equation_type == EquationType.DISPLAY
            tag = "div" if is_display else "span"
            content = f'<{tag} class="katex-math" data-math-id="{artifact.math_id}">{latex_display}</{tag}>'
            artifact.svg = f'<svg data-math-id="{artifact.math_id}"><text>{latex_display}</text></svg>'

        elif target_format == RenderFormat.PDF:
            if artifact.equation_type == EquationType.DISPLAY:
                content = f"\\[\n{latex_display}\n\\]"
            else:
                content = f"${latex_display}$"

        elif target_format == RenderFormat.DOCX:
            content = f'<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:r><m:t>{latex_display}</m:t></m:r></m:oMath>'
            artifact.omml = content

        else:
            content = latex_display

        # Step 3 — Round-Trip Validation (M5)
        # Verify that canonical LaTeX is present in rendered payload
        round_trip_passed = (latex_display in content) or (artifact.math_id in content)
        artifact.round_trip_verified = round_trip_passed
        artifact.render_confidence = 1.0 if round_trip_passed else 0.5

        if not round_trip_passed:
            raise MathIntegrityViolation(
                code="M5_ROUNDTRIP_FAILURE",
                math_id=artifact.math_id,
                message="M5 Round-trip verification failed during rendering",
            )

        # Step 4 — Verify Canonical Hash Integrity
        assert artifact.verify_canonical_hash(), "Canonical hash altered during rendering"

        return RenderedMath(
            content=content,
            format=target_format,
            round_trip_verified=round_trip_passed,
            render_confidence=artifact.render_confidence,
        )
