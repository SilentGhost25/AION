"""
AION Math Integrity Architecture — Math Boundary Guard
======================================================
Extracts mathematical expressions from prose before LLM generation,
replacing them with [MATH:eq_...] placeholders.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple
from .contracts import MathArtifact, MathIntegrityViolation, ProtectedTextEnvelope
from .encoding_guard import EncodingInvariantGuard
from .normalizer import MathNormalizer


MATH_BOUNDARY_PATTERNS = [
    # Display Math
    r'\\\[.*?\\\]',
    r'\$\$.*?\$\$',
    r'\\begin\{equation\}.*?\\end\{equation\}',
    r'\\begin\{align\}.*?\\end\{align\}',
    # Inline Math
    r'\\\(.*?\\\)',
    r'(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)',
    # Unicode Math Sequences
    r'[α-ωΑ-Ωµ][=\+\-\*/×÷±\^_\(\)\[\]0-9a-zA-Z\s\.]*',
]


class MathBoundaryGuard:
    """Math Boundary Guard protecting prose from stochastic math mutation."""

    @classmethod
    def protect(
        cls,
        text: str,
        document_id: str = "doc_001",
        page: int = 1,
    ) -> ProtectedTextEnvelope:
        """Extract math expressions into placeholders and create MathArtifacts."""
        # Step 1 — M3 Invariant Check
        EncodingInvariantGuard.assert_clean(text, "MathBoundaryGuard.protect")

        spans: List[Tuple[int, int, str]] = []
        for pattern in MATH_BOUNDARY_PATTERNS:
            for match in re.finditer(pattern, text, re.DOTALL):
                spans.append((match.start(), match.end(), match.group()))

        # Sort and deduplicate overlapping spans
        spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
        merged_spans: List[Tuple[int, int, str]] = []
        last_end = -1
        for start, end, match_str in spans:
            if start >= last_end:
                merged_spans.append((start, end, match_str))
                last_end = end

        protected_text = text
        artifacts: Dict[str, MathArtifact] = {}
        offset_counter = 1

        # Process in reverse order to preserve string indices
        for start, end, raw_math in reversed(merged_spans):
            math_id = f"eq_{document_id}_{page}_{offset_counter:03d}"
            placeholder = f"[MATH:{math_id}]"

            artifact = MathNormalizer.normalize(
                raw_text=raw_math,
                math_id=math_id,
                document_id=document_id,
                page=page,
            )

            artifacts[placeholder] = artifact
            protected_text = protected_text[:start] + placeholder + protected_text[end:]
            offset_counter += 1

        return ProtectedTextEnvelope(
            text=protected_text,
            artifacts=artifacts,
            original=text,
            document_id=document_id,
        )

    @classmethod
    def restore(cls, envelope: ProtectedTextEnvelope) -> str:
        """Re-insert canonical LaTeX math representations into text for rendering."""
        return envelope.restore()
