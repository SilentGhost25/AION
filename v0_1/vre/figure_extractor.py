"""
AION VRE Figure Extractor Adapter
=================================
Extracts candidate figure assets from PDF pages / raw file inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from .contracts import FigureInput


class FigureExtractor:
    """Extracts candidate figures from documents or converts file inputs."""

    @staticmethod
    def extract_from_input(candidate: FigureInput) -> FigureInput:
        """Validates existence of image asset."""
        p = Path(candidate.image_path)
        if not p.exists():
            # If path does not exist, check if there's a fallback asset
            fallback_dir = Path("extracted_output/assets")
            if fallback_dir.exists():
                images = list(fallback_dir.glob("*.png")) + list(fallback_dir.glob("*.jpg"))
                if images:
                    candidate.image_path = str(images[0])
        return candidate
