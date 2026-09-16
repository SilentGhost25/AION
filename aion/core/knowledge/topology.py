"""
AION v2 Document Topology Classifier
====================================
Analyzes structural features of the DocumentDOM to distinguish document archetypes:
- TEXTBOOK: Dense hierarchical chapters, deep subsections, high word density.
- SLIDE_DECK: Per-slide titles, bulleted items, visual aids, low word density.
- LECTURE_NOTES: Mixed format, concise outlines, moderate density.

ARCHITECTURAL INVARIANT:
The topology classifier determines document structural layout.
Slide titles are mapped to SectionNodes, NEVER to ModuleNodes!
Module boundaries are strictly assessment constraints, not typographic artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from aion.core.dom.artifacts import ArtifactType, HeadingArtifact, TextArtifact, FigureArtifact
from aion.core.dom.document_dom import DocumentDOM


class DocumentTopology(str, Enum):
    TEXTBOOK = "TEXTBOOK"
    SLIDE_DECK = "SLIDE_DECK"
    LECTURE_NOTES = "LECTURE_NOTES"


@dataclass
class TopologyAnalysis:
    """
    Observable diagnostic result of topology analysis.
    """
    topology: DocumentTopology
    confidence: float
    recommended_sectioning: str           # "SLIDE_BASED" | "HIERARCHICAL"
    avg_words_per_page: float
    heading_to_page_ratio: float
    figure_to_page_ratio: float
    total_pages: int
    total_headings: int
    total_figures: int
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology": self.topology.value,
            "confidence": round(self.confidence, 3),
            "recommended_sectioning": self.recommended_sectioning,
            "avg_words_per_page": round(self.avg_words_per_page, 1),
            "heading_to_page_ratio": round(self.heading_to_page_ratio, 3),
            "figure_to_page_ratio": round(self.figure_to_page_ratio, 3),
            "total_pages": self.total_pages,
            "total_headings": self.total_headings,
            "total_figures": self.total_figures,
            "rationale": self.rationale,
        }


class DocumentTopologyClassifier:
    """
    Determines document archetype using non-destructive structural metrics.
    """

    @classmethod
    def classify(cls, dom: DocumentDOM) -> TopologyAnalysis:
        pages = max(dom.total_pages, 1)

        # Collect metrics from DOM registry
        all_artifacts = list(dom.registry.all())
        text_artifacts = [a for a in all_artifacts if isinstance(a, TextArtifact)]
        heading_artifacts = [a for a in all_artifacts if isinstance(a, HeadingArtifact)]
        figure_artifacts = [a for a in all_artifacts if isinstance(a, FigureArtifact)]

        total_words = 0
        for t in text_artifacts:
            words = (t.normalized_text or t.raw_text).split()
            total_words += len(words)

        avg_words_per_page = total_words / pages
        heading_ratio = len(heading_artifacts) / pages
        figure_ratio = len(figure_artifacts) / pages

        # Decision Logic:
        # 1. Slide decks typically have ~1 heading per page (0.6 - 1.5), and low words per page (< 180).
        # 2. Textbooks have high word count (> 300 words/page) and low heading-to-page ratio (< 0.5).
        # 3. Lecture notes fall in between.

        if avg_words_per_page < 180 and heading_ratio >= 0.5:
            topology = DocumentTopology.SLIDE_DECK
            confidence = min(0.95, 0.70 + (0.25 if heading_ratio >= 0.7 else 0.10))
            recommended_sectioning = "SLIDE_BASED"
            rationale = (
                f"Low word density ({avg_words_per_page:.1f} w/p) and high heading ratio "
                f"({heading_ratio:.2f} h/p) indicate a presentation slide deck."
            )
        elif avg_words_per_page >= 300 and heading_ratio < 0.55:
            topology = DocumentTopology.TEXTBOOK
            confidence = min(0.95, 0.75 + (0.20 if avg_words_per_page > 450 else 0.10))
            recommended_sectioning = "HIERARCHICAL"
            rationale = (
                f"High word density ({avg_words_per_page:.1f} w/p) with sparse structural headings "
                f"({heading_ratio:.2f} h/p) indicates dense textbook material."
            )
        else:
            topology = DocumentTopology.LECTURE_NOTES
            confidence = 0.80
            recommended_sectioning = "HIERARCHICAL"
            rationale = (
                f"Moderate word density ({avg_words_per_page:.1f} w/p) and heading distribution "
                f"({heading_ratio:.2f} h/p) characteristic of academic lecture notes."
            )

        return TopologyAnalysis(
            topology=topology,
            confidence=confidence,
            recommended_sectioning=recommended_sectioning,
            avg_words_per_page=avg_words_per_page,
            heading_to_page_ratio=heading_ratio,
            figure_to_page_ratio=figure_ratio,
            total_pages=pages,
            total_headings=len(heading_artifacts),
            total_figures=len(figure_artifacts),
            rationale=rationale,
        )
