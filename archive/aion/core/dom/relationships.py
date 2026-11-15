"""
AION v2 Relationship Graph & Extensible Reference Linkers
=========================================================
Preserves the semantic and structural web connecting artifacts across modalities.
Supports deterministic multi-strategy linking: explicit references, section
containment, and spatial proximity fallback.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from aion.core.dom.artifacts import (
    ArtifactType,
    BaseArtifact,
    FigureArtifact,
    EquationArtifact,
    TableArtifact,
    AlgorithmArtifact,
    TextArtifact,
    CaptionArtifact,
)


class RelationshipType(str, Enum):
    REFERENCES = "REFERENCES"       # Text explicitly references a figure, equation, or table
    CONTAINED_IN = "CONTAINED_IN"   # Artifact is structurally contained within a section/module
    GOVERNS = "GOVERNS"             # Equation or law governs a concept or procedure
    ILLUSTRATES = "ILLUSTRATES"     # Figure or diagram illustrates a concept or text
    EXEMPLIFIES = "EXEMPLIFIES"     # Worked example demonstrates an equation or concept
    DERIVED_FROM = "DERIVED_FROM"   # Equation is mathematically derived from parent equation
    PROXIMAL_TO = "PROXIMAL_TO"     # Spatial proximity heuristic on the same page


@dataclass
class Relationship:
    """
    Directed edge in the academic relationship graph.
    """
    source_id: str
    target_id: str
    rel_type: RelationshipType
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "rel_type": self.rel_type.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Relationship:
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            rel_type=RelationshipType(data["rel_type"]),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


class RelationshipGraph:
    """
    Bidirectional in-memory graph connecting document artifacts.
    """

    def __init__(self):
        self._outgoing: Dict[str, List[Relationship]] = {}
        self._incoming: Dict[str, List[Relationship]] = {}

    def add_relationship(self, rel: Relationship) -> None:
        """Add a directed relationship between artifacts."""
        self._outgoing.setdefault(rel.source_id, []).append(rel)
        self._incoming.setdefault(rel.target_id, []).append(rel)

    def get_outgoing(
        self, source_id: str, rel_type: Optional[RelationshipType] = None
    ) -> List[Relationship]:
        """Get all relationships originating from source_id."""
        edges = self._outgoing.get(source_id, [])
        if rel_type is None:
            return list(edges)
        return [e for e in edges if e.rel_type == rel_type]

    def get_incoming(
        self, target_id: str, rel_type: Optional[RelationshipType] = None
    ) -> List[Relationship]:
        """Get all relationships pointing to target_id."""
        edges = self._incoming.get(target_id, [])
        if rel_type is None:
            return list(edges)
        return [e for e in edges if e.rel_type == rel_type]

    def get_related_artifact_ids(
        self, artifact_id: str, rel_type: Optional[RelationshipType] = None
    ) -> Set[str]:
        """Return all unique artifact IDs directly connected to artifact_id in either direction."""
        res: Set[str] = set()
        for edge in self.get_outgoing(artifact_id, rel_type):
            res.add(edge.target_id)
        for edge in self.get_incoming(artifact_id, rel_type):
            res.add(edge.source_id)
        return res

    def to_dict(self) -> Dict[str, Any]:
        all_edges = []
        for edges in self._outgoing.values():
            all_edges.extend([e.to_dict() for e in edges])
        return {"edges": all_edges}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RelationshipGraph:
        graph = cls()
        for e_dict in data.get("edges", []):
            graph.add_relationship(Relationship.from_dict(e_dict))
        return graph


# ---------------------------------------------------------------------------
# Extensible Linking Pipeline
# ---------------------------------------------------------------------------

class BaseReferenceLinker(ABC):
    """Abstract interface for artifact relationship linkers."""

    @abstractmethod
    def link(self, artifacts: List[BaseArtifact], graph: RelationshipGraph) -> int:
        """
        Analyze artifacts and populate edges into graph.
        Returns the count of relationships added.
        """
        pass


class ExplicitReferenceLinker(BaseReferenceLinker):
    """
    Tier 1 Linker: Explicit symbolic references.
    Finds textual references like 'Figure 3.4', 'Fig. 2', 'Eq. (4.1)', 'Table 1'
    and matches them against registered captions, labels, or numbering.
    """

    FIG_PATTERN = re.compile(r'(?i)\b(?:fig(?:ure)?\.?)\s*(\d+(?:[.\-_]\d+)?[a-z]?)', re.UNICODE)
    EQ_PATTERN = re.compile(r'(?i)\b(?:eq(?:uation)?\.?)\s*(?:\(?(\d+(?:[.\-_]\d+)?[a-z]?)\)?)', re.UNICODE)
    TBL_PATTERN = re.compile(r'(?i)\b(?:table)\s*(\d+(?:[.\-_]\d+)?[a-z]?)', re.UNICODE)
    ALG_PATTERN = re.compile(r'(?i)\b(?:alg(?:orithm)?\.?)\s*(\d+(?:[.\-_]\d+)?[a-z]?)', re.UNICODE)

    def link(self, artifacts: List[BaseArtifact], graph: RelationshipGraph) -> int:
        added = 0
        figures_by_label: Dict[str, List[FigureArtifact]] = {}
        equations_by_label: Dict[str, List[EquationArtifact]] = {}
        tables_by_label: Dict[str, List[TableArtifact]] = {}
        algorithms_by_label: Dict[str, List[AlgorithmArtifact]] = {}

        # Index target artifacts by their label / numbering
        for a in artifacts:
            if isinstance(a, FigureArtifact):
                for match in self.FIG_PATTERN.findall(a.caption or ""):
                    norm = self._norm_label(match)
                    figures_by_label.setdefault(norm, []).append(a)
            elif isinstance(a, EquationArtifact):
                if a.equation_label:
                    norm = self._norm_label(a.equation_label)
                    equations_by_label.setdefault(norm, []).append(a)
            elif isinstance(a, TableArtifact):
                for match in self.TBL_PATTERN.findall(a.caption or ""):
                    norm = self._norm_label(match)
                    tables_by_label.setdefault(norm, []).append(a)
            elif isinstance(a, AlgorithmArtifact):
                for match in self.ALG_PATTERN.findall(a.title or ""):
                    norm = self._norm_label(match)
                    algorithms_by_label.setdefault(norm, []).append(a)

        # Match text blocks
        for a in artifacts:
            if not isinstance(a, TextArtifact):
                continue
            text = a.normalized_text or a.raw_text

            # Figures
            for match in self.FIG_PATTERN.findall(text):
                norm = self._norm_label(match)
                for fig in figures_by_label.get(norm, []):
                    graph.add_relationship(Relationship(
                        source_id=a.artifact_id,
                        target_id=fig.artifact_id,
                        rel_type=RelationshipType.REFERENCES,
                        confidence=0.98,
                        metadata={"matched_label": norm, "strategy": "explicit_regex"}
                    ))
                    if fig.artifact_id not in a.referenced_artifact_ids:
                        a.referenced_artifact_ids.append(fig.artifact_id)
                    added += 1

            # Equations
            for match in self.EQ_PATTERN.findall(text):
                norm = self._norm_label(match)
                for eq in equations_by_label.get(norm, []):
                    graph.add_relationship(Relationship(
                        source_id=a.artifact_id,
                        target_id=eq.artifact_id,
                        rel_type=RelationshipType.REFERENCES,
                        confidence=0.95,
                        metadata={"matched_label": norm, "strategy": "explicit_regex"}
                    ))
                    if eq.artifact_id not in a.referenced_artifact_ids:
                        a.referenced_artifact_ids.append(eq.artifact_id)
                    added += 1

            # Tables
            for match in self.TBL_PATTERN.findall(text):
                norm = self._norm_label(match)
                for tbl in tables_by_label.get(norm, []):
                    graph.add_relationship(Relationship(
                        source_id=a.artifact_id,
                        target_id=tbl.artifact_id,
                        rel_type=RelationshipType.REFERENCES,
                        confidence=0.95,
                        metadata={"matched_label": norm, "strategy": "explicit_regex"}
                    ))
                    if tbl.artifact_id not in a.referenced_artifact_ids:
                        a.referenced_artifact_ids.append(tbl.artifact_id)
                    added += 1

        return added

    def _norm_label(self, raw: str) -> str:
        s = raw.strip("()[] \t\r\n").lower()
        return re.sub(r"[-_]", ".", s)


class SectionContainmentLinker(BaseReferenceLinker):
    """
    Tier 2 Linker: Structural enclosure.
    Connects every artifact to its enclosing section.
    """

    def link(self, artifacts: List[BaseArtifact], graph: RelationshipGraph) -> int:
        added = 0
        for a in artifacts:
            if a.section_id:
                graph.add_relationship(Relationship(
                    source_id=a.artifact_id,
                    target_id=a.section_id,
                    rel_type=RelationshipType.CONTAINED_IN,
                    confidence=1.0,
                    metadata={"module_index": a.module_index, "strategy": "section_scoping"}
                ))
                added += 1
        return added


class SpatialProximityLinker(BaseReferenceLinker):
    """
    Tier 3 Linker: Spatial geometric proximity fallback.
    For unreferenced figures or unnumbered display equations, links to the
    closest preceding or subsequent TextArtifact on the same page.
    """

    def __init__(self, max_vertical_gap_pts: float = 180.0):
        self.max_gap = max_vertical_gap_pts

    def link(self, artifacts: List[BaseArtifact], graph: RelationshipGraph) -> int:
        added = 0
        by_page: Dict[int, List[BaseArtifact]] = {}
        for a in artifacts:
            if a.source_bbox is not None:
                by_page.setdefault(a.page_number, []).append(a)

        for page, page_artifacts in by_page.items():
            texts = [a for a in page_artifacts if isinstance(a, TextArtifact) and a.source_bbox]
            targets = [
                a for a in page_artifacts
                if isinstance(a, (FigureArtifact, EquationArtifact, TableArtifact))
                and a.source_bbox
            ]

            for tgt in targets:
                # Check if target already has incoming REFERENCES
                if graph.get_incoming(tgt.artifact_id, RelationshipType.REFERENCES):
                    continue

                best_text: Optional[TextArtifact] = None
                min_dist = float("inf")
                tgt_y0, tgt_y1 = tgt.source_bbox[1], tgt.source_bbox[3]

                for txt in texts:
                    txt_y0, txt_y1 = txt.source_bbox[1], txt.source_bbox[3]
                    # Vertical gap
                    if txt_y1 <= tgt_y0:
                        dist = tgt_y0 - txt_y1  # text above target
                    elif tgt_y1 <= txt_y0:
                        dist = txt_y0 - tgt_y1  # text below target
                    else:
                        dist = 0.0              # overlapping bounding boxes

                    if dist < min_dist and dist <= self.max_gap:
                        min_dist = dist
                        best_text = txt

                if best_text:
                    conf = max(0.60, 0.90 - (min_dist / self.max_gap) * 0.30)
                    graph.add_relationship(Relationship(
                        source_id=best_text.artifact_id,
                        target_id=tgt.artifact_id,
                        rel_type=RelationshipType.PROXIMAL_TO,
                        confidence=round(conf, 3),
                        metadata={"distance_pts": round(min_dist, 1), "strategy": "spatial_proximity"}
                    ))
                    if tgt.artifact_id not in best_text.referenced_artifact_ids:
                        best_text.referenced_artifact_ids.append(tgt.artifact_id)
                    added += 1

        return added


class DeterministicReferenceLinker(BaseReferenceLinker):
    """
    Composite Linker chaining Tier 1 (Explicit), Tier 2 (Containment), and Tier 3 (Proximity).
    """

    def __init__(self, linkers: Optional[List[BaseReferenceLinker]] = None):
        self.linkers = linkers or [
            ExplicitReferenceLinker(),
            SectionContainmentLinker(),
            SpatialProximityLinker(),
        ]

    def link(self, artifacts: List[BaseArtifact], graph: RelationshipGraph) -> int:
        total = 0
        for linker in self.linkers:
            total += linker.link(artifacts, graph)
        return total
