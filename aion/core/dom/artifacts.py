"""
AION v2 Typed Academic Artifacts
=================================
Defines the canonical, strongly typed multimodal artifact primitives.
Preserves every document element in its native semantic representation
without flattening into monolithic text chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ArtifactType(str, Enum):
    TEXT = "TEXT"
    HEADING = "HEADING"
    EQUATION = "EQUATION"
    FIGURE = "FIGURE"
    TABLE = "TABLE"
    ALGORITHM = "ALGORITHM"
    CAPTION = "CAPTION"
    LIST = "LIST"
    EXAMPLE = "EXAMPLE"
    DEFINITION = "DEFINITION"
    PROCEDURE = "PROCEDURE"
    CODE = "CODE"
    REFERENCE = "REFERENCE"


@dataclass
class BaseArtifact:
    """
    Base class for all canonical document artifacts.
    Provides immutable provenance, geometric bounds, and structural identity.
    """
    artifact_id: str                      # Canonical ID, e.g. "TXT-M1-P04-001", "FIG-M3-P42-002"
    artifact_type: ArtifactType
    module_index: int = 0                 # 1 to 5 (0 = unassigned / global)
    section_id: str = ""                  # ID of the containing SectionNode
    page_number: int = 1                  # 1-indexed source document page
    source_bbox: Optional[Tuple[float, float, float, float]] = None # (x0, y0, x1, y1) in PDF points
    confidence: float = 1.0               # Extraction confidence (0.0 - 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to serializable dictionary."""
        d = dict(self.__dict__)
        d["artifact_type"] = self.artifact_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaseArtifact:
        """Instantiate artifact from dictionary."""
        d = dict(data)
        d["artifact_type"] = ArtifactType(d["artifact_type"])
        if "source_bbox" in d and d["source_bbox"] is not None:
            d["source_bbox"] = tuple(d["source_bbox"])
        return cls(**d)


@dataclass
class TextArtifact(BaseArtifact):
    """
    Discrete body text paragraph or semantic block.
    Text is normalized strictly within its own boundary.
    """
    artifact_type: ArtifactType = ArtifactType.TEXT
    raw_text: str = ""
    normalized_text: str = ""
    reading_order: int = 0
    referenced_artifact_ids: List[str] = field(default_factory=list)


@dataclass
class HeadingArtifact(BaseArtifact):
    """
    Structural heading or chapter/module boundary.
    """
    artifact_type: ArtifactType = ArtifactType.HEADING
    title: str = ""
    level: int = 2                       # 1 = Module/Chapter, 2 = Section, 3 = Subsection
    numbering: Optional[str] = None      # e.g., "3.4", "Module 2", "Chapter 1"


@dataclass
class EquationArtifact(BaseArtifact):
    """
    Mathematical formula or symbolic expression.
    Preserves verified LaTeX and free variables.
    """
    artifact_type: ArtifactType = ArtifactType.EQUATION
    latex: str = ""
    variables: List[str] = field(default_factory=list) # e.g. ["V_1", "V_2", "N_1", "N_2"]
    is_inline: bool = False
    is_verified_syntax: bool = True
    equation_label: Optional[str] = None # e.g. "(3.1)", "(4.2a)"
    verification_status: str = "VERIFIED" # "VERIFIED" | "CANDIDATE" | "FAILED"



@dataclass
class FigureArtifact(BaseArtifact):
    """
    Visual graphic, schematic, circuit, graph, or architectural diagram.
    Asset path is NEVER absolute on the artifact; it uses an abstract asset_key
    resolved dynamically through ArtifactRegistry.
    """
    artifact_type: ArtifactType = ArtifactType.FIGURE
    asset_key: str = ""                  # Canonical registry key, e.g. "assets/figs/FIG-M3-002.png"
    image_hash: str = ""                 # SHA256 hex digest for integrity and deduplication
    caption: str = ""
    semantic_description: str = ""        # Visual semantic summary extracted during ingestion
    figure_type: str = "DIAGRAM"         # SCHEMATIC, CIRCUIT, GRAPH, PLOT, ARCHITECTURE, FLOWCHART
    width: int = 0
    height: int = 0
    dpi: int = 150


@dataclass
class TableArtifact(BaseArtifact):
    """
    Tabular data structure.
    Preserves structured rows, headers, and fast markdown representation.
    """
    artifact_type: ArtifactType = ArtifactType.TABLE
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    markdown_repr: str = ""
    caption: str = ""


@dataclass
class AlgorithmArtifact(BaseArtifact):
    """
    Algorithmic procedure or pseudocode block.
    """
    artifact_type: ArtifactType = ArtifactType.ALGORITHM
    title: str = ""
    steps: List[str] = field(default_factory=list)
    complexity: Optional[str] = None     # e.g. "O(n log n)"
    pseudocode: str = ""


@dataclass
class CaptionArtifact(BaseArtifact):
    """
    Explicit figure, table, or equation caption separated from the visual object.
    """
    artifact_type: ArtifactType = ArtifactType.CAPTION
    target_artifact_id: Optional[str] = None # Artifact ID this caption describes
    label: str = ""                      # e.g. "Figure 3.4", "Table 2.1"
    caption_text: str = ""


@dataclass
class ListArtifact(BaseArtifact):
    """
    Ordered or unordered academic enumeration.
    """
    artifact_type: ArtifactType = ArtifactType.LIST
    items: List[str] = field(default_factory=list)
    ordered: bool = False


@dataclass
class ExampleArtifact(BaseArtifact):
    """
    Explicit worked example or numerical problem with solution steps.
    """
    artifact_type: ArtifactType = ArtifactType.EXAMPLE
    title: str = ""
    problem_statement: str = ""
    solution_outline: str = ""
    numerical_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefinitionArtifact(BaseArtifact):
    """
    Formal academic definition of a term, theorem, or axiom.
    """
    artifact_type: ArtifactType = ArtifactType.DEFINITION
    term: str = ""
    definition: str = ""
    formal_notation: Optional[str] = None


@dataclass
class ProcedureArtifact(BaseArtifact):
    """
    Step-by-step laboratory or engineering procedure.
    """
    artifact_type: ArtifactType = ArtifactType.PROCEDURE
    title: str = ""
    steps: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    expected_outcome: str = ""


@dataclass
class CodeArtifact(BaseArtifact):
    """
    Source code listing or implementation snippet.
    """
    artifact_type: ArtifactType = ArtifactType.CODE
    language: str = "text"
    code: str = ""


@dataclass
class ReferenceArtifact(BaseArtifact):
    """
    Bibliographic reference or footnote citation.
    """
    artifact_type: ArtifactType = ArtifactType.REFERENCE
    citation_key: str = ""               # e.g. "[1]", "Tanenbaum2011"
    raw_citation: str = ""


ARTIFACT_CLASS_MAP: Dict[ArtifactType, type] = {
    ArtifactType.TEXT: TextArtifact,
    ArtifactType.HEADING: HeadingArtifact,
    ArtifactType.EQUATION: EquationArtifact,
    ArtifactType.FIGURE: FigureArtifact,
    ArtifactType.TABLE: TableArtifact,
    ArtifactType.ALGORITHM: AlgorithmArtifact,
    ArtifactType.CAPTION: CaptionArtifact,
    ArtifactType.LIST: ListArtifact,
    ArtifactType.EXAMPLE: ExampleArtifact,
    ArtifactType.DEFINITION: DefinitionArtifact,
    ArtifactType.PROCEDURE: ProcedureArtifact,
    ArtifactType.CODE: CodeArtifact,
    ArtifactType.REFERENCE: ReferenceArtifact,
}


def create_artifact_from_dict(data: Dict[str, Any]) -> BaseArtifact:
    """Factory helper to reconstruct the appropriate artifact subclass from a dict."""
    atype_str = data.get("artifact_type", "TEXT")
    atype = ArtifactType(atype_str)
    target_cls = ARTIFACT_CLASS_MAP.get(atype, BaseArtifact)
    d = dict(data)
    d["artifact_type"] = atype
    if "source_bbox" in d and d["source_bbox"] is not None:
        d["source_bbox"] = tuple(d["source_bbox"])
    return target_cls(**d)
