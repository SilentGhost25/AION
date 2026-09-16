"""
AION v2 Model-Capability-Aware Evidence Bundles & Budgets
=========================================================
Defines the canonical evidence package presented to generative models.
Enforces archetype-driven evidence budgets to prevent context dilution.
Adapts output serialization dynamically to the target model's modalities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from aion.core.dom.artifacts import (
    ArtifactType,
    BaseArtifact,
    TextArtifact,
    HeadingArtifact,
    EquationArtifact,
    FigureArtifact,
    TableArtifact,
    AlgorithmArtifact,
    ExampleArtifact,
    DefinitionArtifact,
    ProcedureArtifact,
    CodeArtifact,
)
from aion.core.dom.artifact_registry import ArtifactRegistry


class ModelCapability(str, Enum):
    TEXT_ONLY = "TEXT_ONLY"         # e.g., Qwen2.5-32B, DeepSeek-R1-32B, Llama-3.3-70B
    MULTIMODAL = "MULTIMODAL"       # e.g., Qwen2.5-VL-7B, Qwen2.5-VL-32B, Llama-3.2-Vision


class QuestionArchetype(str, Enum):
    NUMERICAL = "NUMERICAL"               # Parameter calculations, circuit equations, quantitative proofs
    DERIVATION = "DERIVATION"             # Step-by-step mathematical theorems and governing laws
    CIRCUIT_SYSTEM = "CIRCUIT_SYSTEM"     # Physical schematic, system block diagrams, flowcharts
    COMPARISON = "COMPARISON"             # Tabular contrast, trade-offs, multi-concept matrices
    ALGORITHMIC = "ALGORITHMIC"           # Pseudocode, execution traces, asymptotic complexity
    CONCEPTUAL = "CONCEPTUAL"             # Core definitions, principles, taxonomy
    PROCEDURAL = "PROCEDURAL"             # Laboratory protocols, installation, step sequences


@dataclass
class EvidenceBudget:
    """
    Enforces maximum and minimum artifact allocations for an archetype.
    Prevents both prompt starvation and context dilution.
    """
    archetype: QuestionArchetype
    max_text_blocks: int = 2
    min_equations: int = 0
    max_equations: int = 0
    min_figures: int = 0
    max_figures: int = 0
    min_tables: int = 0
    max_tables: int = 0
    min_algorithms: int = 0
    max_algorithms: int = 0


ARCHETYPE_BUDGETS: Dict[QuestionArchetype, EvidenceBudget] = {
    QuestionArchetype.NUMERICAL: EvidenceBudget(
        archetype=QuestionArchetype.NUMERICAL,
        max_text_blocks=2,
        min_equations=1,
        max_equations=3,
        min_figures=0,
        max_figures=1,
        min_tables=0,
        max_tables=1,
    ),
    QuestionArchetype.DERIVATION: EvidenceBudget(
        archetype=QuestionArchetype.DERIVATION,
        max_text_blocks=1,
        min_equations=2,
        max_equations=5,
        min_figures=0,
        max_figures=1,
        min_tables=0,
        max_tables=0,
    ),
    QuestionArchetype.CIRCUIT_SYSTEM: EvidenceBudget(
        archetype=QuestionArchetype.CIRCUIT_SYSTEM,
        max_text_blocks=2,
        min_equations=0,
        max_equations=2,
        min_figures=1,  # Mandatory figure!
        max_figures=1,
        min_tables=0,
        max_tables=0,
    ),
    QuestionArchetype.COMPARISON: EvidenceBudget(
        archetype=QuestionArchetype.COMPARISON,
        max_text_blocks=3,
        min_equations=0,
        max_equations=0,
        min_figures=0,
        max_figures=0,
        min_tables=1,  # Mandatory table!
        max_tables=1,
    ),
    QuestionArchetype.ALGORITHMIC: EvidenceBudget(
        archetype=QuestionArchetype.ALGORITHMIC,
        max_text_blocks=1,
        min_equations=0,
        max_equations=1,
        min_figures=0,
        max_figures=1,
        min_algorithms=1,  # Mandatory algorithm!
        max_algorithms=1,
    ),
    QuestionArchetype.CONCEPTUAL: EvidenceBudget(
        archetype=QuestionArchetype.CONCEPTUAL,
        max_text_blocks=3,
        min_equations=0,
        max_equations=1,
        min_figures=0,
        max_figures=1,
        min_tables=0,
        max_tables=1,
    ),
    QuestionArchetype.PROCEDURAL: EvidenceBudget(
        archetype=QuestionArchetype.PROCEDURAL,
        max_text_blocks=2,
        min_equations=0,
        max_equations=0,
        min_figures=0,
        max_figures=1,
        min_tables=0,
        max_tables=1,
    ),
}


@dataclass
class EvidenceBundle:
    """
    Curated, boundary-locked evidence bundle assembled for drafting one question.
    Preserves all native multimodal artifacts. Zero flattened string blobs.
    """
    bundle_id: str                         # e.g., "BNDL-M3-Q05A"
    module_index: int                      # 1 to 5
    section_id: str                        # e.g., "SEC-M3-004"
    section_title: str                     # e.g., "3.4 Congestion Control in TCP"
    primary_concept: str                   # e.g., "TCP Congestion Avoidance (AIMD)"
    archetype: QuestionArchetype = QuestionArchetype.CONCEPTUAL

    # Typed Native Artifacts
    texts: List[TextArtifact] = field(default_factory=list)
    headings: List[HeadingArtifact] = field(default_factory=list)
    equations: List[EquationArtifact] = field(default_factory=list)
    figures: List[FigureArtifact] = field(default_factory=list)
    tables: List[TableArtifact] = field(default_factory=list)
    algorithms: List[AlgorithmArtifact] = field(default_factory=list)
    examples: List[ExampleArtifact] = field(default_factory=list)
    definitions: List[DefinitionArtifact] = field(default_factory=list)
    procedures: List[ProcedureArtifact] = field(default_factory=list)
    codes: List[CodeArtifact] = field(default_factory=list)

    # Cross-artifact relationship hints
    relationships: List[Dict[str, Any]] = field(default_factory=list)

    def to_llm_payload(
        self,
        capability: ModelCapability = ModelCapability.TEXT_ONLY,
        registry: Optional[ArtifactRegistry] = None,
    ) -> Dict[str, Any]:
        """
        Serialize evidence into a model-capability-aware JSON payload.
        - TEXT_ONLY: Uses verified LaTeX, structured paragraphs, markdown tables,
          and visual semantic descriptors (zero raw image paths).
        - MULTIMODAL: Preserves resolved asset file paths, raw table rows/cols,
          and multimodal prompt tokens.
        """
        base_payload: Dict[str, Any] = {
            "bundle_id": self.bundle_id,
            "module_index": self.module_index,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "primary_concept": self.primary_concept,
            "archetype": self.archetype.value,
            "context_paragraphs": [
                {
                    "artifact_id": t.artifact_id,
                    "text": t.normalized_text or t.raw_text,
                    "referenced_figures": t.referenced_artifact_ids,
                }
                for t in self.texts
            ],
            "definitions": [
                {"term": d.term, "definition": d.definition, "notation": d.formal_notation}
                for d in self.definitions
            ],
            "governing_equations": [
                {
                    "artifact_id": e.artifact_id,
                    "latex": e.latex,
                    "variables": e.variables,
                    "label": e.equation_label,
                }
                for e in self.equations
            ],
            "algorithms": [
                {
                    "artifact_id": a.artifact_id,
                    "title": a.title,
                    "steps": a.steps,
                    "complexity": a.complexity,
                }
                for a in self.algorithms
            ],
            "worked_examples": [
                {
                    "title": ex.title,
                    "problem": ex.problem_statement,
                    "solution": ex.solution_outline,
                    "values": ex.numerical_values,
                }
                for ex in self.examples
            ],
            "procedures": [
                {"title": p.title, "steps": p.steps, "prerequisites": p.prerequisites}
                for p in self.procedures
            ],
        }

        if capability == ModelCapability.TEXT_ONLY:
            # For text-only models (Qwen2.5-32B, DeepSeek-R1):
            # Figures are represented by their extracted academic captions, visual elements,
            # and opaque figure_id. Tables are formatted as clean markdown tables.
            base_payload["available_figures"] = [
                {
                    "figure_id": f.artifact_id,
                    "caption": f.caption,
                    "figure_type": f.figure_type,
                    "visual_summary": f.semantic_description,
                }
                for f in self.figures
            ]
            base_payload["tables"] = [
                {
                    "table_id": t.artifact_id,
                    "caption": t.caption,
                    "markdown": t.markdown_repr,
                }
                for t in self.tables
            ]

        elif capability == ModelCapability.MULTIMODAL:
            # For multimodal vision-capable models (Qwen2.5-VL):
            # Resolve actual physical image paths from registry so the model can inspect bitmaps directly.
            multimodal_figures = []
            for f in self.figures:
                resolved_path = ""
                if registry is not None:
                    p = registry.resolve_figure_path(f.artifact_id)
                    if p and p.exists():
                        resolved_path = str(p)
                multimodal_figures.append({
                    "figure_id": f.artifact_id,
                    "image_path": resolved_path,
                    "caption": f.caption,
                    "dimensions": [f.width, f.height],
                    "figure_type": f.figure_type,
                })
            base_payload["multimodal_figures"] = multimodal_figures
            base_payload["tables"] = [
                {
                    "table_id": t.artifact_id,
                    "caption": t.caption,
                    "headers": t.headers,
                    "rows": t.rows,
                    "markdown": t.markdown_repr,
                }
                for t in self.tables
            ]

        return base_payload
