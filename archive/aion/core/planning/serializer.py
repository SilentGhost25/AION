"""
AION v2 Capability Serializer
==============================
Decoupled payload serializer adapting structured EvidenceBundles into model-ready payloads.
- TEXT_ONLY: Uses verified LaTeX, markdown tables, visual summaries (zero filesystem paths).
- MULTIMODAL: Resolves abstract asset keys dynamically via ArtifactRegistry into concrete image assets.
Preserves Artifact ID -> ArtifactRegistry -> Asset Resolver boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aion.core.dom.artifacts import (
    ArtifactType,
    EquationArtifact,
    FigureArtifact,
    TableArtifact,
    AlgorithmArtifact,
    TextArtifact,
)
from aion.core.dom.document_dom import DocumentDOM
from aion.core.planning.contracts import EvidenceBundle, EvidenceRole


class CapabilitySerializer:
    """
    Serializes an EvidenceBundle according to model capability without polluting the planner.
    """

    @classmethod
    def serialize_text_only(
        cls,
        bundle: EvidenceBundle,
        dom: DocumentDOM,
    ) -> Dict[str, Any]:
        """
        Produce a pure text-grounded payload suitable for language-only models (e.g. Qwen2.5-14B, DeepSeek-R1).
        """
        artifacts = bundle.resolve_artifacts(dom)

        primary_texts = []
        contextual_notes = []
        equations = []
        figures = []
        tables = []
        algorithms = []

        ref_map = {ref.artifact_id: ref for ref in bundle.item_refs}

        for art in artifacts:
            ref = ref_map.get(art.artifact_id)
            role = ref.evidence_role if ref else EvidenceRole.PRIMARY

            if isinstance(art, TextArtifact):
                text_val = art.normalized_text or art.raw_text
                item = {
                    "artifact_id": art.artifact_id,
                    "text": text_val,
                    "page": art.page_number,
                }
                if role == EvidenceRole.PRIMARY:
                    primary_texts.append(item)
                else:
                    contextual_notes.append(item)

            elif isinstance(art, EquationArtifact):
                equations.append({
                    "artifact_id": art.artifact_id,
                    "latex": art.latex,
                    "variables": art.variables,
                    "label": art.equation_label,
                    "status": art.verification_status,
                })

            elif isinstance(art, FigureArtifact):
                figures.append({
                    "figure_id": art.artifact_id,
                    "caption": art.caption or f"Diagram on page {art.page_number}",
                    "figure_type": art.figure_type,
                    "visual_summary": art.semantic_description,
                })

            elif isinstance(art, TableArtifact):
                tables.append({
                    "table_id": art.artifact_id,
                    "caption": art.caption,
                    "markdown": art.markdown_repr,
                })

            elif isinstance(art, AlgorithmArtifact):
                algorithms.append({
                    "artifact_id": art.artifact_id,
                    "title": art.title,
                    "steps": art.steps,
                    "complexity": art.complexity,
                })

        return {
            "bundle_id": bundle.bundle_id,
            "spec_id": bundle.spec_id,
            "primary_ku_id": bundle.primary_ku_id,
            "format": "TEXT_ONLY",
            "source_paragraphs": primary_texts,
            "contextual_notes": contextual_notes,
            "governing_equations": equations,
            "available_figures": figures,
            "tables": tables,
            "algorithms": algorithms,
            "budget_used": bundle.budget_used,
        }

    @classmethod
    def serialize_multimodal(
        cls,
        bundle: EvidenceBundle,
        dom: DocumentDOM,
    ) -> Dict[str, Any]:
        """
        Produce a vision-grounded payload suitable for multimodal models (e.g. Qwen2.5-VL).
        Resolves asset keys through ArtifactRegistry into physical image paths.
        """
        base = cls.serialize_text_only(bundle, dom)
        base["format"] = "MULTIMODAL"

        multimodal_figures = []
        for aid in bundle.figure_artifact_ids:
            art = dom.registry.get(aid)
            if isinstance(art, FigureArtifact):
                resolved_path = ""
                try:
                    p = dom.registry.resolve_asset_path(art.asset_key)
                    if p.exists():
                        resolved_path = str(p)
                except Exception:
                    resolved_path = ""

                multimodal_figures.append({
                    "figure_id": art.artifact_id,
                    "image_path": resolved_path,
                    "caption": art.caption,
                    "dimensions": [art.width, art.height],
                    "figure_type": art.figure_type,
                })

        base["multimodal_figures"] = multimodal_figures

        # Include structured table matrices
        raw_tables = []
        for aid in bundle.table_artifact_ids:
            art = dom.registry.get(aid)
            if isinstance(art, TableArtifact):
                raw_tables.append({
                    "table_id": art.artifact_id,
                    "headers": art.headers,
                    "rows": art.rows,
                    "markdown": art.markdown_repr,
                })
        base["tables"] = raw_tables

        return base
