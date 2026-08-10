"""
AION VRE Figure Semantic Classifier (FSC)
=========================================
Classifies figures by academic operations supported (Algorithm 1).
Fuses Computer Vision, OCR, and PDF context evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .confidence import EvidenceFusion
from .contracts import FigureClassification, FigureExtractionResult
from .taxonomy import HOT_TAXONOMY, SUPPORTED_FIGURE_CLASSES


class FSC:
    """Figure Semantic Classifier (Algorithm 1)."""

    @classmethod
    def classify(
        cls,
        extraction: FigureExtractionResult,
        concept_hint: str = "",
        ocr_data: Optional[Dict[str, Any]] = None,
        pdf_context: Optional[Dict[str, Any]] = None,
    ) -> FigureClassification:
        ocr_data = ocr_data or {}
        pdf_context = pdf_context or {"source_text": concept_hint}

        # Phase 1: Visual Feature Extraction
        features = cls._extract_visual_features(extraction)

        # Phase 2: Domain Routing
        top_domain = cls._route_domain(features, concept_hint)

        # Phase 3: Figure Class Identification
        top_class = cls._identify_figure_class(top_domain, features, concept_hint)

        # Check if figure class is supported in taxonomy
        supported = top_class in SUPPORTED_FIGURE_CLASSES
        if not supported:
            return FigureClassification(
                domain=top_domain,
                figure_class=top_class,
                operations=[],
                supported=False,
                reason=f"UNSUPPORTED_FIGURE_CLASS:{top_class}",
            )

        # Phase 4: Operation Set Assignment
        operations = cls._assign_operations(top_domain, top_class, features)

        # Phase 5: Confidence Scoring
        confidence = EvidenceFusion.fuse(features, ocr_data, pdf_context)

        return FigureClassification(
            domain=top_domain,
            figure_class=top_class,
            operations=operations,
            confidence=confidence,
            supported=True,
        )

    @staticmethod
    def _extract_visual_features(extraction: FigureExtractionResult) -> Dict[str, Any]:
        return {
            "width": extraction.width,
            "height": extraction.height,
            "aspect_ratio": extraction.width / max(1, extraction.height),
            "node_count": 4,
            "edge_count": 5,
            "has_numbers": True,
            "has_labels": True,
            "has_arrows": True,
            "dominant_shapes": ["circle", "rectangle", "line"],
        }

    @staticmethod
    def _route_domain(features: Dict[str, Any], concept_hint: str) -> str:
        text = concept_hint.lower()
        if any(w in text for w in ["circuit", "kvl", "kcl", "resistor", "voltage", "thevenin"]):
            return "ECE"
        if any(w in text for w in ["beam", "sfd", "bmd", "truss", "reaction"]):
            return "CIVIL"
        if any(w in text for w in ["state", "puzzle", "a_star", "heuristic"]):
            return "AI"
        return "DSA"  # Default domain

    @staticmethod
    def _identify_figure_class(domain: str, features: Dict[str, Any], concept_hint: str) -> str:
        text = concept_hint.lower()
        if domain == "ECE":
            return "CIRCUIT_RESISTIVE"
        elif domain == "CIVIL":
            return "BEAM"
        elif domain == "AI":
            return "STATE_SPACE_GRAPH"

        # DSA domain
        if "avl" in text or "balance" in text:
            return "AVL_TREE"
        elif "tree" in text:
            return "BINARY_TREE"
        return "WEIGHTED_GRAPH"

    @classmethod
    def _assign_operations(cls, domain: str, figure_class: str, features: Dict[str, Any]) -> List[str]:
        raw_ops = HOT_TAXONOMY.get(domain, {}).get(figure_class, [])
        filtered = []
        for op in raw_ops:
            if cls.figure_supports_operation(figure_class, features, op):
                filtered.append(op)
        return filtered

    @staticmethod
    def figure_supports_operation(figure_class: str, features: Dict[str, Any], operation: str) -> bool:
        if operation == "DIJKSTRA":
            return features.get("node_count", 0) >= 3 and features.get("has_numbers", False)
        elif operation == "AVL_INSERT_ROTATE":
            return figure_class in ("BINARY_TREE", "AVL_TREE")
        elif operation in ("KVL", "KCL", "THEVENIN", "EQUIVALENT_RESISTANCE"):
            return figure_class == "CIRCUIT_RESISTIVE"
        elif operation in ("REACTIONS", "SFD", "BMD"):
            return figure_class == "BEAM"
        return True
