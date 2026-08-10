"""
AION VRE Confidence & Evidence Fusion
======================================
Fuses evidence across Computer Vision (CV), OCR, and PDF context.
"""

from __future__ import annotations

from typing import Any, Dict
from .contracts import ConfidenceMetrics


class EvidenceFusion:
    """Fuses multi-source evidence to calculate composite confidence scores."""

    @staticmethod
    def fuse(
        cv_features: Dict[str, Any],
        ocr_data: Dict[str, Any],
        pdf_context: Dict[str, Any],
    ) -> ConfidenceMetrics:
        """
        Calculates distinct confidence scores across domain, class, topology,
        ocr, and semantic context.
        """
        # 1. Domain confidence from visual feature signals
        domain_conf = 0.5
        if cv_features.get("has_arrows") or cv_features.get("node_count", 0) > 2:
            domain_conf += 0.35
        if cv_features.get("has_numbers"):
            domain_conf += 0.10
        domain_conf = min(0.98, domain_conf)

        # 2. Class confidence from structural signature matching
        class_conf = 0.60
        if cv_features.get("node_count", 0) >= 3 and cv_features.get("edge_count", 0) >= 2:
            class_conf += 0.30
        class_conf = min(0.95, class_conf)

        # 3. Topology confidence from graph connection extraction
        top_conf = 0.70 if cv_features.get("edge_count", 0) > 0 else 0.40

        # 4. OCR confidence from text extraction quality
        ocr_conf = float(ocr_data.get("confidence", 0.5))

        # 5. Semantic confidence from surrounding text context relevance
        semantic_text = pdf_context.get("source_text", "")
        semantic_conf = 0.75 if len(semantic_text.split()) > 10 else 0.45

        return ConfidenceMetrics(
            domain_confidence=round(domain_conf, 2),
            class_confidence=round(class_conf, 2),
            topology_confidence=round(top_conf, 2),
            ocr_confidence=round(ocr_conf, 2),
            semantic_confidence=round(semantic_conf, 2),
        )
