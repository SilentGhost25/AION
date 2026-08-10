"""
Tests for Figure Semantic Classifier (FSC)
"""

import pytest
from v0_1.vre.fsc import FSC
from v0_1.vre.contracts import FigureExtractionResult


def test_fsc_classification():
    extraction = FigureExtractionResult(
        status="PASS",
        image_path="test.png",
        page_number=1,
        bbox=None,
        confidence=0.85,
        extraction_method="test",
        width=400,
        height=300,
    )

    cls_result = FSC.classify(extraction, concept_hint="Dijkstra's algorithm shortest path")
    assert cls_result.supported is True
    assert cls_result.domain == "DSA"
    assert cls_result.figure_class == "WEIGHTED_GRAPH"
    assert "DIJKSTRA" in cls_result.operations
    assert cls_result.confidence.composite_confidence > 0.50


def test_fsc_circuit_classification():
    extraction = FigureExtractionResult(
        status="PASS", image_path="test.png", page_number=1, bbox=None,
        confidence=0.85, extraction_method="test", width=400, height=300,
    )
    cls_result = FSC.classify(extraction, concept_hint="KVL circuit analysis resistor voltage")
    assert cls_result.supported is True
    assert cls_result.domain == "ECE"
    assert cls_result.figure_class == "CIRCUIT_RESISTIVE"
    assert "KVL" in cls_result.operations
