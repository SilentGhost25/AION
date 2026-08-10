"""
Tests for VKOC and VKO Validator
"""

import pytest
from v0_1.vre.vkoc import VKOC
from v0_1.vre.vko_validator import VKOValidator
from v0_1.vre.fsc import FSC
from v0_1.vre.contracts import FigureExtractionResult


def test_vkoc_building_and_validation():
    extraction = FigureExtractionResult(
        status="PASS", image_path="test.png", page_number=1, bbox=None,
        confidence=0.85, extraction_method="test", width=400, height=300,
    )
    classification = FSC.classify(extraction, concept_hint="Dijkstra shortest path")

    vko = VKOC.build(extraction, classification)
    assert vko.id.startswith("vko_")
    assert vko.figure_class == "WEIGHTED_GRAPH"
    assert len(vko.topology.nodes) > 0
    assert len(vko.topology.edges) > 0

    valid, errors = VKOValidator.validate(vko)
    assert valid is True
    assert len(errors) == 0


def test_vko_validator_failures():
    extraction = FigureExtractionResult(
        status="PASS", image_path="test.png", page_number=1, bbox=None,
        confidence=0.85, extraction_method="test", width=400, height=300,
    )
    classification = FSC.classify(extraction, concept_hint="Dijkstra shortest path")
    vko = VKOC.build(extraction, classification)

    # Corrupt node in edge
    vko.topology.edges[0].from_node = "NON_EXISTENT_NODE"
    valid, errors = VKOValidator.validate(vko)
    assert valid is False
    assert any("INVALID_EDGE_SOURCE" in e for e in errors)
