"""
Tests for Quantity Parser and Deterministic Domain Solvers
"""

import pytest
from v0_1.vre.quantity_parser import QuantityParser
from v0_1.vre.solvers import GraphSolver, CircuitSolver, BeamSolver, TreeSolver
from v0_1.vre.vkoc import VKOC
from v0_1.vre.fsc import FSC
from v0_1.vre.vqg import VQGBuilder
from v0_1.vre.contracts import FigureExtractionResult, QuantityType


def test_quantity_parser_symbol_normalization():
    norm1 = QuantityParser.normalize_symbol("10O")
    assert "10 Ω" in norm1

    val, unit, conf = QuantityParser.parse_quantity("10O", expected_type=QuantityType.RESISTANCE)
    assert val == 10.0
    assert "Ω" in unit
    assert conf > 0.80


def test_graph_solver_dijkstra():
    extraction = FigureExtractionResult(status="PASS", image_path="test.png", page_number=1, bbox=None, confidence=0.85, extraction_method="test", width=400, height=300)
    classification = FSC.classify(extraction, concept_hint="Dijkstra shortest path")
    vko = VKOC.build(extraction, classification)
    vqg = VQGBuilder.build(vko)

    solution = GraphSolver.solve(vko, vqg.operation_chains[0])
    assert solution["operation"] == "DIJKSTRA"
    assert solution["source"] == "A"
    assert solution["destination"] == "D"
    assert solution["shortest_path"] == ["A", "B", "D"]
    assert solution["total_cost"] == 9.0
    assert solution["unique_solution"] is True


def test_circuit_solver():
    extraction = FigureExtractionResult(status="PASS", image_path="test.png", page_number=1, bbox=None, confidence=0.85, extraction_method="test", width=400, height=300)
    classification = FSC.classify(extraction, concept_hint="KVL circuit analysis")
    vko = VKOC.build(extraction, classification)
    vqg = VQGBuilder.build(vko)

    solution = CircuitSolver.solve(vko, vqg.operation_chains[0])
    assert solution["operation"] == "EQUIVALENT_RESISTANCE"
    assert solution["r_equivalent"] == 60.0
    assert solution["total_current"] == 0.2
    assert solution["unique_solution"] is True


def test_beam_solver():
    extraction = FigureExtractionResult(status="PASS", image_path="test.png", page_number=1, bbox=None, confidence=0.85, extraction_method="test", width=400, height=300)
    classification = FSC.classify(extraction, concept_hint="Beam reaction forces SFD BMD")
    vko = VKOC.build(extraction, classification)
    vqg = VQGBuilder.build(vko)

    solution = BeamSolver.solve(vko, vqg.operation_chains[0])
    assert solution["operation"] == "REACTIONS"
    assert solution["reaction_A"] == 10.0
    assert solution["reaction_B"] == 10.0
    assert solution["unique_solution"] is True
