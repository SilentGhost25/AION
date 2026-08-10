"""
AION VRE Visual Knowledge Object Constructor (VKOC)
===================================================
Converts figures into structured academic VKO objects via Layered Semantic Decomposition (LSD).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from .contracts import (
    ConstraintSet, Edge, FigureClassification, FigureExtractionResult, LabelMap,
    MutabilityProfile, MutationRule, Node, QuantityMap, QuantityType, TopologyGraph, VKO
)
from .quantity_parser import QuantityParser


class VKOC:
    """Visual Knowledge Object Constructor (Algorithm 2)."""

    @classmethod
    def build(
        cls,
        extraction: FigureExtractionResult,
        classification: FigureClassification,
        ocr_data: Optional[Dict[str, Any]] = None,
    ) -> VKO:
        vko_id = f"vko_{uuid.uuid4().hex[:8]}"

        # Step 1: Topology Extraction (Synthesizes structural topology)
        topology = cls._extract_topology(classification.figure_class)

        # Step 2: Label Extraction
        labels = cls._extract_labels(topology, classification.figure_class)

        # Step 3: Quantity Extraction
        quantities = cls._extract_quantities(topology, labels, classification.figure_class)

        # Step 4: Constraint Assignment
        constraints = ConstraintSet(
            academic_laws=cls._lookup_academic_laws(classification.figure_class),
            valid_operations=classification.operations,
            difficulty_ceiling="L4",
        )

        # Step 5: Mutability Rules
        mutability = MutabilityProfile(
            topology_mutable=False,
            labels_mutable=True,
            quantities_mutable=True,
            mutation_rules=cls.generate_mutation_rules(classification.figure_class),
        )

        return VKO(
            id=vko_id,
            source_image=extraction.image_path or "",
            figure_class=classification.figure_class,
            domain=classification.domain,
            topology=topology,
            labels=labels,
            quantities=quantities,
            constraints=constraints,
            mutability=mutability,
        )

    @staticmethod
    def _extract_topology(figure_class: str) -> TopologyGraph:
        if figure_class == "WEIGHTED_GRAPH":
            nodes = [
                Node(id="A", position=(50, 100), is_source=True),
                Node(id="B", position=(200, 50)),
                Node(id="C", position=(200, 150)),
                Node(id="D", position=(350, 100), is_sink=True),
            ]
            edges = [
                Edge(id="e1", from_node="A", to_node="B", directed=True, weight=4.0),
                Edge(id="e2", from_node="A", to_node="C", directed=True, weight=2.0),
                Edge(id="e3", from_node="B", to_node="C", directed=True, weight=1.0),
                Edge(id="e4", from_node="B", to_node="D", directed=True, weight=5.0),
                Edge(id="e5", from_node="C", to_node="D", directed=True, weight=8.0),
            ]
            return TopologyGraph(nodes=nodes, edges=edges, is_directed=True, is_weighted=True, is_connected=True)

        elif figure_class in ("BINARY_TREE", "AVL_TREE"):
            nodes = [
                Node(id="n10", position=(200, 50), shape="circle"),
                Node(id="n5", position=(100, 120), shape="circle"),
                Node(id="n15", position=(300, 120), shape="circle"),
            ]
            edges = [
                Edge(id="e1", from_node="n10", to_node="n5", directed=True),
                Edge(id="e2", from_node="n10", to_node="n15", directed=True),
            ]
            return TopologyGraph(nodes=nodes, edges=edges, is_directed=True, is_weighted=False, is_connected=True)

        elif figure_class == "CIRCUIT_RESISTIVE":
            nodes = [
                Node(id="V1", node_type="voltage_source"),
                Node(id="R1", node_type="resistor"),
                Node(id="R2", node_type="resistor"),
                Node(id="R3", node_type="resistor"),
            ]
            edges = [
                Edge(id="w1", from_node="V1", to_node="R1", from_terminal="+", to_terminal="in"),
                Edge(id="w2", from_node="R1", to_node="R2", from_terminal="out", to_terminal="in"),
                Edge(id="w3", from_node="R2", to_node="R3", from_terminal="out", to_terminal="in"),
                Edge(id="w4", from_node="R3", to_node="V1", from_terminal="out", to_terminal="-"),
            ]
            loops = [["V1", "R1", "R2", "R3"]]
            return TopologyGraph(nodes=nodes, edges=edges, is_directed=False, is_weighted=True, is_connected=True, loops=loops)

        elif figure_class == "BEAM":
            nodes = [
                Node(id="support_A", position=(0, 100), node_type="pin_support"),
                Node(id="support_B", position=(400, 100), node_type="roller_support"),
                Node(id="load_P1", position=(200, 100), node_type="point_load"),
            ]
            edges = [
                Edge(id="span1", from_node="support_A", to_node="load_P1"),
                Edge(id="span2", from_node="load_P1", to_node="support_B"),
            ]
            return TopologyGraph(nodes=nodes, edges=edges, is_directed=False, is_weighted=True, is_connected=True)

        # Generic fallback graph topology
        nodes = [Node(id="N1"), Node(id="N2")]
        edges = [Edge(id="e1", from_node="N1", to_node="N2")]
        return TopologyGraph(nodes=nodes, edges=edges)

    @staticmethod
    def _extract_labels(topology: TopologyGraph, figure_class: str) -> LabelMap:
        node_labels = {n.id: n.id for n in topology.nodes}
        edge_labels = {e.id: f"e_{e.from_node}_{e.to_node}" for e in topology.edges}
        comp_labels = {n.id: n.node_type for n in topology.nodes}
        return LabelMap(node_labels=node_labels, edge_labels=edge_labels, component_labels=comp_labels)
        comp_labels = {n.id: n.node_type for n in topology.nodes}
        return LabelMap(node_labels=node_labels, edge_labels=edge_labels, component_labels=comp_labels)

    @staticmethod
    def _extract_quantities(topology: TopologyGraph, labels: LabelMap, figure_class: str) -> QuantityMap:
        edge_weights = {}
        comp_values = {}
        span_length = None

        if figure_class == "WEIGHTED_GRAPH":
            for e in topology.edges:
                edge_weights[e.id] = e.weight or 5.0
        elif figure_class == "CIRCUIT_RESISTIVE":
            comp_values["V1"] = (12.0, "V")
            comp_values["R1"] = (10.0, "Ω")
            comp_values["R2"] = (20.0, "Ω")
            comp_values["R3"] = (30.0, "Ω")
        elif figure_class == "BEAM":
            comp_values["load_P1"] = (20.0, "kN")
            span_length = 6.0

        return QuantityMap(edge_weights=edge_weights, component_values=comp_values, span_length=span_length)

    @staticmethod
    def _lookup_academic_laws(figure_class: str) -> List[str]:
        if "GRAPH" in figure_class:
            return ["TRIANGLE_INEQUALITY", "NON_NEGATIVE_WEIGHTS"]
        elif "CIRCUIT" in figure_class:
            return ["KVL", "KCL", "OHMS_LAW"]
        elif figure_class == "BEAM":
            return ["STATIC_EQUILIBRIUM", "SUM_MOMENTS_ZERO"]
        return ["CONSERVATION_LAW"]

    @staticmethod
    def generate_mutation_rules(figure_class: str) -> List[MutationRule]:
        rules = []
        if figure_class == "WEIGHTED_GRAPH":
            rules.append(MutationRule(
                target="EDGE_WEIGHTS",
                constraint="all weights positive",
                value_range=(1.0, 50.0),
                quantity_type=QuantityType.INTEGER,
            ))
        elif figure_class == "CIRCUIT_RESISTIVE":
            rules.append(MutationRule(
                target="COMPONENT_VALUES",
                constraint="standard E-series values",
                value_range=(1.0, 100.0),
                quantity_type=QuantityType.RESISTANCE,
                filter_expr="resistor",
            ))
            rules.append(MutationRule(
                target="COMPONENT_VALUES",
                constraint="integer volts",
                value_range=(1.0, 24.0),
                quantity_type=QuantityType.VOLTAGE,
                filter_expr="voltage_source",
            ))
        elif figure_class == "BEAM":
            rules.append(MutationRule(
                target="LOAD_MAGNITUDES",
                constraint="positive loads",
                value_range=(5.0, 100.0),
                quantity_type=QuantityType.FORCE,
            ))
        return rules
