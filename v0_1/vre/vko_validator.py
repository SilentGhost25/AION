"""
AION VRE VKO Validator (Integrity Gate)
========================================
Validates structural integrity rules for VKO instances per figure class.
"""

from __future__ import annotations

from typing import List, Tuple
from .contracts import VKO


class VKOValidator:
    """VKO Integrity Gate validating structural rules before reasoning chain enumeration."""

    @classmethod
    def validate(cls, vko: VKO) -> Tuple[bool, List[str]]:
        errors = []

        if not vko or not vko.id:
            return (False, ["NULL_OR_EMPTY_VKO"])

        cls_name = vko.figure_class

        if "GRAPH" in cls_name:
            cls._validate_graph(vko, errors)
        elif "TREE" in cls_name:
            cls._validate_tree(vko, errors)
        elif "CIRCUIT" in cls_name:
            cls._validate_circuit(vko, errors)
        elif cls_name == "BEAM":
            cls._validate_beam(vko, errors)
        else:
            cls._validate_generic(vko, errors)

        return (len(errors) == 0, errors)

    @staticmethod
    def _validate_graph(vko: VKO, errors: List[str]) -> None:
        nodes = {n.id for n in vko.topology.nodes}
        if not nodes:
            errors.append("GRAPH_NO_NODES")
            return

        for edge in vko.topology.edges:
            if edge.from_node not in nodes:
                errors.append(f"INVALID_EDGE_SOURCE:{edge.from_node}")
            if edge.to_node not in nodes:
                errors.append(f"INVALID_EDGE_DEST:{edge.to_node}")

        if vko.topology.is_weighted:
            for edge in vko.topology.edges:
                w = vko.quantities.edge_weights.get(edge.id)
                if w is None or w <= 0:
                    errors.append(f"INVALID_EDGE_WEIGHT:{edge.id}")

    @staticmethod
    def _validate_tree(vko: VKO, errors: List[str]) -> None:
        if not vko.topology.nodes:
            errors.append("TREE_NO_NODES")
            return

        # Check max children count (binary tree requirement: max 2 children per parent)
        child_counts = {}
        for edge in vko.topology.edges:
            child_counts[edge.from_node] = child_counts.get(edge.from_node, 0) + 1
            if child_counts[edge.from_node] > 2:
                errors.append(f"EXCEEDS_BINARY_TREE_DEGREE:{edge.from_node}")

    @staticmethod
    def _validate_circuit(vko: VKO, errors: List[str]) -> None:
        if not vko.topology.nodes:
            errors.append("CIRCUIT_NO_COMPONENTS")
            return

        if not vko.quantities.component_values:
            errors.append("CIRCUIT_NO_COMPONENT_VALUES")

    @staticmethod
    def _validate_beam(vko: VKO, errors: List[str]) -> None:
        if not vko.topology.nodes:
            errors.append("BEAM_NO_SUPPORTS_OR_LOADS")
            return

        if vko.quantities.span_length is None or vko.quantities.span_length <= 0:
            errors.append("INVALID_BEAM_SPAN_LENGTH")

    @staticmethod
    def _validate_generic(vko: VKO, errors: List[str]) -> None:
        if not vko.topology.nodes:
            errors.append("GENERIC_VKO_EMPTY")
