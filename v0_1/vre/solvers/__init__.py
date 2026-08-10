"""
AION VRE Solvers Factory
========================
Provides unified access to deterministic domain solvers.
"""

from __future__ import annotations

from typing import Any, Dict
from ..contracts import OperationChain, VKO
from .beam_solver import BeamSolver
from .circuit_solver import CircuitSolver
from .generic_solver import GenericSolver
from .graph_solver import GraphSolver
from .tree_solver import TreeSolver


def get_solver(figure_class: str):
    """Returns the authoritative domain solver for a given figure class."""
    if "GRAPH" in figure_class:
        return GraphSolver
    elif "TREE" in figure_class:
        return TreeSolver
    elif "CIRCUIT" in figure_class:
        return CircuitSolver
    elif figure_class == "BEAM":
        return BeamSolver
    return GenericSolver


def solve_vko(vko: VKO, chain: OperationChain) -> Dict[str, Any]:
    """Helper method to execute deterministic domain solving."""
    solver = get_solver(vko.figure_class)
    return solver.solve(vko, chain)


__all__ = [
    "GraphSolver",
    "TreeSolver",
    "CircuitSolver",
    "BeamSolver",
    "GenericSolver",
    "get_solver",
    "solve_vko",
]
