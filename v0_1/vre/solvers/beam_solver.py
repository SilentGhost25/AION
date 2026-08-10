"""
AION VRE Beam Domain Solver
===========================
Deterministic equilibrium, reaction force, and SFD/BMD solver.
"""

from __future__ import annotations

from typing import Any, Dict
from ..contracts import OperationChain, VKO
from ..errors import SolverError


class BeamSolver:
    """Deterministic Beam Solver for support reactions and SFD/BMD values."""

    @classmethod
    def solve(cls, vko: VKO, chain: OperationChain) -> Dict[str, Any]:
        span = vko.quantities.span_length or 6.0
        load_val = vko.quantities.component_values.get("load_P1", (20.0, "kN"))[0]

        # Simply supported beam with central point load
        reaction_A = load_val / 2.0
        reaction_B = load_val / 2.0
        max_moment = (load_val * span) / 4.0

        return {
            "operation": "REACTIONS",
            "span_length": span,
            "point_load": load_val,
            "reaction_A": reaction_A,
            "reaction_B": reaction_B,
            "max_bending_moment": max_moment,
            "unique_solution": True,
        }
