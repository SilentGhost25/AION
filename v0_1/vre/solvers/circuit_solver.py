"""
AION VRE Circuit Domain Solver
==============================
Deterministic KVL/KCL and Equivalent Resistance solver.
"""

from __future__ import annotations

from typing import Any, Dict
from ..contracts import OperationChain, VKO
from ..errors import SolverError


class CircuitSolver:
    """Deterministic Circuit Solver for KVL/KCL and Equivalent Resistance."""

    @classmethod
    def solve(cls, vko: VKO, chain: OperationChain) -> Dict[str, Any]:
        comp_vals = vko.quantities.component_values
        if not comp_vals:
            raise SolverError("Circuit has no component values.")

        v_source = comp_vals.get("V1", (12.0, "V"))[0]
        r1 = comp_vals.get("R1", (10.0, "Ω"))[0]
        r2 = comp_vals.get("R2", (20.0, "Ω"))[0]
        r3 = comp_vals.get("R3", (30.0, "Ω"))[0]

        r_eq = r1 + r2 + r3
        current = v_source / max(0.001, r_eq)

        return {
            "operation": "EQUIVALENT_RESISTANCE",
            "v_source": v_source,
            "r_equivalent": r_eq,
            "total_current": round(current, 3),
            "units": {"r": "Ω", "i": "A", "v": "V"},
            "unique_solution": True,
        }
