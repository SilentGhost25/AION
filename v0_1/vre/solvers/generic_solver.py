"""
AION VRE Generic Fallback Domain Solver
========================================
Fallback solver for unstructured figure queries.
"""

from __future__ import annotations

from typing import Any, Dict
from ..contracts import OperationChain, VKO


class GenericSolver:
    """Generic fallback solver."""

    @classmethod
    def solve(cls, vko: VKO, chain: OperationChain) -> Dict[str, Any]:
        return {
            "operation": chain.steps[0].operation if chain.steps else "GENERIC",
            "vko_id": vko.id,
            "unique_solution": True,
        }
