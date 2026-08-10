"""
AION VRE Numerical Parameter Engine (NPE)
=========================================
Implements Constraint-Respecting Perturbation (CRP) linked with domain solvers
to guarantee valid, fresh numerical parameters (Algorithm 5).
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple
from .contracts import OperationChain, QuantityType, VKO
from .errors import SolverError
from .solvers import solve_vko


class NPE:
    """Numerical Parameter Engine (Algorithm 5)."""

    MAX_ATTEMPTS: int = 25

    @classmethod
    def generate(cls, vko: VKO, chain: OperationChain) -> Tuple[VKO, Dict[str, Any]]:
        """
        Perturbs VKO quantities according to mutation rules and verifies
        mathematical solvability using the authoritative domain solver.
        Returns (new_vko, reference_solution).
        """
        rules = vko.mutability.mutation_rules
        if not rules or not vko.mutability.quantities_mutable:
            # Fall back to original VKO and solve directly
            sol = solve_vko(vko, chain)
            return (vko, sol)

        attempt = 0
        while attempt < cls.MAX_ATTEMPTS:
            attempt += 1
            candidate_vko = vko.clone()
            candidate_vko.id = f"{vko.id}_mut_{attempt}"

            # Apply candidate mutations
            cls._mutate_quantities(candidate_vko, rules)

            try:
                # Solvability Gate: Authoritative domain solver must succeed
                solution = solve_vko(candidate_vko, chain)
                if solution.get("unique_solution"):
                    return (candidate_vko, solution)
            except SolverError:
                continue

        # Fallback to original VKO if all mutations failed
        sol = solve_vko(vko, chain)
        return (vko, sol)

    @classmethod
    def _mutate_quantities(cls, vko: VKO, rules: list) -> None:
        for rule in rules:
            min_val, max_val = rule.value_range
            if rule.target == "EDGE_WEIGHTS":
                for edge_id in list(vko.quantities.edge_weights.keys()):
                    if rule.quantity_type == QuantityType.INTEGER:
                        new_w = float(random.randint(int(min_val), int(max_val)))
                    else:
                        new_w = round(random.uniform(min_val, max_val), 1)
                    vko.quantities.edge_weights[edge_id] = new_w

            elif rule.target == "COMPONENT_VALUES":
                for comp_id, (val, unit) in list(vko.quantities.component_values.items()):
                    if rule.quantity_type == QuantityType.RESISTANCE and "Ω" in unit:
                        e_series = [10.0, 15.0, 20.0, 30.0, 47.0, 68.0, 100.0]
                        vko.quantities.component_values[comp_id] = (random.choice(e_series), unit)
                    elif rule.quantity_type == QuantityType.VOLTAGE and "V" in unit:
                        vko.quantities.component_values[comp_id] = (float(random.randint(5, 24)), unit)

            elif rule.target == "LOAD_MAGNITUDES":
                for comp_id, (val, unit) in list(vko.quantities.component_values.items()):
                    vko.quantities.component_values[comp_id] = (float(random.randint(10, 50)), unit)
