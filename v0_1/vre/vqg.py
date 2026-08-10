"""
AION VRE Visual Question Graph (VQG)
====================================
Operation Chain Modeling (Algorithm 3) generating scored reasoning chains.
"""

from __future__ import annotations

from typing import Dict, List
from .contracts import OperationChain, OperationStep, VKO, VQG


class VQGBuilder:
    """Visual Question Graph Builder (Algorithm 3)."""

    MAX_CHAINS: int = 5

    @classmethod
    def build(cls, vko: VKO) -> VQG:
        vqg = VQG(vko_id=vko.id)
        candidate_chains: List[OperationChain] = []

        for op in vko.constraints.valid_operations:
            chains = cls._enumerate_chains_for_op(op, vko)
            candidate_chains.extend(chains)

        # Score & Filter Top-K
        scored_chains = cls._score_and_filter_chains(candidate_chains, vko)
        vqg.operation_chains = scored_chains[: cls.MAX_CHAINS]

        for chain in vqg.operation_chains:
            vqg.bloom_mapping.setdefault(chain.bloom_level, []).append(chain)

        return vqg

    @classmethod
    def _enumerate_chains_for_op(cls, start_op: str, vko: VKO) -> List[OperationChain]:
        chains = []

        if start_op == "DIJKSTRA":
            c1 = OperationChain(
                chain_id="chain_dijkstra_basic",
                bloom_level="L3",
                steps=[
                    OperationStep(step_number=1, operation="DIJKSTRA", input_type="weighted_graph", output_type="shortest_path_tree"),
                ],
                expected_output_type="PATH_AND_COST",
                marks_estimate=7,
            )
            c2 = OperationChain(
                chain_id="chain_dijkstra_cost",
                bloom_level="L4",
                steps=[
                    OperationStep(step_number=1, operation="DIJKSTRA", input_type="weighted_graph", output_type="shortest_path_tree"),
                    OperationStep(step_number=2, operation="COMPUTE_COST", input_type="shortest_path", output_type="numeric"),
                ],
                expected_output_type="NUMERIC",
                marks_estimate=10,
            )
            chains.extend([c1, c2])

        elif start_op in ("EQUIVALENT_RESISTANCE", "KVL"):
            c1 = OperationChain(
                chain_id="chain_circuit_req",
                bloom_level="L3",
                steps=[
                    OperationStep(step_number=1, operation="EQUIVALENT_RESISTANCE", input_type="resistive_circuit", output_type="numeric"),
                ],
                expected_output_type="NUMERIC",
                marks_estimate=6,
            )
            chains.append(c1)

        elif start_op in ("REACTIONS", "SFD", "BMD"):
            c1 = OperationChain(
                chain_id="chain_beam_reactions",
                bloom_level="L3",
                steps=[
                    OperationStep(step_number=1, operation="REACTIONS", input_type="beam_structure", output_type="force_values"),
                ],
                expected_output_type="NUMERIC",
                marks_estimate=8,
            )
            chains.append(c1)

        elif start_op in ("INSERT_ROTATE", "INSERT"):
            c1 = OperationChain(
                chain_id="chain_avl_insert",
                bloom_level="L3",
                steps=[
                    OperationStep(step_number=1, operation="INSERT", input_type="avl_tree", output_type="modified_tree"),
                    OperationStep(step_number=2, operation="INSERT_ROTATE", input_type="unbalanced_tree", output_type="balanced_tree"),
                ],
                expected_output_type="TREE",
                marks_estimate=8,
            )
            chains.append(c1)

        else:
            c1 = OperationChain(
                chain_id=f"chain_generic_{start_op}",
                bloom_level="L3",
                steps=[OperationStep(step_number=1, operation=start_op, input_type="vko", output_type="generic")],
                expected_output_type="GENERIC",
                marks_estimate=5,
            )
            chains.append(c1)

        return chains

    @staticmethod
    def _score_and_filter_chains(chains: List[OperationChain], vko: VKO) -> List[OperationChain]:
        for c in chains:
            # Calculate quality score
            score = 0.85
            if c.bloom_level in ("L3", "L4"):
                score += 0.10
            if len(c.steps) > 1:
                score += 0.05
            c.score = round(score, 2)

        chains.sort(key=lambda c: c.score, reverse=True)
        return chains
