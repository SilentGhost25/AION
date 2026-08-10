"""
AION VTU Visual Policy
======================
Defines domain priors for VTU visual question generation.
"""

from __future__ import annotations

from typing import Dict

VISUAL_PREFERRED: Dict[str, bool] = {
    "dijkstra": True,
    "dijkstra_algorithm": True,
    "bfs": True,
    "dfs": True,
    "avl_rotation": True,
    "avl_tree": True,
    "bst_traversal": True,
    "weighted_graph": True,
    "circuit_analysis": True,
    "kvl": True,
    "kcl": True,
    "equivalent_resistance": True,
    "sfd_bmd": True,
    "beam_reactions": True,
    "a_star_search": True,
}

VISUAL_DISCOURAGED: Dict[str, bool] = {
    "definition": True,
    "definition_questions": True,
    "list": True,
    "explain": True,
    "describe": True,
    "time_complexity": True,
    "comparison_questions": True,
    "proof_questions": True,
}

BLOOM_VISUAL_PRIORS: Dict[str, float] = {
    "L1": 0.05,  # Usually NO IMAGE
    "L2": 0.15,  # Usually NO IMAGE
    "L3": 0.80,  # IMAGE strongly preferred if operation requires diagram
    "L4": 0.90,  # IMAGE strongly preferred for visual analytical operations
    "L5": 0.70,  # IMAGE when evaluation depends on visual transformation
    "L6": 0.40,  # IMAGE when design/construction requires it
}
