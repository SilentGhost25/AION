"""
AION VRE Hierarchical Operation Taxonomy (HOT)
================================================
Defines static domains, figure classes, operations, concept dependencies,
and grounding vocabularies.
"""

from __future__ import annotations

from typing import Dict, List, Set

# -- 1. Hierarchical Operation Taxonomy (HOT) -----------------------------------

HOT_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "DSA": {
        "WEIGHTED_GRAPH": [
            "DIJKSTRA",
            "BELLMAN_FORD",
            "PRIM",
            "KRUSKAL",
            "MST",
        ],
        "UNWEIGHTED_GRAPH": [
            "BFS",
            "DFS",
            "CONNECTED_CHECK",
            "CYCLE_DETECT",
        ],
        "BINARY_TREE": [
            "INSERT",
            "DELETE",
            "SEARCH",
            "TRAVERSE",
            "HEIGHT",
        ],
        "AVL_TREE": [
            "INSERT_ROTATE",
            "DELETE_ROTATE",
            "HEIGHT",
            "BALANCE_FACTOR",
        ],
        "HEAP": [
            "INSERT",
            "DELETE_MAX",
            "HEAPIFY",
            "BUILD_HEAP",
        ],
    },
    "AI": {
        "STATE_SPACE_GRAPH": [
            "A_STAR",
            "BFS",
            "DFS",
            "HEURISTIC_COMPUTE",
        ],
        "EIGHT_PUZZLE_GRID": [
            "MANHATTAN",
            "MISPLACED",
            "A_STAR_SOLVE",
            "MOVE_SEQUENCE",
        ],
        "DECISION_TREE": [
            "CLASSIFY",
            "ENTROPY",
            "INFORMATION_GAIN",
        ],
    },
    "ECE": {
        "CIRCUIT_RESISTIVE": [
            "KVL",
            "KCL",
            "THEVENIN",
            "NORTON",
            "EQUIVALENT_RESISTANCE",
        ],
        "CIRCUIT_RLC": [
            "IMPEDANCE",
            "RESONANCE",
            "FREQUENCY_RESPONSE",
        ],
        "LOGIC_CIRCUIT": [
            "TRUTH_TABLE",
            "BOOLEAN_SIMPLIFY",
            "TIMING_DIAGRAM",
        ],
        "BLOCK_DIAGRAM": [
            "REDUCE",
            "TRANSFER_FUNCTION",
            "STABILITY",
        ],
    },
    "CIVIL": {
        "BEAM": [
            "REACTIONS",
            "SFD",
            "BMD",
            "DEFLECTION",
        ],
        "TRUSS": [
            "METHOD_OF_JOINTS",
            "METHOD_OF_SECTIONS",
            "FORCE_MEMBER",
        ],
    },
}

# -- 2. Visual Dependency Mappings ----------------------------------------------

CONCEPT_VISUAL_DEPENDENCY: Dict[str, str] = {
    # ALWAYS needs image
    "dijkstra_algorithm": "ALWAYS",
    "dijkstra": "ALWAYS",
    "avl_rotations": "ALWAYS",
    "avl_tree": "ALWAYS",
    "a_star_search": "ALWAYS",
    "8_puzzle": "ALWAYS",
    "circuit_analysis": "ALWAYS",
    "thevenin_theorem": "ALWAYS",
    "block_diagram_reduction": "ALWAYS",
    "sfd_bmd": "ALWAYS",
    "beam_reactions": "ALWAYS",

    # FREQUENT
    "minimum_spanning_tree": "FREQUENT",
    "tree_traversal": "FREQUENT",
    "graph_coloring": "FREQUENT",
    "logic_gates": "FREQUENT",

    # SOMETIMES
    "sorting_algorithms": "SOMETIMES",
    "dynamic_programming": "SOMETIMES",

    # RARELY / NEVER
    "definition_questions": "NEVER",
    "comparison_questions": "NEVER",
    "proof_questions": "NEVER",
    "time_complexity": "NEVER",
}

BLOOM_LEVEL_VISUAL_TENDENCY: Dict[str, float] = {
    "L1": 0.05,
    "L2": 0.15,
    "L3": 0.55,
    "L4": 0.70,
    "L5": 0.45,
    "L6": 0.20,
}

# -- 3. Grounding Vocabulary ----------------------------------------------------

GROUNDING_VOCABULARY: Dict[str, Dict[str, List[str]]] = {
    "WEIGHTED_GRAPH": {
        "REQUIRED": ["source_node", "graph"],
        "RECOMMENDED": ["node_count", "edge_weights"],
        "FORBIDDEN": ["describe the figure", "observe the diagram", "what is shown"],
    },
    "AVL_TREE": {
        "REQUIRED": ["keys", "avl tree"],
        "RECOMMENDED": ["balance factor", "rotations"],
        "FORBIDDEN": ["explain the picture", "describe the tree"],
    },
    "CIRCUIT_RESISTIVE": {
        "REQUIRED": ["circuit", "resistance"],
        "RECOMMENDED": ["voltage source", "nodes"],
        "FORBIDDEN": ["describe the circuit", "explain the diagram"],
    },
    "BEAM": {
        "REQUIRED": ["beam", "span"],
        "RECOMMENDED": ["loads", "reactions"],
        "FORBIDDEN": ["describe the beam", "observe the figure"],
    },
}

SUPPORTED_FIGURE_CLASSES: Set[str] = {
    "WEIGHTED_GRAPH",
    "UNWEIGHTED_GRAPH",
    "BINARY_TREE",
    "AVL_TREE",
    "HEAP",
    "STATE_SPACE_GRAPH",
    "EIGHT_PUZZLE_GRID",
    "DECISION_TREE",
    "CIRCUIT_RESISTIVE",
    "CIRCUIT_RLC",
    "LOGIC_CIRCUIT",
    "BLOCK_DIAGRAM",
    "BEAM",
    "TRUSS",
}
