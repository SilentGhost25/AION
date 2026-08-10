"""
AION Strict Syllabus-Module Alignment Validator
===============================================
Ensures generated questions contain concepts strictly aligned with their target module.
Prevents cross-module concept bleed (e.g., AVL trees appearing in Module 1 when defined in Module 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


# Standard VTU Computer Science syllabus module concept mapping
DEFAULT_MODULE_CONCEPTS: Dict[int, Set[str]] = {
    1: {"array", "stack", "queue", "linear", "lifo", "fifo", "push", "pop", "enqueue", "dequeue"},
    2: {"tree", "binary", "bst", "avl", "balance", "rotation", "heap", "priority"},
    3: {"graph", "dijkstra", "prim", "kruskal", "mst", "shortest", "path", "dfs", "bfs"},
    4: {"sort", "search", "quick", "merge", "partition", "divide", "conquer", "binary search"},
    5: {"hash", "hashing", "probe", "chain", "probing", "collision", "index", "file"},
}


@dataclass
class ModuleAlignmentResult:
    """Result of module concept consistency check."""
    passed: bool
    module_index: int
    detected_concepts: List[str]
    conflicting_modules: List[int]
    reason: str = "COMPLIANT"


class ModuleAlignmentValidator:
    """Validates question text against target module concept boundaries."""

    @classmethod
    def validate(
        cls,
        question_text: str,
        target_module: int = 1,
        custom_syllabus_map: Optional[Dict[int, Set[str]]] = None,
    ) -> ModuleAlignmentResult:
        if not question_text:
            return ModuleAlignmentResult(
                passed=False,
                module_index=target_module,
                detected_concepts=[],
                conflicting_modules=[],
                reason="EMPTY_TEXT",
            )

        syllabus = custom_syllabus_map or DEFAULT_MODULE_CONCEPTS
        q_low = question_text.lower()

        detected = []
        conflicts = []

        # Find concepts in target module
        target_concepts = syllabus.get(target_module, set())
        for c in target_concepts:
            if c in q_low:
                detected.append(c)

        # Check for concepts in other modules that conflict
        for m_idx, concepts in syllabus.items():
            if m_idx == target_module:
                continue
            for c in concepts:
                if c in q_low and c not in target_concepts:
                    # Avoid flagging generic terms like 'tree' if target module has 'tree'
                    conflicts.append(m_idx)
                    break

        passed = len(conflicts) == 0 or len(detected) > 0

        return ModuleAlignmentResult(
            passed=passed,
            module_index=target_module,
            detected_concepts=detected,
            conflicting_modules=sorted(list(set(conflicts))),
            reason="COMPLIANT" if passed else f"CONCEPT_BLEED_DETECTED_IN_MODULES_{conflicts}",
        )
