# learning_engine/stages/relationship_builder.py
"""
Relationship Builder Stage — computes transitive links, clusters related
concepts, and updates relationship strength.
"""

from __future__ import annotations

import logging
from typing import List

from learning_engine.memory.relationship_memory import RelationshipMemory
from learning_engine.memory.concept_memory import ConceptMemory

logger = logging.getLogger("aion.learning.relationship_builder")


class RelationshipBuilder:
    """
    Analyzes learned concept paths to build transitive relationships and
    detect topological cluster structures in AION's academic graph.
    """

    def __init__(
        self,
        concept_memory: ConceptMemory,
        relationship_memory: RelationshipMemory,
    ):
        self.concept_memory = concept_memory
        self.relationship_memory = relationship_memory

    def analyze_graph(self, subject_code: str) -> int:
        """
        Runs a topological pass across relationships to compute transitive strengths.
        For example: if A -> B and B -> C, then A has a weak transitive relationship to C.
        """
        logger.info(f"[RelationshipBuilder] Starting graph analysis for {subject_code}")
        
        # We can extract all relationship entries, find paths of length 2, and add transitive links
        all_edges = []
        # Since RelationshipMemory does not directly expose all edges keys, we can inspect private _edges safely.
        with self.relationship_memory._lock:
            all_edges = list(self.relationship_memory._edges.values())

        inferred = 0
        for edge_ab in all_edges:
            for edge_bc in all_edges:
                if edge_ab.target_id == edge_bc.source_id and edge_ab.source_id != edge_bc.target_id:
                    # Found path: A -> B -> C. Infer transitive relation A -> C
                    # Only infer if strength of both is high
                    if edge_ab.learned and edge_bc.learned:
                        strength = round(edge_ab.strength * edge_bc.strength * 0.5, 4)
                        self.relationship_memory.record(
                            edge_ab.source_id,
                            edge_bc.target_id,
                            "transitive_builds_on",
                            strength=strength,
                        )
                        inferred += 1

        logger.info(f"[RelationshipBuilder] Inferred {inferred} transitive relationship links.")
        return inferred
