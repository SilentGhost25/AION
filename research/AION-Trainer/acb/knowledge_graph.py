# AION-Trainer/acb/knowledge_graph.py
"""
Course Knowledge Graph — manages the semantic linkages and prerequisite
dependencies between academic concepts.

Enables curriculum planning (topological sorting of concepts), prerequisite
path search, and detection of circular dependencies (cycles).
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from acb.concept import Concept, ConceptStore

logger = logging.getLogger("aion.acb.graph")


class KnowledgeGraph:
    def __init__(self, store: ConceptStore):
        self.store = store

    def get_prerequisites_recursive(self, concept_id: str, visited: Optional[Set[str]] = None) -> List[str]:
        """Collect all recursive prerequisite concept IDs in topological dependency order."""
        if visited is None:
            visited = set()
        concept = self.store.get(concept_id)
        if not concept or concept_id in visited:
            return []
        visited.add(concept_id)
        
        result = []
        for prereq_id in concept.prerequisites:
            result.extend(self.get_prerequisites_recursive(prereq_id, visited))
            result.append(prereq_id)
            
        # Return unique list maintaining order
        seen = set()
        return [x for x in result if not (x in seen or seen.add(x))]

    def find_cycles(self) -> List[List[str]]:
        """DFS-based cycle detection returning trace paths of all cycles found."""
        cycles = []
        # color state: 0 = unvisited, 1 = visiting, 2 = visited
        color = {c.concept_id: 0 for c in self.store.all_concepts()}
        parent = {}

        def dfs(node_id):
            color[node_id] = 1
            concept = self.store.get(node_id)
            if concept:
                for neighbor_id in concept.prerequisites:
                    # Initialize color if node is newly introduced
                    if neighbor_id not in color:
                        color[neighbor_id] = 0
                    if color[neighbor_id] == 0:
                        parent[neighbor_id] = node_id
                        dfs(neighbor_id)
                    elif color[neighbor_id] == 1:
                        # Trace back cycle
                        cycle = []
                        curr = node_id
                        while curr != neighbor_id:
                            cycle.append(curr)
                            curr = parent.get(curr)
                            if curr is None:
                                break
                        cycle.append(neighbor_id)
                        cycle.reverse()
                        cycles.append(cycle)
            color[node_id] = 2

        for concept in self.store.all_concepts():
            if color[concept.concept_id] == 0:
                dfs(concept.concept_id)
        return cycles

    def topological_sort(self) -> List[Concept]:
        """
        Sort concepts such that prerequisites appear before dependent concepts.
        If cycles are found, a best-effort sort is returned with log warnings.
        """
        cycles = self.find_cycles()
        if cycles:
            logger.warning(f"[KnowledgeGraph] Cycles detected in graph: {cycles}. Sorting best-effort.")

        visited = set()
        temp_visited = set()
        order = []

        def visit(concept_id):
            if concept_id in visited:
                return
            if concept_id in temp_visited:
                return  # break cycle loop
            temp_visited.add(concept_id)
            
            concept = self.store.get(concept_id)
            if concept:
                for prereq_id in concept.prerequisites:
                    visit(prereq_id)
                    
            temp_visited.remove(concept_id)
            visited.add(concept_id)
            if concept:
                order.append(concept)

        for concept in self.store.all_concepts():
            if concept.concept_id not in visited:
                visit(concept.concept_id)
                
        return order

    def add_dependency(self, concept_id: str, prerequisite_id: str):
        """Add a prerequisite dependency link."""
        concept = self.store.get(concept_id)
        prereq = self.store.get(prerequisite_id)
        if concept and prereq:
            if prerequisite_id not in concept.prerequisites:
                concept.prerequisites.append(prerequisite_id)
            if concept_id not in prereq.related_concepts:
                prereq.related_concepts.append(concept_id)
            concept.touch()
            prereq.touch()

    def remove_dependency(self, concept_id: str, prerequisite_id: str):
        """Remove a prerequisite dependency link."""
        concept = self.store.get(concept_id)
        prereq = self.store.get(prerequisite_id)
        if concept and prereq:
            if prerequisite_id in concept.prerequisites:
                concept.prerequisites.remove(prerequisite_id)
            if concept_id in prereq.related_concepts:
                prereq.related_concepts.remove(concept_id)
            concept.touch()
            prereq.touch()
