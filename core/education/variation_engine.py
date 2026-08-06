"""
Pedagogical Variation Engine — Tracks and diversifies question archetypes
Per audit: If I upload BST today and again tomorrow, model should produce
Numerical, Comparison, Proof, Application, Case Study, Debugging... without repeating.

Requires dedicated variation planner with memory of previously used archetypes.
"""

from typing import List, Dict, Set
from collections import defaultdict
import json
import pathlib

class VariationEngine:
    """Tracks used archetypes per concept and ensures diversity."""
    
    def __init__(self, memory_path: str = "memory/variation.json"):
        self.memory_path = pathlib.Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.used_archetypes: Dict[str, Set[str]] = defaultdict(set)
        self._load()
    
    def _load(self):
        if self.memory_path.exists():
            try:
                data = json.loads(self.memory_path.read_text())
                for k, v in data.items():
                    self.used_archetypes[k] = set(v)
            except:
                pass
    
    def _save(self):
        try:
            data = {k: list(v) for k, v in self.used_archetypes.items()}
            self.memory_path.write_text(json.dumps(data, indent=2))
        except:
            pass
    
    def get_used_for_concept(self, concept_id: str) -> Set[str]:
        return self.used_archetypes.get(concept_id, set())
    
    def mark_used(self, concept_id: str, archetype: str):
        self.used_archetypes[concept_id].add(archetype)
        self._save()
    
    def choose_archetype(self, concept_id: str, available: List[str]) -> str:
        """Choose least-used archetype for concept."""
        used = self.get_used_for_concept(concept_id)
        unused = [a for a in available if a not in used]
        if unused:
            chosen = unused[0]
        else:
            # All used, reset and pick first
            self.used_archetypes[concept_id] = set()
            chosen = available[0]
        self.mark_used(concept_id, chosen)
        return chosen
    
    def get_variation_stats(self) -> Dict:
        return {k: list(v) for k, v in self.used_archetypes.items()}
