"""
AION Module: Concept Memory Store (Knowledge Evolution Graph Layer)
Maturity:    v0.1 — PERSISTENT JSON CONCEPT STORE
Upgrades to: Multi-modal Vector & Graph Database (Postgres + Qdrant / Neo4j)
Contract:    Stores and retrieves Concept objects, handling deduplication & provenance evolution.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from .schemas import Concept


class ConceptMemoryStore:
    """
    Persistent memory layer for concepts across document uploads.
    Establishes the foundation for the Knowledge Evolution Graph.
    """

    def __init__(self, storage_path: str = "memory/concepts.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.concepts: Dict[str, Concept] = {}
        self.content_index: Dict[str, Concept] = {}
        self._load()

    def _load(self):
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    data = []
                else:
                    data = json.loads(content)
                for item in data:
                    concept = Concept(
                        concept_id=item.get("concept_id", ""),
                        content=item.get("content", ""),
                        confidence=item.get("confidence", 0.5),
                        definition_dna=item.get("definition_dna"),
                        relationship_dna=item.get("relationship_dna", []),
                        bloom_dna=item.get("bloom_dna"),
                        source_dna=item.get("source_dna"),
                        evolution_history=item.get("evolution_history", []),
                    )
                    self.concepts[concept.concept_id] = concept
                    self.content_index[concept.content.lower().strip()] = concept
        except Exception:
            self.concepts = {}
            self.content_index = {}
            self.save()

    def save(self):
        serialized = []
        for c in self.concepts.values():
            serialized.append({
                "concept_id": c.concept_id,
                "content": c.content,
                "confidence": c.confidence,
                "definition_dna": c.definition_dna,
                "relationship_dna": c.relationship_dna,
                "bloom_dna": c.bloom_dna,
                "source_dna": c.source_dna,
                "evolution_history": c.evolution_history,
            })
        try:
            temp_path = self.storage_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2, ensure_ascii=False)
            if self.storage_path.exists():
                try:
                    self.storage_path.unlink()
                except Exception:
                    pass
            temp_path.replace(self.storage_path)
        except Exception:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2, ensure_ascii=False)

    def find_duplicate(self, content: str) -> Optional[Concept]:
        """O(1) Hash-index lookup to detect existing concepts instantaneously."""
        content_clean = content.lower().strip()
        return self.content_index.get(content_clean)

    def upsert_concept(self, concept: Concept, save_now: bool = True) -> Concept:
        """
        Inserts a new concept or evolves an existing one if re-encountered.
        Updates provenance & evolution_history.
        """
        existing = self.find_duplicate(concept.content)
        now_str = datetime.now().isoformat()

        if existing:
            existing.confidence = min(1.0, round(existing.confidence + 0.1, 2))
            evolution_entry = {
                "timestamp": now_str,
                "action": "re_encountered",
                "source_dna": concept.source_dna,
                "boosted_confidence": existing.confidence,
            }
            existing.evolution_history.append(evolution_entry)
            if save_now:
                self.save()
            return existing
        else:
            concept.evolution_history.append({
                "timestamp": now_str,
                "action": "created",
                "source_dna": concept.source_dna,
                "initial_confidence": concept.confidence,
            })
            self.concepts[concept.concept_id] = concept
            self.content_index[concept.content.lower().strip()] = concept
            if save_now:
                self.save()
            return concept

    def get_all(self) -> List[Concept]:
        return list(self.concepts.values())

    def get_by_id(self, concept_id: str) -> Optional[Concept]:
        return self.concepts.get(concept_id)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_concepts": len(self.concepts),
            "storage_file": str(self.storage_path),
        }
