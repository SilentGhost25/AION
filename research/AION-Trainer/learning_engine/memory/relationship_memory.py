# learning_engine/memory/relationship_memory.py
"""
Relationship Memory — tracks which concept-to-concept relationships
AION has learned and how well it understands them.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RelationshipEntry:
    source_id: str
    target_id: str
    relation_type: str        # prerequisite | builds_on | contrasts_with | example_of
    strength: float = 0.0     # 0 = unknown, 1 = well understood
    learned: bool = False


class RelationshipMemory:
    def __init__(self, storage_path: str = None):
        self._edges: Dict[Tuple[str, str], RelationshipEntry] = {}
        self._lock = threading.RLock()
        self._path = Path(storage_path) if storage_path else None

    def record(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        strength: float = 0.5,
    ):
        with self._lock:
            key = (source_id, target_id)
            if key in self._edges:
                entry = self._edges[key]
                entry.strength = min(1.0, entry.strength * 0.7 + strength * 0.3)
                entry.learned = entry.strength >= 0.7
            else:
                self._edges[key] = RelationshipEntry(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    strength=strength,
                    learned=strength >= 0.7,
                )

    def get_related(self, concept_id: str) -> List[RelationshipEntry]:
        with self._lock:
            return [
                e for (src, _), e in self._edges.items()
                if src == concept_id
            ]

    def learned_count(self) -> int:
        return sum(1 for e in self._edges.values() if e.learned)

    def total_count(self) -> int:
        return len(self._edges)

    def save(self):
        if not self._path:
            return
        with self._lock:
            data = {f"{k[0]}|{k[1]}": asdict(v) for k, v in self._edges.items()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def load(self):
        if not self._path or not self._path.exists():
            return
        data = json.loads(self._path.read_text())
        with self._lock:
            for key_str, entry_data in data.items():
                src, tgt = key_str.split("|", 1)
                self._edges[(src, tgt)] = RelationshipEntry(**entry_data)
