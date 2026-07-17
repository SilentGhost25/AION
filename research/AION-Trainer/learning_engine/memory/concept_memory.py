# learning_engine/memory/concept_memory.py
"""
Concept Memory — stores what AION knows about each concept,
how well it knows it (multi-dimensional confidence), and
the current learning stage.

Integrates with the existing Concept Store but adds the learning
layer on top without modifying any ACB or ESE code.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from learning_engine.stages import ConceptStage


@dataclass
class ConceptConfidence:
    """
    Multi-dimensional confidence for one concept.
    Each dimension represents a different cognitive ability.
    """
    understand: float  = 0.0   # Can AION define / paraphrase it?
    explain:    float  = 0.0   # Can AION write a coherent explanation?
    compare:    float  = 0.0   # Can AION contrast with related concepts?
    generate_question: float = 0.0   # Can AION write a valid exam question?
    generate_answer:   float = 0.0   # Can AION write a correct expected answer?
    predict_exam_use:  float = 0.0   # Does AION know when examiners ask this?

    def overall(self) -> float:
        vals = [
            self.understand, self.explain, self.compare,
            self.generate_question, self.generate_answer, self.predict_exam_use,
        ]
        return sum(vals) / len(vals)

    def weakest_dimension(self) -> str:
        dims = {
            "understand": self.understand,
            "explain": self.explain,
            "compare": self.compare,
            "generate_question": self.generate_question,
            "generate_answer": self.generate_answer,
            "predict_exam_use": self.predict_exam_use,
        }
        return min(dims, key=dims.get)

    def update(self, dimension: str, value: float, alpha: float = 0.3):
        """Exponential moving average update — prevents single-pass overfit."""
        current = getattr(self, dimension, 0.0)
        setattr(self, dimension, round(current * (1 - alpha) + value * alpha, 4))


@dataclass
class ConceptMemoryEntry:
    concept_id: str
    name: str
    subject_code: str
    module: int
    stage: int = ConceptStage.DISCOVERED
    confidence: ConceptConfidence = field(default_factory=ConceptConfidence)
    times_studied: int = 0
    times_revisited: int = 0
    last_studied_at: Optional[str] = None
    weak_flag: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def current_stage(self) -> ConceptStage:
        return ConceptStage(self.stage)

    def advance_stage(self) -> bool:
        """Advance to next stage if not already at maximum. Returns True if advanced."""
        if self.stage < ConceptStage.EXAMINER_LEVEL:
            self.stage += 1
            return True
        return False

    def needs_revisit(self, threshold: float = 0.70) -> bool:
        return self.confidence.overall() < threshold or self.weak_flag

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptMemoryEntry":
        conf_data = data.pop("confidence", {})
        entry = cls(**data)
        if conf_data:
            entry.confidence = ConceptConfidence(**conf_data)
        return entry


class ConceptMemory:
    """
    Thread-safe store for concept learning state.

    Designed to wrap the existing ConceptStore — it does not replace
    it. The ConceptStore owns academic content; ConceptMemory owns
    learning progress.
    """

    REVISIT_THRESHOLD = 0.70

    def __init__(self, storage_path: str = None):
        self._entries: Dict[str, ConceptMemoryEntry] = {}
        self._lock = threading.RLock()
        self._path = Path(storage_path) if storage_path else None

    def initialise_from_store(self, concept_store, subject_code: str):
        """Bootstrap memory from an existing ConceptStore."""
        with self._lock:
            for concept in concept_store.concepts_for_subject(subject_code):
                if concept.concept_id not in self._entries:
                    entry = ConceptMemoryEntry(
                        concept_id=concept.concept_id,
                        name=concept.name,
                        subject_code=subject_code,
                        module=concept.primary_module() or 0,
                    )
                    self._entries[concept.concept_id] = entry

    def get(self, concept_id: str) -> Optional[ConceptMemoryEntry]:
        return self._entries.get(concept_id)

    def update_confidence(
        self,
        concept_id: str,
        dimension: str,
        value: float,
        alpha: float = 0.3,
    ):
        with self._lock:
            entry = self._entries.get(concept_id)
            if entry:
                entry.confidence.update(dimension, value, alpha)
                entry.last_studied_at = datetime.utcnow().isoformat()

    def advance_stage(self, concept_id: str) -> Optional[ConceptStage]:
        with self._lock:
            entry = self._entries.get(concept_id)
            if entry and entry.advance_stage():
                return entry.current_stage
        return None

    def mark_weak(self, concept_id: str, note: str = ""):
        with self._lock:
            entry = self._entries.get(concept_id)
            if entry:
                entry.weak_flag = True
                if note:
                    entry.notes.append(note)

    def clear_weak(self, concept_id: str):
        with self._lock:
            entry = self._entries.get(concept_id)
            if entry:
                entry.weak_flag = False

    def record_study(self, concept_id: str, is_revisit: bool = False):
        with self._lock:
            entry = self._entries.get(concept_id)
            if entry:
                if is_revisit:
                    entry.times_revisited += 1
                else:
                    entry.times_studied += 1
                entry.last_studied_at = datetime.utcnow().isoformat()

    def weak_concepts(self, subject_code: str = None) -> List[ConceptMemoryEntry]:
        with self._lock:
            entries = list(self._entries.values())
            if subject_code:
                entries = [e for e in entries if e.subject_code == subject_code]
            return [e for e in entries if e.needs_revisit(self.REVISIT_THRESHOLD)]

    def concepts_at_stage(
        self, stage: ConceptStage, subject_code: str = None
    ) -> List[ConceptMemoryEntry]:
        with self._lock:
            entries = list(self._entries.values())
            if subject_code:
                entries = [e for e in entries if e.subject_code == subject_code]
            return [e for e in entries if e.current_stage == stage]

    def all_for_module(self, subject_code: str, module: int) -> List[ConceptMemoryEntry]:
        with self._lock:
            return [
                e for e in self._entries.values()
                if e.subject_code == subject_code and e.module == module
            ]

    def save(self):
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {cid: e.to_dict() for cid, e in self._entries.items()}
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def load(self):
        if not self._path or not self._path.exists():
            return
        data = json.loads(self._path.read_text())
        with self._lock:
            for cid, entry_data in data.items():
                self._entries[cid] = ConceptMemoryEntry.from_dict(entry_data)
