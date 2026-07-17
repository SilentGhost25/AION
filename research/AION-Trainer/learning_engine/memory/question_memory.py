# learning_engine/memory/question_memory.py
"""
Question Memory — stores every question AION has generated with
its full academic metadata, outcome, and professor feedback.
"""

from __future__ import annotations

import json
import uuid
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class QuestionRecord:
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    concept_id: str = ""
    question_text: str = ""
    bloom_level: str = ""
    marks: int = 0
    difficulty: str = ""
    question_type: str = ""
    grammar_score: float = 0.0
    vtu_style_score: float = 0.0
    bloom_alignment: float = 0.0
    novelty_score: float = 0.0
    accepted: bool = False
    rejection_reason: str = ""
    professor_edited: bool = False
    edited_text: str = ""
    epoch: int = 0
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class QuestionMemory:
    def __init__(self, storage_path: str = None):
        self._records: Dict[str, QuestionRecord] = {}
        self._lock = threading.RLock()
        self._path = Path(storage_path) if storage_path else None

    def store(self, record: QuestionRecord) -> str:
        with self._lock:
            self._records[record.record_id] = record
        return record.record_id

    def get(self, record_id: str) -> Optional[QuestionRecord]:
        return self._records.get(record_id)

    def for_concept(self, concept_id: str) -> List[QuestionRecord]:
        return [r for r in self._records.values() if r.concept_id == concept_id]

    def accepted_texts(self, concept_id: str) -> List[str]:
        return [r.question_text for r in self.for_concept(concept_id) if r.accepted]

    def acceptance_rate(self) -> float:
        with self._lock:
            if not self._records:
                return 0.0
            accepted = sum(1 for r in self._records.values() if r.accepted)
            return round(accepted / len(self._records), 4)

    def average_grammar_score(self) -> float:
        with self._lock:
            records = list(self._records.values())
            if not records:
                return 0.0
            return round(sum(r.grammar_score for r in records) / len(records), 4)

    def total_generated(self) -> int:
        return len(self._records)

    def total_accepted(self) -> int:
        return sum(1 for r in self._records.values() if r.accepted)

    def save(self):
        if not self._path:
            return
        with self._lock:
            data = {rid: asdict(r) for rid, r in self._records.items()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def load(self):
        if not self._path or not self._path.exists():
            return
        data = json.loads(self._path.read_text())
        with self._lock:
            for rid, r_data in data.items():
                self._records[rid] = QuestionRecord(**r_data)
