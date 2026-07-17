# learning_engine/memory/answer_memory.py
"""
Answer Memory — stores ideal expected answers as the foundation
for later question generation and grading.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AnswerRecord:
    concept_id: str
    question_text: str
    expected_answer: str
    answer_components: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    marks: int = 10
    verified: bool = False
    epoch_created: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AnswerMemory:
    def __init__(self, storage_path: str = None):
        self._answers: Dict[str, List[AnswerRecord]] = {}
        self._lock = threading.RLock()
        self._path = Path(storage_path) if storage_path else None

    def store(self, record: AnswerRecord):
        with self._lock:
            self._answers.setdefault(record.concept_id, []).append(record)

    def best_answer(self, concept_id: str) -> Optional[AnswerRecord]:
        answers = self._answers.get(concept_id, [])
        if not answers:
            return None
        return max(answers, key=lambda a: a.quality_score)

    def all_for_concept(self, concept_id: str) -> List[AnswerRecord]:
        return list(self._answers.get(concept_id, []))

    def average_quality(self) -> float:
        with self._lock:
            all_records = [a for answers in self._answers.values() for a in answers]
            if not all_records:
                return 0.0
            return round(sum(a.quality_score for a in all_records) / len(all_records), 4)

    def coverage(self, all_concept_ids: List[str]) -> float:
        if not all_concept_ids:
            return 0.0
        covered = sum(1 for cid in all_concept_ids if cid in self._answers)
        return round(covered / len(all_concept_ids), 4)

    def save(self):
        if not self._path:
            return
        with self._lock:
            data = {
                cid: [asdict(a) for a in answers]
                for cid, answers in self._answers.items()
            }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def load(self):
        if not self._path or not self._path.exists():
            return
        data = json.loads(self._path.read_text())
        with self._lock:
            for cid, answers in data.items():
                self._answers[cid] = [AnswerRecord(**a) for a in answers]
