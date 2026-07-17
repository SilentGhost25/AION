# learning_engine/memory/examiner_memory.py
"""
Examiner Memory — learns professor preferences from PYQs and
question banks: which verbs they prefer, at which marks, for
which topics.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExaminerPattern:
    verb: str
    marks: int
    bloom_level: str
    frequency: int = 1
    concept_types: List[str] = field(default_factory=list)


class ExaminerMemory:
    def __init__(self, storage_path: str = None):
        self._patterns: Dict[str, ExaminerPattern] = {}
        self._verb_freq: Dict[str, int] = defaultdict(int)
        self._marks_freq: Dict[int, int] = defaultdict(int)
        self._lock = threading.RLock()
        self._path = Path(storage_path) if storage_path else None

    def observe(self, verb: str, marks: int, bloom: str, concept_type: str = ""):
        with self._lock:
            key = f"{verb}_{marks}_{bloom}"
            if key in self._patterns:
                self._patterns[key].frequency += 1
                if concept_type and concept_type not in self._patterns[key].concept_types:
                    self._patterns[key].concept_types.append(concept_type)
            else:
                self._patterns[key] = ExaminerPattern(
                    verb=verb, marks=marks, bloom_level=bloom,
                    concept_types=[concept_type] if concept_type else [],
                )
            self._verb_freq[verb.lower()] += 1
            self._marks_freq[marks] += 1

    def preferred_verb(self, marks: int) -> str:
        with self._lock:
            candidates = [
                (p.verb, p.frequency) for p in self._patterns.values()
                if p.marks == marks
            ]
            if candidates:
                return max(candidates, key=lambda x: x[1])[0]
            # Fallback
            return "Explain" if marks >= 10 else "Define"

    def preferred_marks(self) -> List[int]:
        with self._lock:
            return sorted(self._marks_freq, key=self._marks_freq.get, reverse=True)[:5]

    def verb_distribution(self) -> Dict[str, float]:
        with self._lock:
            total = sum(self._verb_freq.values()) or 1
            return {v: round(c / total, 4) for v, c in self._verb_freq.items()}

    def save(self):
        if not self._path:
            return
        with self._lock:
            data = {
                "patterns": {k: asdict(v) for k, v in self._patterns.items()},
                "verb_freq": dict(self._verb_freq),
                "marks_freq": {str(k): v for k, v in self._marks_freq.items()},
            }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def load(self):
        if not self._path or not self._path.exists():
            return
        data = json.loads(self._path.read_text())
        with self._lock:
            for k, v in data.get("patterns", {}).items():
                self._patterns[k] = ExaminerPattern(**v)
            self._verb_freq.update(data.get("verb_freq", {}))
            self._marks_freq.update({int(k): v for k, v in data.get("marks_freq", {}).items()})
