# learning_engine/memory/mistake_memory.py
"""
Mistake Memory — records every rejected question, why it was rejected,
and whether AION corrected the error in subsequent epochs.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MistakeEntry:
    mistake_id: str
    concept_id: str
    generated_text: str
    rejection_reason: str
    rejection_categories: List[str] = field(default_factory=list)
    epoch: int = 0
    corrected: bool = False
    corrected_in_epoch: Optional[int] = None
    recorded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class MistakeMemory:
    def __init__(self, storage_path: str = None):
        self._mistakes: Dict[str, MistakeEntry] = {}
        self._lock = threading.RLock()
        self._path = Path(storage_path) if storage_path else None

    def record(
        self,
        mistake_id: str,
        concept_id: str,
        generated_text: str,
        reason: str,
        categories: List[str],
        epoch: int,
    ) -> MistakeEntry:
        entry = MistakeEntry(
            mistake_id=mistake_id,
            concept_id=concept_id,
            generated_text=generated_text,
            rejection_reason=reason,
            rejection_categories=categories,
            epoch=epoch,
        )
        with self._lock:
            self._mistakes[mistake_id] = entry
        return entry

    def mark_corrected(self, mistake_id: str, epoch: int):
        with self._lock:
            if mistake_id in self._mistakes:
                self._mistakes[mistake_id].corrected = True
                self._mistakes[mistake_id].corrected_in_epoch = epoch

    def uncorrected_for_concept(self, concept_id: str) -> List[MistakeEntry]:
        with self._lock:
            return [
                m for m in self._mistakes.values()
                if m.concept_id == concept_id and not m.corrected
            ]

    def category_frequencies(self) -> Dict[str, int]:
        with self._lock:
            freq: Dict[str, int] = {}
            for m in self._mistakes.values():
                for cat in m.rejection_categories:
                    freq[cat] = freq.get(cat, 0) + 1
            return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

    def correction_rate(self) -> float:
        with self._lock:
            if not self._mistakes:
                return 1.0
            corrected = sum(1 for m in self._mistakes.values() if m.corrected)
            return round(corrected / len(self._mistakes), 4)

    def save(self):
        if not self._path:
            return
        with self._lock:
            data = {mid: asdict(m) for mid, m in self._mistakes.items()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def load(self):
        if not self._path or not self._path.exists():
            return
        data = json.loads(self._path.read_text())
        with self._lock:
            for mid, m_data in data.items():
                self._mistakes[mid] = MistakeEntry(**m_data)
