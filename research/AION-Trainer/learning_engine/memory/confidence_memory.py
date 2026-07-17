# learning_engine/memory/confidence_memory.py
"""
Confidence Memory — logs historical confidence curves for each concept
over epochs, enabling analytics on study improvement.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ConfidenceSnapshot:
    epoch: int
    understand: float
    explain: float
    compare: float
    generate_question: float
    generate_answer: float
    predict_exam_use: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ConfidenceMemory:
    """
    Logs and maintains the history of confidence dimensions per concept over epochs.
    """

    def __init__(self, storage_path: str = None):
        self._history: Dict[str, List[ConfidenceSnapshot]] = {}
        self._lock = threading.RLock()
        self._path = Path(storage_path) if storage_path else None

    def record_snapshot(
        self,
        concept_id: str,
        epoch: int,
        understand: float,
        explain: float,
        compare: float,
        generate_question: float,
        generate_answer: float,
        predict_exam_use: float,
    ):
        snapshot = ConfidenceSnapshot(
            epoch=epoch,
            understand=understand,
            explain=explain,
            compare=compare,
            generate_question=generate_question,
            generate_answer=generate_answer,
            predict_exam_use=predict_exam_use,
        )
        with self._lock:
            self._history.setdefault(concept_id, []).append(snapshot)

    def get_history(self, concept_id: str) -> List[ConfidenceSnapshot]:
        with self._lock:
            return list(self._history.get(concept_id, []))

    def get_latest(self, concept_id: str) -> Optional[ConfidenceSnapshot]:
        with self._lock:
            history = self._history.get(concept_id)
            if history:
                return history[-1]
            return None

    def save(self):
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                cid: [asdict(snap) for snap in snaps]
                for cid, snaps in self._history.items()
            }
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def load(self):
        if not self._path or not self._path.exists():
            return
        data = json.loads(self._path.read_text())
        with self._lock:
            for cid, snaps_data in data.items():
                self._history[cid] = [ConfidenceSnapshot(**snap) for snap in snaps_data]
