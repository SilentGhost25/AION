# learning_engine/progress/progress_tracker.py
"""
Progress Tracker — records and aggregates reports across learning epochs,
offering historical curve queries and saving files to disk.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

from learning_engine.progress.epoch_report import EpochReport


class ProgressTracker:
    def __init__(self, storage_path: str = None):
        self._reports: List[EpochReport] = []
        self._lock = threading.RLock()
        self._path = Path(storage_path) if storage_path else None

    def record_epoch(self, report: EpochReport):
        with self._lock:
            # Avoid duplicates of the same epoch index
            self._reports = [r for r in self._reports if r.epoch != report.epoch]
            self._reports.append(report)
            self._reports.sort(key=lambda r: r.epoch)

    def get_all(self) -> List[EpochReport]:
        with self._lock:
            return list(self._reports)

    def get_latest(self) -> Optional[EpochReport]:
        with self._lock:
            if self._reports:
                return self._reports[-1]
            return None

    def get_metric_history(self, metric_name: str) -> List[float]:
        """
        Extracts a list of floats corresponding to a specific metric over epochs.
        e.g., metric_name='concept_understanding'
        """
        with self._lock:
            history = []
            for r in self._reports:
                val = getattr(r, metric_name, 0.0)
                history.append(float(val))
            return history

    def clear(self):
        with self._lock:
            self._reports.clear()

    def save(self):
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = [r.to_dict() for r in self._reports]
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def load(self):
        if not self._path or not self._path.exists():
            return
        data = json.loads(self._path.read_text())
        with self._lock:
            self._reports = [EpochReport.from_dict(r) for r in data]
            self._reports.sort(key=lambda r: r.epoch)
