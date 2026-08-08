"""
AION v2 Pipeline Trace System
=============================
Event-driven observability system recording stage events across the generation pipeline.
Saves persistent JSON trace logs to logs/YYYY-MM-DD/request_<request_id>.json.

Production-safe. Zero laptop-specific code.
"""

import os
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class StageEvent:
    stage_name:   str
    status:       str             # PASS, FAIL, SKIP, WARN
    duration_ms:  float = 0.0
    metrics:      Dict[str, Any] = field(default_factory=dict)
    reason_code:  Optional[str]  = None
    message:      Optional[str]  = None
    timestamp:    str            = field(default_factory=lambda: datetime.now().isoformat())


class PipelineTrace:
    """
    Heart of AION Pipeline Observability.
    Collects stage execution events and persists JSON logs.
    """

    def __init__(self, request_id: Optional[str] = None, subject: str = "Unknown"):
        self.request_id: str = request_id or str(uuid.uuid4())[:8]
        self.subject: str = subject
        self.created_at: str = datetime.now().isoformat()
        self.start_time: float = time.time()
        self.events: List[StageEvent] = []
        self.status: str = "RUNNING"
        self.model: str = "qwen2.5:14b"
        self.error: Optional[str] = None

    def stage(
        self,
        stage_name: str,
        status: str = "PASS",
        duration_ms: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
        reason_code: Optional[str] = None,
        message: Optional[str] = None,
    ):
        """Record an event for a pipeline stage."""
        event = StageEvent(
            stage_name  = stage_name,
            status      = status.upper(),
            duration_ms = round(duration_ms, 2),
            metrics     = metrics or {},
            reason_code = reason_code,
            message     = message,
        )
        self.events.append(event)
        if status.upper() == "FAIL":
            self.status = "FAILED"

    def complete(self, status: str = "COMPLETED"):
        """Mark overall trace as completed."""
        if self.status != "FAILED":
            self.status = status

    def fail(self, error_message: str):
        """Mark overall trace as failed with error message."""
        self.status = "FAILED"
        self.error = error_message
        self.stage("PipelineError", status="FAIL", message=error_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to structured dictionary."""
        total_duration = round((time.time() - self.start_time) * 1000, 2)
        return {
            "request_id":     self.request_id,
            "subject":        self.subject,
            "status":         self.status,
            "model":          self.model,
            "created_at":     self.created_at,
            "total_duration_ms": total_duration,
            "error":          self.error,
            "stage_results":  [asdict(e) for e in self.events],
        }

    def print_summary(self):
        """Render clean ASCII summary box in server terminal."""
        total_duration = round((time.time() - self.start_time) * 1000, 2)
        print("\n========================================================")
        print(f"               PIPELINE TRACE SUMMARY ({self.request_id})")
        print("========================================================")
        print(f"  Subject        : {self.subject}")
        print(f"  Overall Status : {self.status}")
        print(f"  Total Duration : {total_duration} ms")
        print("--------------------------------------------------------")

        for event in self.events:
            symbol = "✓" if event.status == "PASS" else ("✗" if event.status == "FAIL" else "○")
            metric_str = ", ".join(f"{k}={v}" for k, v in event.metrics.items()) if event.metrics else ""
            msg = f" ({event.message})" if event.message else ""
            reason = f" [{event.reason_code}]" if event.reason_code else ""

            print(f"  {event.stage_name:<20} {symbol} {event.status:<5} {event.duration_ms:>7.1f} ms {metric_str}{reason}{msg}")

        print("========================================================\n")

    def save_log(self, base_dir: str = "logs") -> str:
        """Write trace log to logs/YYYY-MM-DD/request_<id>.json."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            log_dir = Path(base_dir) / today
            log_dir.mkdir(parents=True, exist_ok=True)

            log_path = log_dir / f"request_{self.request_id}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)

            return str(log_path)
        except Exception as e:
            print(f"[PIPELINE TRACE ERROR] Failed to save trace log: {e}")
            return ""
