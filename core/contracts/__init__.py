"""
AION Core Contracts Package
============================
Provides GenerationRequest (frontend->backend contract) and
PipelineTrace (per-request audit log) for the generate_stream route.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from core.contracts.text_chunk import TextChunk


# -- GenerationRequest ---------------------------------------------------------

class GenerationRequest:
    """
    Typed contract for every inbound /api/generate/stream request.
    Constructed via from_dict(); all fields have safe defaults so the
    route never crashes on a missing key.
    """

    def __init__(
        self,
        subject:     str,
        exam_type:   str,
        difficulty:  str,
        model:       str,
        mode:        str,
        file_id:     Optional[str],
        file_path:   Optional[str],
        notes_text:  Optional[str],
        department:  str,
        semester:    int,
        file_ids:    Optional[List[str]] = None,
    ) -> None:
        self.subject    = subject
        self.exam_type  = exam_type
        self.difficulty = difficulty
        self.model      = model
        self.mode       = mode
        self.file_id    = file_id
        self.file_ids   = file_ids
        self.file_path  = file_path
        self.notes_text = notes_text
        self.department = department
        self.semester   = semester

    # -- factory ---------------------------------------------------------------

    @classmethod
    def from_dict(cls, body: Dict[str, Any]) -> "GenerationRequest":
        from core.config.production_model import get_production_model  # lazy import

        file_ids_raw = body.get("file_ids") or body.get("fileIds")
        if isinstance(file_ids_raw, list):
            file_ids = [str(fid).strip() for fid in file_ids_raw if str(fid).strip()]
        else:
            file_ids = None

        return cls(
            subject    = (body.get("subject")    or "Unknown").strip(),
            exam_type  = (body.get("exam_type")  or body.get("examType") or "IA").strip(),
            difficulty = (body.get("difficulty") or "mixed").strip(),
            model      = (body.get("model")      or get_production_model()).strip(),
            mode       = (body.get("mode")       or "turbo").strip(),
            file_id    = body.get("file_id")     or body.get("fileId"),
            file_ids   = file_ids,
            file_path  = body.get("file_path"),
            notes_text = body.get("notes_text")  or body.get("notesText"),
            department = (body.get("department") or "Engineering").strip(),
            semester   = int(body.get("semester") or 5),
        )

    # -- validation ------------------------------------------------------------

    def validate(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not self.subject:
            errors.append("subject is required")
        allowed_exams = {"IA", "IAT1", "IAT2", "SEE"}
        if self.exam_type.upper() not in allowed_exams:
            errors.append(
                f"exam_type must be one of {sorted(allowed_exams)}, got '{self.exam_type}'"
            )
        return (len(errors) == 0), errors

    # -- diagnostics -----------------------------------------------------------

    def print_received_summary(self) -> None:
        print(
            f"[REQUEST] subject={self.subject!r}  exam={self.exam_type}  "
            f"diff={self.difficulty}  model={self.model}  mode={self.mode}  "
            f"file_id={self.file_id}  has_notes={'yes' if self.notes_text else 'no'}",
            flush=True,
        )


# -- PipelineTrace -------------------------------------------------------------

class PipelineTrace:
    """
    Lightweight per-request audit log.  Records each stage result and
    can persist to disk for post-mortem analysis.
    """

    LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "pipeline"

    def __init__(self, subject: str = "") -> None:
        self.request_id: str = str(uuid.uuid4())[:12]
        self.subject:    str = subject
        self.model:      str = ""
        self.started_at: str = datetime.now().isoformat()
        self.finished_at: Optional[str] = None
        self.status:     str = "running"        # running | complete | failed
        self.stages:     List[Dict[str, Any]] = []
        self.fail_reason: Optional[str] = None

    # -- stage recording -------------------------------------------------------

    def stage(
        self,
        name:        str,
        status:      str = "PASS",
        message:     str = "",
        metrics:     Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "stage":   name,
            "status":  status,
            "time":    datetime.now().isoformat(),
        }
        if message:
            entry["message"] = message
        if metrics:
            entry["metrics"] = metrics
        if duration_ms is not None:
            entry["duration_ms"] = round(duration_ms, 1)
        self.stages.append(entry)
        indicator = "PASS" if status == "PASS" else "FAIL"
        print(
            f"[TRACE {self.request_id}] {indicator} {name}"
            + (f" -- {message}" if message else ""),
            flush=True,
        )

    def complete(self) -> None:
        self.status      = "complete"
        self.finished_at = datetime.now().isoformat()

    def fail(self, reason: str) -> None:
        self.status      = "failed"
        self.fail_reason = reason
        self.finished_at = datetime.now().isoformat()

    # -- reporting -------------------------------------------------------------

    def print_summary(self) -> None:
        passed  = sum(1 for s in self.stages if s["status"] == "PASS")
        failed  = sum(1 for s in self.stages if s["status"] == "FAIL")
        elapsed = ""
        if self.finished_at:
            try:
                start = datetime.fromisoformat(self.started_at)
                end   = datetime.fromisoformat(self.finished_at)
                elapsed = f"  ({(end - start).total_seconds():.1f}s)"
            except Exception:
                pass
        print(
            f"[TRACE {self.request_id}] Summary: {self.status.upper()} "
            f"| stages passed={passed} failed={failed}{elapsed}",
            flush=True,
        )
        if self.fail_reason:
            print(f"[TRACE {self.request_id}] Failure: {self.fail_reason}", flush=True)

    def save_log(self) -> None:
        """Persist trace as JSON to logs/pipeline/. Silently skips on any error."""
        try:
            self.LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_file = self.LOG_DIR / f"{self.request_id}.json"
            payload = {
                "request_id":  self.request_id,
                "subject":     self.subject,
                "model":       self.model,
                "status":      self.status,
                "started_at":  self.started_at,
                "finished_at": self.finished_at,
                "fail_reason": self.fail_reason,
                "stages":      self.stages,
            }
            log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[TRACE] Log save failed: {exc}", flush=True)
