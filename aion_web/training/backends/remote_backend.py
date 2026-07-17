# aion_web/training/backends/remote_backend.py
"""
Remote Backend — delegates to the AION Training Server via HTTP.
The GUI becomes a thin client; all heavy computation runs on the
server's GPU.

Rules:
    - Sends real files to the real server
    - Streams real progress via SSE
    - NEVER uses mock data
    - On failure: raises BackendError with the server's error message
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import requests

from aion_web.training.backends.base import (
    TrainingBackend, TrainingMode, BackendError,
    JobHandle, ProgressEvent, AnalysisOutput, TrainingOutput,
)


class RemoteBackend(TrainingBackend):
    """
    Thin HTTP client that delegates to the AION Training Server.
    """

    mode = TrainingMode.REMOTE

    def __init__(self, server_url: str, token: str, timeout: int = 30):
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {token}"})
        self._active_job_id: Optional[str] = None
        self._lock = threading.Lock()

    # ── Core operations ───────────────────────────────────────────────

    def analyse(
        self,
        file_paths: List[str],
        subject_code: str = "",
    ) -> AnalysisOutput:
        try:
            # Create session on server
            r = self._post("/studio/session")
            session_id = r["session_id"]

            # Upload files
            for file_path in file_paths:
                self._upload_file(session_id, file_path)

            # Trigger analysis
            result = self._post(f"/studio/session/{session_id}/analyse")

            # Fetch full analysis
            analysis = self._get(f"/studio/session/{session_id}/analysis")
            preview = self._get(f"/studio/session/{session_id}/preview")
            ambiguities_data = self._get(f"/studio/session/{session_id}/ambiguities")

            return AnalysisOutput(
                session_id=session_id,
                subject_code=analysis.get("subject_code", subject_code),
                subject_name=analysis.get("subject_name", ""),
                department="",
                semester=0,
                books=result.get("books", 0),
                notes=result.get("notes", 0),
                question_banks=result.get("question_banks", 0),
                previous_papers=result.get("previous_papers", 0),
                module_summaries=preview.get("modules", []),
                ambiguities=ambiguities_data.get("ambiguities", []),
                train_enabled=result.get("train_enabled", False),
                mode=TrainingMode.REMOTE,
            )
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"Remote analysis failed: {e}") from e

    def train(self, session_id: str, subject_code: str) -> JobHandle:
        try:
            # Confirm course then start training
            self._post(f"/studio/session/{session_id}/confirm")
            result = self._post(f"/studio/session/{session_id}/train")

            job_id = result.get("job_id")
            if not job_id:
                raise BackendError("Server did not return a job_id")

            with self._lock:
                self._active_job_id = job_id

            return JobHandle(
                job_id=job_id,
                mode=self.mode,
                subject_code=subject_code,
                started_at=datetime.utcnow().isoformat(),
            )
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"Failed to start remote training: {e}") from e

    def get_progress(self, job_id: str) -> Generator[ProgressEvent, None, None]:
        url = f"{self.server_url}/jobs/{job_id}/stream"
        try:
            with self._session.get(url, stream=True, timeout=None) as r:
                if r.status_code != 200:
                    yield ProgressEvent(
                        message=f"Server returned {r.status_code}",
                        fraction=0.0, is_terminal=True, is_error=True,
                    )
                    return

                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        payload = json.loads(line[len("data: "):])
                    except json.JSONDecodeError:
                        continue

                    is_terminal = payload.get("terminal", False)
                    is_error = payload.get("level") == "ERROR"
                    fraction = payload.get("metrics", {}).get("fraction", -1.0)

                    yield ProgressEvent(
                        message=payload.get("message", ""),
                        fraction=fraction if fraction >= 0 else 0.5,
                        stage=payload.get("stage", ""),
                        metrics=payload.get("metrics") or {},
                        is_terminal=is_terminal,
                        is_error=is_error,
                    )

                    if is_terminal:
                        with self._lock:
                            if self._active_job_id == job_id:
                                self._active_job_id = None
                        return

        except requests.exceptions.ConnectionError as e:
            yield ProgressEvent(
                message=f"Connection to server lost: {e}",
                fraction=0.0, is_terminal=True, is_error=True,
            )

    def cancel(self, job_id: str) -> bool:
        try:
            self._post(f"/jobs/{job_id}/cancel")
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
            return True
        except Exception:
            return False

    def resolve_ambiguity(
        self,
        session_id: str,
        ambiguity_id: str,
        action: str,
        value: Any,
    ) -> Dict[str, Any]:
        try:
            return self._post(f"/studio/session/{session_id}/resolve", json={
                "ambiguity_id": ambiguity_id,
                "selected_action": action,
                "selected_value": value,
            })
        except Exception as e:
            raise BackendError(f"Failed to resolve ambiguity: {e}") from e

    def confirm_course(self, session_id: str) -> bool:
        try:
            self._post(f"/studio/session/{session_id}/confirm")
            return True
        except Exception as e:
            raise BackendError(f"Confirmation failed: {e}") from e

    def health_check(self) -> Dict[str, Any]:
        try:
            r = self._session.get(f"{self.server_url}/health", timeout=5)
            if r.status_code == 200:
                return {
                    "healthy": True,
                    "details": f"Server at {self.server_url} is reachable.",
                    "mode": self.mode.value,
                }
            return {
                "healthy": False,
                "details": f"Server returned {r.status_code}.",
                "mode": self.mode.value,
            }
        except Exception as e:
            return {
                "healthy": False,
                "details": f"Cannot reach server at {self.server_url}: {e}",
                "mode": self.mode.value,
            }

    @property
    def is_busy(self) -> bool:
        return self._active_job_id is not None

    # ── Private HTTP helpers ──────────────────────────────────────────

    def _get(self, path: str) -> Dict:
        r = self._session.get(f"{self.server_url}{path}", timeout=self.timeout)
        self._raise_for_status(r)
        return r.json()

    def _post(self, path: str, json: Dict = None) -> Dict:
        r = self._session.post(
            f"{self.server_url}{path}", json=json or {}, timeout=self.timeout
        )
        self._raise_for_status(r)
        return r.json()

    def _upload_file(self, session_id: str, file_path: str):
        path = Path(file_path)
        if not path.exists():
            raise BackendError(f"File not found: {file_path}")
        with open(path, "rb") as f:
            r = self._session.post(
                f"{self.server_url}/studio/session/{session_id}/upload",
                files={"file": (path.name, f, "application/octet-stream")},
                timeout=120,
            )
        self._raise_for_status(r)

    def _raise_for_status(self, response: requests.Response):
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise BackendError(
                f"Server error {response.status_code}: {detail}"
            )
