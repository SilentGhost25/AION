"""
AION API v1 — Jobs & Event Streaming Router
Manages asynchronous job lifecycle and SSE event streams.
"""

import time
import json
import uuid
import threading
from typing import Dict, Any, Optional
from flask import Blueprint, jsonify, request, Response, stream_with_context

jobs_bp = Blueprint("jobs_api", __name__)

# Global thread-safe job store
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def create_job(job_type: str, metadata: Optional[dict] = None) -> str:
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "type": job_type,
            "status": "queued",
            "progress": 0,
            "message": "Job queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "metadata": metadata or {},
            "result": None,
            "error": None,
            "events": [],
        }
    return job_id


def update_job(
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    result: Optional[Any] = None,
    error: Optional[str] = None,
):
    with _jobs_lock:
        if job_id not in _jobs:
            return
        job = _jobs[job_id]
        if status:
            job["status"] = status
        if progress is not None:
            job["progress"] = progress
        if message:
            job["message"] = message
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        job["updated_at"] = time.time()

        event_payload = {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
            "timestamp": job["updated_at"],
        }
        if result:
            event_payload["result"] = result
        if error:
            event_payload["error"] = error

        job["events"].append(event_payload)


def get_job_state(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        return _jobs.get(job_id)


@jobs_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    state = get_job_state(job_id)
    if not state:
        return jsonify({"error": f"Job {job_id} not found"}), 404
    return jsonify(state)


@jobs_bp.route("/jobs/<job_id>/events", methods=["GET"])
def stream_job_events(job_id):
    state = get_job_state(job_id)
    if not state:
        return jsonify({"error": f"Job {job_id} not found"}), 404

    @stream_with_context
    def generate():
        last_index = 0
        while True:
            current_state = get_job_state(job_id)
            if not current_state:
                break

            events = current_state.get("events", [])
            while last_index < len(events):
                evt = events[last_index]
                last_index += 1
                event_type = "complete" if evt["status"] in ("completed", "failed") else "progress"
                yield f"event: {event_type}\ndata: {json.dumps(evt)}\n\n"

            if current_state["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream")
