# tests/unit/test_web_backends.py
from __future__ import annotations

import time
import json
import pytest
from flask import Flask

from aion_web.training.backends.base import (
    TrainingMode, BackendBusyError, BackendError, JobHandle, ProgressEvent,
)
from aion_web.training.backend_registry import BackendRegistry
from aion_web.training.mode_config import ModeConfig
from aion_web.training.backend_api import training_bp


def test_registry_initialization_and_mode_colors():
    registry = BackendRegistry()
    backend = registry.get()
    
    assert backend.mode == TrainingMode.DEMO
    assert backend.color == "orange"
    assert backend.display_name == "Demo Mode"
    assert registry.current_mode() == TrainingMode.DEMO


def test_registry_switching():
    registry = BackendRegistry()
    config = ModeConfig(
        server_url="http://localhost:8000",
        server_token="test-token-123",
        local_model="llama3.2:3b",
    )
    
    # Switch to REMOTE
    remote_backend = registry.switch(TrainingMode.REMOTE, config)
    assert remote_backend.mode == TrainingMode.REMOTE
    assert remote_backend.color == "blue"
    assert remote_backend.display_name == "Remote Server"
    
    # Switch to LOCAL
    local_backend = registry.switch(TrainingMode.LOCAL, config)
    assert local_backend.mode == TrainingMode.LOCAL
    assert local_backend.color == "green"
    assert local_backend.display_name == "Local Training"


def test_registry_switch_busy_guard():
    registry = BackendRegistry()
    config = ModeConfig()
    
    backend = registry.get()
    
    # Start a training job on DemoBackend
    handle = backend.train("sess-1", "BAI401")
    assert backend.is_busy is True
    
    # Attempting to switch mode while busy must raise BackendBusyError
    with pytest.raises(BackendBusyError) as exc_info:
        registry.switch(TrainingMode.LOCAL, config)
        
    assert "currently running a job" in str(exc_info.value)
    
    # Cancel job
    backend.cancel(handle.job_id)
    assert backend.is_busy is False
    
    # Now switching should succeed
    registry.switch(TrainingMode.LOCAL, config)
    assert registry.current_mode() == TrainingMode.LOCAL


def test_demo_backend_simulation():
    registry = BackendRegistry()
    backend = registry.get()
    
    # Run analysis
    analysis = backend.analyse(["/path/to/book.pdf"], "BAI401")
    assert analysis.session_id.startswith("DEMO-")
    assert analysis.books == 1
    assert len(analysis.module_summaries) > 0
    assert analysis.train_enabled is True
    
    # Start training
    handle = backend.train(analysis.session_id, "BAI401")
    assert handle.job_id.startswith("DEMO-JOB-")
    assert backend.is_busy is True
    
    # Collect progress (use stage_delay=0.0 in DemoBackend to make it instant for tests)
    backend.stage_delay = 0.0
    events = list(backend.get_progress(handle.job_id))
    
    assert len(events) > 0
    assert events[-1].is_terminal is True
    assert events[-1].fraction == 1.0
    assert backend.is_busy is False


@pytest.fixture
def flask_app():
    app = Flask("test_app")
    app.config["BACKEND_REGISTRY"] = BackendRegistry()
    app.config["MODE_CONFIG"] = ModeConfig()
    app.register_blueprint(training_bp)
    return app


def test_api_mode_endpoints(flask_app):
    client = flask_app.test_client()
    
    # Test GET /api/training/mode
    res = client.get("/api/training/mode")
    assert res.status_code == 200
    data = res.json
    assert data["active_mode"] == "demo"
    assert data["config"]["local_model"] == "llama3.2:3b"
    
    # Test POST /api/training/mode (switch to LOCAL)
    res = client.post("/api/training/mode", json={
        "mode": "local",
        "local_model": "mistral:7b"
    })
    assert res.status_code == 200
    assert res.json["switched_to"] == "local"
    
    # Verify switch took place
    res = client.get("/api/training/mode")
    assert res.json["active_mode"] == "local"
    assert flask_app.config["MODE_CONFIG"].local_model == "mistral:7b"


def test_api_analyse_and_train_demo(flask_app):
    client = flask_app.test_client()
    
    # Ensure active mode is demo
    client.post("/api/training/mode", json={"mode": "demo"})
    
    # Run analysis
    res = client.post("/api/training/analyse", json={
        "file_paths": ["book.pdf", "notes.docx"],
        "subject_code": "BAI401"
    })
    assert res.status_code == 200
    session_id = res.json["session_id"]
    assert session_id.startswith("DEMO-")
    
    # Run training
    res = client.post("/api/training/train", json={
        "session_id": session_id,
        "subject_code": "BAI401"
    })
    assert res.status_code == 200
    job_id = res.json["job_id"]
    assert job_id.startswith("DEMO-JOB-")
    
    # Stream progress
    registry = flask_app.config["BACKEND_REGISTRY"]
    registry.get().stage_delay = 0.0
    
    res = client.get(f"/api/training/progress/{job_id}")
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["Content-Type"]
    
    # Read first line of output stream
    stream_data = res.data.decode("utf-8")
    lines = stream_data.split("\n")
    assert len(lines) > 0
    # Find a data line
    data_lines = [l for l in lines if l.startswith("data:")]
    assert len(data_lines) > 0
    payload = json.loads(data_lines[0][5:])
    assert "message" in payload
