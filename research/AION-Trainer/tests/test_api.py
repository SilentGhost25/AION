import os
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import shutil

from server.api import app, job_queue, manifest_manager, model_registry


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("AION_SERVER_TOKEN", "test-token-0000")
    yield TestClient(app)


def test_api_ping(api_client):
    res = api_client.get("/ping")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "service": "aion_trainer"}


def test_api_rejects_unauthorized_token(api_client):
    res = api_client.get("/jobs", headers={"X-AION-Token": "wrong-token"})
    assert res.status_code == 401


def test_api_list_jobs(api_client):
    res = api_client.get("/jobs", headers={"X-AION-Token": "test-token-0000"})
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_api_submit_job(api_client):
    res = api_client.post(
        "/jobs",
        json={"subject": "BAI401", "job_type": "learn", "params": {"force": True}},
        headers={"X-AION-Token": "test-token-0000"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_api_get_job_details(api_client):
    job = job_queue.submit("BAI401", "learn")
    res = api_client.get(f"/jobs/{job.id}", headers={"X-AION-Token": "test-token-0000"})
    assert res.status_code == 200
    assert res.json()["job_id"] == job.id


def test_api_upload_file(api_client, academic_root):
    # Set app's manifest manager path to match test root
    manifest_manager.academic_root = Path(academic_root)
    
    # Create fake upload payload
    file_content = b"fake file content"
    files = {"file": ("test_book.pdf", file_content, "application/pdf")}
    data = {"category": "textbooks"}
    
    res = api_client.post(
        "/manifest/BAI401/upload",
        files=files,
        data=data,
        headers={"X-AION-Token": "test-token-0000"}
    )
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert res.json()["filename"] == "test_book.pdf"
    
    # Cleanup uploaded file
    uploaded_path = Path(res.json()["path"])
    if uploaded_path.exists():
        uploaded_path.unlink()
