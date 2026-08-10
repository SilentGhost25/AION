"""
Real HTTP E2E Server Endpoint Test
===================================
Tests Flask API routes via test_client to verify HTTP responses, JSON serialization,
and server status endpoint.
"""

import json
import pytest
from aion_api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_http_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data["status"] in ("ready", "healthy", "degraded")
    assert data["api_version"] == "v1.0"
    assert "model" in data
    assert "pipeline" in data


def test_http_v1_root_endpoint(client):
    response = client.get("/api/v1")
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data["name"] == "AION API Gateway"
    assert "workspaces" in data


def test_http_emergency_generate_endpoint_not_found(client):
    payload = {"file_path": "non_existent_file.pdf", "n_questions": 3}
    response = client.post("/api/generate/emergency", json=payload)
    assert response.status_code == 404
