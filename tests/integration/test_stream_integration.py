# tests/integration/test_stream_integration.py

import json
import pytest
from unittest.mock import patch
from aion_api import app
from tests.integration.test_production_smoke import mock_robust_llm_call, mock_extract, SyncExecutor


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_generate_stream_endpoint(client):
    """
    Integration test verifying the /api/generate/stream endpoint structure
    and its output formatting.
    """
    payload = {
        "subject": "Data Structures",
        "department": "AIML",
        "semester": 3,
        "exam_type": "IAT-1",
        "selected_modules": [1],
        "bloom_levels": ["L2", "L3"],
        "difficulty": "MIXED",
        "model": "qwen2.5:14b",
        "notes_text": "Syllabus details: Stacks and Queues are core modules.",
    }

    # Call generate/stream (which yields SSE) with mocks to run hermetically
    with patch("core.extraction.gateway.ExtractionGateway.extract", mock_extract), \
         patch("v0_1.main.upload", lambda x: x), \
         patch("v0_1.llm.RobustLLMCaller.call", mock_robust_llm_call), \
         patch("v0_1.main.ThreadPoolExecutor", SyncExecutor):
        response = client.post(
            "/api/generate/stream",
            json=payload
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["Content-Type"]

    # Read events
    data = response.get_data(as_text=True)
    assert "stage_update" in data
