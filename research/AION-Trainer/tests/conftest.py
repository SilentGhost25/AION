import os
import pytest
from pathlib import Path

from server.job_queue import JobQueue
from server.manifest import ManifestManager
from server.model_registry import ModelRegistry


@pytest.fixture(autouse=True)
def server_token(monkeypatch):
    monkeypatch.setenv("AION_SERVER_TOKEN", "test-token-0000")
    yield "test-token-0000"


@pytest.fixture
def job_queue(tmp_path):
    return JobQueue(db_path=str(tmp_path / "jobs.db"))


@pytest.fixture
def academic_root(tmp_path):
    root = tmp_path / "academic"
    subject_dir = root / "AIML" / "semester_4" / "BAI401"
    for sub in ["textbooks", "notes", "question_bank", "previous_papers",
                "answer_keys", "syllabus", "images"]:
        (subject_dir / sub).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def manifest_manager(academic_root):
    return ManifestManager(str(academic_root))


@pytest.fixture
def model_registry(tmp_path):
    return ModelRegistry(str(tmp_path / "models"))


def write_fake_file(path: Path, content: str = "fake content"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
