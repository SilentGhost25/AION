import pytest
from pathlib import Path

from server.pipeline_runner import PipelineRunner
from server.job_queue import Job
from conftest import write_fake_file


class DummyParser:
    def __init__(self, *args, **kwargs): pass
    def parse_all(self, file_paths):
        from preprocessing.parallel_parser import KnowledgeObject
        return [
            KnowledgeObject(object_id="KO-001", content="Constraint Satisfaction", module_hint=1, source_file="x"),
            KnowledgeObject(object_id="KO-002", content="Backtracking search", module_hint=1, source_file="y")
        ]


class DummyDatasetBuilder:
    def __init__(self, *args, **kwargs): pass
    def build(self, knowledge_objects):
        # Write dummy train.jsonl
        output_dir = Path("dataset/BAI401")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "train.jsonl", "w") as f:
            f.write('{"knowledge": "Constraint satisfaction models"}\n')


class DummyTrainer:
    def __init__(self, *args, **kwargs): pass
    def train(self):
        # Create dummy checkpoints
        Path("checkpoints").mkdir(parents=True, exist_ok=True)
        with open("checkpoints/aion_model_latest.pt", "w") as f:
            f.write("model weights")


@pytest.fixture
def clean_dirs():
    # Make sure we clean up dirs after test runs
    import shutil
    yield
    for d in [Path("dataset/BAI401"), Path("knowledge/BAI401")]:
        if d.exists():
            shutil.rmtree(d)
    for f in [Path("checkpoints/aion_model_latest.pt"), Path("models/registry.json")]:
        if f.exists():
            f.unlink()


def test_pipeline_runner_learn_cycle(job_queue, academic_root, clean_dirs):
    config = {
        "server": {
            "academic_root": str(academic_root),
            "dataset_root": str(academic_root.parent / "dataset"),
            "knowledge_root": str(academic_root.parent / "knowledge"),
            "models_root": str(academic_root.parent / "models"),
            "train_config": "configs/train.yaml"
        },
        "checkpoints": {"dir": str(academic_root.parent / "checkpoints")},
        "benchmark": {}
    }

    # Write academic files so scan detects them
    subject_dir = academic_root / "AIML" / "semester_4" / "BAI401"
    write_fake_file(subject_dir / "textbooks" / "book1.pdf")
    write_fake_file(subject_dir / "notes" / "note1.docx")

    runner = PipelineRunner(
        config=config,
        job_queue=job_queue,
        parser_factory=DummyParser,
        dataset_builder_factory=DummyDatasetBuilder,
        trainer_factory=DummyTrainer,
    )

    job = job_queue.submit("BAI401", "learn")
    runner.run(job)

    # Reload job
    completed_job = job_queue.get(job.id)
    assert completed_job.status == "completed"
    assert "candidate_version" in completed_job.result
    assert completed_job.result["candidate_version"] == "0.1"
