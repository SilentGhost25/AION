# research/AION-Trainer/tests/test_learning_engine.py
import pytest
import shutil
import json
from pathlib import Path
from fastapi.testclient import TestClient

from acb.concept import Concept, ModuleLink, ConceptStore
from learning_engine.stages import ConceptStage
from learning_engine.memory.concept_memory import ConceptMemory, ConceptConfidence
from learning_engine.memory.relationship_memory import RelationshipMemory
from learning_engine.memory.examiner_memory import ExaminerMemory
from learning_engine.memory.mistake_memory import MistakeMemory
from learning_engine.memory.confidence_memory import ConfidenceMemory
from learning_engine.memory.question_memory import QuestionMemory
from learning_engine.memory.answer_memory import AnswerMemory
from learning_engine.orchestrator import LearningOrchestrator
from server.api import app


@pytest.fixture
def mock_academic_setup():
    # Set up folders mimicking the academic root
    acad_root = Path("scratch/test_academic")
    if acad_root.exists():
        shutil.rmtree(acad_root)
    acad_root.mkdir(parents=True, exist_ok=True)

    subject_dir = acad_root / "AIML" / "semester_4" / "BAI404"
    subject_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy concepts.json inside the db dir
    db_dir = subject_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    c1 = Concept(
        concept_id="c_peas",
        name="PEAS",
        definition="PEAS stands for Performance, Environment, Actuators, Sensors.",
        explanation="It defines the specifications of an intelligent agent.",
        key_points=["Performance measure", "Environment details", "Actuators description", "Sensors list"],
        algorithms=[],
        applications=["Robot design", "Autonomous vehicles"],
        formulas=[],
        requires_diagram=False,
        importance=0.95,
        previous_paper_frequency=4,
        syllabus_mentions=1,
    )
    c1.module_links.append(ModuleLink(subject_code="BAI404", module=1, is_primary=True))

    c2 = Concept(
        concept_id="c_agent",
        name="Intelligent Agent",
        definition="An agent is anything that can view its environment through sensors and act through actuators.",
        explanation="An agent operates in an environment to achieve goals.",
        key_points=["Sensors and actuators", "Rationality", "Autonomy"],
        algorithms=[],
        applications=["AI search", "Game playing"],
        formulas=[],
        requires_diagram=True,
        diagram_description="Agent-Environment interaction diagram",
        importance=0.9,
        previous_paper_frequency=3,
        syllabus_mentions=1,
    )
    c2.module_links.append(ModuleLink(subject_code="BAI404", module=1, is_primary=True))
    c2.prerequisites = ["PEAS"]

    # Save to both concept store locations that ACBPipeline inspects
    store_data = {"c_peas": c1.to_dict(), "c_agent": c2.to_dict()}
    
    (subject_dir / "concepts.json").write_text(json.dumps(store_data, indent=2))
    (db_dir / "concepts.json").write_text(json.dumps(store_data, indent=2))
    
    # Empty list for sources
    (subject_dir / "sources.json").write_text(json.dumps({}, indent=2))
    (db_dir / "sources.json").write_text(json.dumps({}, indent=2))

    yield str(acad_root)

    # Cleanup
    if acad_root.exists():
        shutil.rmtree(acad_root)


def test_memory_subsystems():
    temp_dir = Path("scratch/test_learning_memories")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Test ConceptMemory
    c_mem = ConceptMemory(str(temp_dir / "concepts.json"))
    c_mem.initialise_from_store(
        ConceptStore(str(temp_dir / "store.json")), "BAI404"
    )
    
    # Simulate a manual bootstrap injection
    from learning_engine.memory.concept_memory import ConceptMemoryEntry
    c_mem._entries["c_peas"] = ConceptMemoryEntry(
        concept_id="c_peas",
        name="PEAS",
        subject_code="BAI404",
        module=1,
    )

    c_mem.update_confidence("c_peas", "understand", 0.8, alpha=1.0)
    entry = c_mem.get("c_peas")
    assert entry is not None
    assert entry.confidence.understand == 0.8
    assert entry.current_stage == ConceptStage.DISCOVERED

    # Advance stage
    c_mem.advance_stage("c_peas")
    assert entry.current_stage == ConceptStage.RECOGNISED

    # Save and reload
    c_mem.save()
    c_mem2 = ConceptMemory(str(temp_dir / "concepts.json"))
    c_mem2.load()
    assert c_mem2.get("c_peas").confidence.understand == 0.8

    # Test MistakeMemory
    m_mem = MistakeMemory(str(temp_dir / "mistakes.json"))
    m_mem.record(
        mistake_id="m1",
        concept_id="c_peas",
        generated_text="Define PEAS.",
        reason="too short",
        categories=["capitalization"],
        epoch=1,
    )
    assert len(m_mem.uncorrected_for_concept("c_peas")) == 1
    m_mem.mark_corrected("m1", 2)
    assert len(m_mem.uncorrected_for_concept("c_peas")) == 0

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def test_learning_orchestrator(mock_academic_setup):
    orch = LearningOrchestrator(
        subject_code="BAI404",
        academic_root=mock_academic_setup,
        db_dir=str(Path(mock_academic_setup) / "AIML" / "semester_4" / "BAI404" / "db"),
    )

    # Run Epoch 1
    report1 = orch.run_epoch(1)
    assert report1.epoch == 1
    assert report1.concept_understanding > 0
    assert report1.weak_concepts_count >= 0

    # Advance a concept stage manually to QUESTIONABLE to trigger question generation
    entry = orch.concept_memory.get("c_peas")
    assert entry is not None
    entry.stage = ConceptStage.QUESTIONABLE
    orch.concept_memory.save()

    # Run Epoch 2
    report2 = orch.run_epoch(2)
    assert report2.epoch == 2

    # Check Academic IQ
    iq_details = orch.calculate_iq()
    assert iq_details.iq_score >= 0


def test_learning_endpoints(mock_academic_setup):
    client = TestClient(app)
    headers = {"X-AION-Token": "test-token-0000"}

    # Run Epoch via API
    resp = client.post(
        "/learning/epoch",
        headers=headers,
        json={
            "subject_code": "BAI404",
            "epoch": 1,
            "academic_root": mock_academic_setup,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # Get Status via API
    resp = client.get(
        "/learning/status",
        headers=headers,
        params={
            "subject_code": "BAI404",
            "academic_root": mock_academic_setup,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["subject_code"] == "BAI404"
    assert "academic_iq" in data

    # Clear history via API
    resp = client.post(
        "/learning/clear",
        headers=headers,
        params={
            "subject_code": "BAI404",
            "academic_root": mock_academic_setup,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
