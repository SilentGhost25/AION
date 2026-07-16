# AION-Trainer/tests/test_training_studio.py
"""
Unit tests for the Training Studio subsystem.
"""

import pytest
import shutil
import json
from pathlib import Path
from fastapi.testclient import TestClient

from training_studio.classifier.document_classifier import DocumentClassifier, DocumentType
from training_studio.classifier.subject_detector import SubjectDetector
from training_studio.classifier.module_mapper import ModuleMapper, TOCEntry
from training_studio.analyser.analysis_result import SessionAnalysisResult, FileAnalysis, AmbiguitySeverity
from training_studio.analyser.ambiguity_detector import AmbiguityDetector
from training_studio.analyser.analysis_pipeline import AnalysisPipeline
from training_studio.preview.course_preview_builder import CoursePreviewBuilder
from training_studio.studio_session import TrainingStudioSession
from acb.syllabus_parser import ParsedSyllabus, SyllabusModule
from server.api import app


class MockLLM:
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 100) -> str:
        return "BAI401"


@pytest.fixture
def dummy_syllabus():
    return ParsedSyllabus(
        subject_code="BAI401",
        subject_name="Artificial Intelligence",
        semester=4,
        department="AIML",
        university="VTU",
        modules=[
            SyllabusModule(
                module_number=1,
                title="Search Algorithms",
                topics=["uninformed search", "intelligent agents", "dfs", "bfs"],
                hours=8,
            ),
            SyllabusModule(
                module_number=2,
                title="Heuristic Search",
                topics=["heuristic", "a* search", "greedy search"],
                hours=8,
            )
        ]
    )


def test_document_classifier():
    classifier = DocumentClassifier()

    # Textbook text signals
    textbook_text = (
        "Chapter 1: Foundations of Intelligent Agents\n"
        "In this chapter we study the definition of rational agent.\n"
        "Theorem 1.1 states that all agents are systems.\n"
        "Figure 1.2 shows the schematic diagram.\n"
    )
    res = classifier.classify_text(textbook_text, filename="artificial_intelligence_textbook.pdf")
    assert res.document_type == DocumentType.TEXTBOOK
    assert res.confidence > 0.0

    # Notes signals
    notes_text = (
        "Module 1 notes: Intelligent Agents\n"
        "Lecture notes on AI basics.\n"
        "• Agent interacts with environment via sensors.\n"
        "• Note: rational agent maximizes expected utility.\n"
    )
    res_notes = classifier.classify_text(notes_text, filename="lecture_notes.pdf")
    assert res_notes.document_type == DocumentType.NOTES


def test_subject_detector():
    llm = MockLLM()
    detector = SubjectDetector(llm_client=llm)

    # 1. Header code match
    header_text = "Subject: BAI401 Artificial Intelligence Exam Paper 2026"
    res1 = detector.detect(header_text)
    assert res1.subject_code == "BAI401"
    assert res1.confidence == 0.99

    # 2. Vocabulary match
    vocab_text = "Today we discuss predicate logic, search algorithms, intelligent agents, game theory."
    res2 = detector.detect(vocab_text)
    assert res2.subject_code == "BAI401"


def test_module_mapper(dummy_syllabus):
    mapper = ModuleMapper(dummy_syllabus)

    toc = [
        TOCEntry(title="Chapter 1: Uninformed search and agents", page=1, level=1),
        TOCEntry(title="Chapter 2: Heuristic search and A* algorithms", page=25, level=1),
    ]

    res = mapper.map(toc)
    assert len(res.mappings) == 2
    assert res.mappings[0].assigned_module == 1
    assert res.mappings[1].assigned_module == 2


def test_ambiguity_detector():
    detector = AmbiguityDetector()
    result = SessionAnalysisResult(subject_code="BAI401")

    # Mismatch scenario: Module 1 file but contains Module 2 content
    fa = FileAnalysis(
        file_id="f1",
        filename="module_1_notes.pdf",
        document_type=DocumentType.NOTES,
        module_mappings=[
            {"chapter_title": "Heuristic Search", "assigned_module": 2, "confidence": 0.9}
        ]
    )
    result.file_analyses.append(fa)

    ambiguities = detector.detect(result)
    assert len(ambiguities) == 1
    assert ambiguities[0].severity == AmbiguitySeverity.WARNING
    assert "Module mismatch" in ambiguities[0].title


def test_analysis_pipeline_and_session(tmp_path, dummy_syllabus):
    # Setup dummy textbooks and syllabus files
    tb_file = tmp_path / "ai_book.txt"
    tb_file.write_text("Chapter 1: Introduction to uninformed search. Rational agent structure.", encoding="utf-8")

    syllabus_file = tmp_path / "vtu_syllabus.txt"
    syllabus_file.write_text("syllabus for BAI401. module 1: uninformed search. module 2: heuristic search.", encoding="utf-8")

    session = TrainingStudioSession(syllabus=dummy_syllabus)
    session.add_file(str(tb_file))
    session.add_file(str(syllabus_file))

    # Test file removal
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("dummy", encoding="utf-8")
    session.add_file(str(dummy_file))
    assert len(session.file_paths) == 3

    session.start_analysis()
    # Find dummy file analysis ID
    dummy_fa = next(fa for fa in session.result.file_analyses if fa.filename == "dummy.txt")
    session.remove_file(dummy_fa.file_id)
    assert len(session.file_paths) == 2


def test_preview_builder():
    result = SessionAnalysisResult(subject_code="BAI401", subject_name="Artificial Intelligence")
    result.file_analyses.append(FileAnalysis(
        file_id="f1",
        filename="book.pdf",
        document_type=DocumentType.TEXTBOOK,
        module_mappings=[
            {"chapter_title": "Intro", "assigned_module": 1, "confidence": 0.9}
        ]
    ))
    builder = CoursePreviewBuilder()
    tree = builder.build_tree(result)
    assert tree["subject_code"] == "BAI401"


def test_studio_api_endpoints(tmp_path):
    client = TestClient(app)
    headers = {"x-aion-token": "test-token-0000"}

    # Write a test file
    test_doc = tmp_path / "vtu_ai_notes.txt"
    test_doc.write_text("module 1. uninformed search and intelligent agents. time complexity BFS.", encoding="utf-8")

    # 1. POST /studio/upload
    with open(test_doc, "rb") as f:
        res_upload = client.post(
            "/studio/upload",
            data={
                "subject_code": "BAI401",
                "academic_root": str(tmp_path),
                "department": "AIML",
                "semester": 4,
            },
            files=[("files", (test_doc.name, f, "text/plain"))],
            headers=headers,
        )
    assert res_upload.status_code == 200
    data = res_upload.json()
    session_id = data["session_id"]

    # 2. POST /studio/session/{session_id}/analyse
    res_analyse = client.post(
        f"/studio/session/{session_id}/analyse",
        headers=headers,
    )
    assert res_analyse.status_code == 200
    analyse_data = res_analyse.json()
    assert analyse_data["analysis_complete"] is True

    # 3. GET /studio/session/{session_id}/preview
    res_preview = client.get(
        f"/studio/session/{session_id}/preview",
        headers=headers,
    )
    assert res_preview.status_code == 200
    preview_data = res_preview.json()
    assert preview_data["subject_code"] == "BAI401"

    # 4. POST /studio/session/{session_id}/resolve (if there are warning/info ambiguities)
    ambiguities = analyse_data.get("ambiguities", [])
    if ambiguities:
        amb = ambiguities[0]
        res_resolve = client.post(
            f"/studio/session/{session_id}/resolve",
            json={
                "ambiguity_id": amb["ambiguity_id"],
                "option_index": 0,
            },
            headers=headers,
        )
        assert res_resolve.status_code == 200

    # 5. POST /studio/session/{session_id}/train (should succeed or raise 400 depending on unresolved errors)
    res_train = client.post(
        f"/studio/session/{session_id}/train",
        json={
            "academic_root": str(tmp_path),
            "epochs": 1,
            "learning_rate": 1e-5,
        },
        headers=headers,
    )
    # Since there are no unresolved errors (only warnings at most for notes), it should pass or fail gracefully
    assert res_train.status_code in (200, 400)
