"""
AION Production Pipeline & Diagnostics Unit Test Suite
======================================================
Tests ProductionPipeline master controller and diagnostics payload generation.
"""

import pytest
from pathlib import Path
from core.artifacts.store import ArtifactStore
from core.artifacts.manifest import DocumentManifest
from core.artifacts.lifecycle import ArtifactStatus
from core.production.safe_pipeline import ProductionPipeline, ProductionPipelineResult


def test_production_pipeline_run(tmp_path, monkeypatch):
    store = ArtifactStore(base_dir=str(tmp_path))
    pdf_file = tmp_path / "lecture_notes.pdf"
    pdf_file.write_bytes(b"%PDF-1.5\nLecture notes on Artificial Intelligence and Machine Learning.\n%EOF\n")

    manifest = store.store_from_temp(str(pdf_file), "lecture_notes.pdf", document_id="doc_pipe_1")
    pipeline = ProductionPipeline(store=store)

    class DummyChunk:
        status = "VALID"
        module_id = 1
        text = "Artificial Intelligence concepts."

    class DummyReport:
        total_chunks = 10
        valid_chunks = 10

    class DummyArtifact:
        chunks = [DummyChunk() for _ in range(10)]
        report = DummyReport()

    monkeypatch.setattr("core.production.safe_pipeline.ExtractionGateway.extract", lambda *a, **kw: DummyArtifact())

    request = {
        "file_id": "doc_pipe_1",
        "request_id": "req_pipe_1",
        "subject": "Artificial Intelligence",
        "exam": "IAT1",
    }

    res = pipeline.run(request)
    assert res.success is True
    assert res.paper is not None
    assert res.paper.total_marks == 50
    assert len(res.paper.or_pairs) == 5

    diag = res.diagnostics
    assert diag["document_id"] == "doc_pipe_1"
    assert diag["source"]["type"] == "PDF"
    assert diag["source"]["authoritative"] is True
    assert "extraction" in diag
    assert "modules" in diag


def test_production_pipeline_diagnostics_endpoint(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    pdf_file = tmp_path / "notes.pdf"
    pdf_file.write_bytes(b"%PDF-1.5\nNotes content.\n%EOF\n")

    manifest = store.store_from_temp(str(pdf_file), "notes.pdf", document_id="doc_diag_1")
    pipeline = ProductionPipeline(store=store)

    diag = pipeline.get_diagnostics("doc_diag_1")
    assert diag["document_id"] == "doc_diag_1"
    assert diag["source"]["type"] == "PDF"
    assert diag["source"]["authoritative"] is True
    assert diag["evidence"]["total"] == 1432
    assert diag["modules"]["1"] == 148
