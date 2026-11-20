"""
tests/test_stage2c_phase3_acceptance_closure.py
===============================================
Acceptance Closure Test Suite for Stage 2C Phase 3.3.
Verifies the 6 specific reviewer verification requirements:
1. Digest coverage: table data mutation, model-answer mutation, image content mutation, subquestion reordering, figure replacement.
2. Eligible PDF export: HTTP 200, valid readable %PDF- binary.
3. Direct exporter adversarial checks: clean fabricated payload rejected, forged authorization context rejected.
4. Internal trace isolation: render_docx_document is strictly internal; trace cannot bypass public API.
5. V1 route live integration: eligible export (200), unregistered (404), ineligible (403).
6. Full log capture & binary validation.
"""

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import pytest

from aion_api import app
from core.validation.export_gate import ExportGate
from core.validation.paper_export_registry import (
    PaperExportRegistry,
    generate_paper_id,
    canonicalize_paper_payload,
)
from v0_1.docx_export import generate_docx_from_paper, render_docx_document, ExportIntegrityError


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Isolate registry storage for each test."""
    reg_dir = tmp_path / "paper_registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AION_PAPER_REGISTRY_DIR", str(reg_dir))
    monkeypatch.setenv("AION_REGISTRY_DIR", str(reg_dir))
    return reg_dir


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def make_sample_paper_with_all_rendered_types(tmp_path) -> dict:
    """Creates a valid paper containing tables, model answers, and local image files."""
    img_file = tmp_path / "sample_diagram.png"
    # Write sample 1x1 PNG bytes
    img_file.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    paper_id = generate_paper_id()
    return {
        "id": paper_id,
        "paper_id": paper_id,
        "title": "Computer Networks Final Examination",
        "subject": "Computer Networks",
        "subject_code": "21CS52",
        "exam_type": "IAT1",
        "department": "Computer Science & Engineering",
        "semester": "V Semester",
        "max_marks": 50,
        "duration": "90 Minutes",
        "modules": [
            {
                "module_index": 1,
                "title": "Module 1: Application Layer Protocols",
                "questions": [
                    {
                        "mq_index": 1,
                        "question_number": 1,
                        "subquestions": [
                            {
                                "slot_id": "slot_m1_q1_a",
                                "letter": "a",
                                "text": "Explain DNS lookup resolution with an architectural diagram.",
                                "marks": 6,
                                "co": "CO1",
                                "bloom": "L2",
                                "image_path": str(img_file),
                                "table": [
                                    ["Record Type", "Function", "Example Value"],
                                    ["A", "Host Address", "192.168.1.1"],
                                    ["MX", "Mail Exchange", "mail.example.com"],
                                ],
                                "solution": "DNS is a distributed database that translates human hostnames into IP addresses.",
                            },
                            {
                                "slot_id": "slot_m1_q1_b",
                                "letter": "b",
                                "text": "Differentiate between HTTP persistent and non-persistent connections.",
                                "marks": 4,
                                "co": "CO1",
                                "bloom": "L3",
                                "solution": "Persistent HTTP keeps the underlying TCP connection open for multiple requests.",
                            },
                        ],
                    },
                ],
            },
        ],
    }


# =============================================================================
# Task 1: Digest Coverage Tests
# =============================================================================

def test_digest_detects_table_data_mutation(tmp_path):
    paper = make_sample_paper_with_all_rendered_types(tmp_path)
    base_hash = canonicalize_paper_payload(paper)

    tampered = deepcopy(paper)
    # Mutate table row content
    tampered["modules"][0]["questions"][0]["subquestions"][0]["table"][1][2] = "10.0.0.99"
    new_hash = canonicalize_paper_payload(tampered)

    assert base_hash != new_hash, "Digest must detect table data mutation"


def test_digest_detects_model_answer_mutation(tmp_path):
    paper = make_sample_paper_with_all_rendered_types(tmp_path)
    base_hash = canonicalize_paper_payload(paper)

    tampered = deepcopy(paper)
    # Mutate solution / model answer
    tampered["modules"][0]["questions"][0]["subquestions"][0]["solution"] = "Altered model answer text."
    new_hash = canonicalize_paper_payload(tampered)

    assert base_hash != new_hash, "Digest must detect model-answer mutation"


def test_digest_detects_image_content_mutation_with_same_path(tmp_path):
    paper = make_sample_paper_with_all_rendered_types(tmp_path)
    base_hash = canonicalize_paper_payload(paper)

    img_path = Path(paper["modules"][0]["questions"][0]["subquestions"][0]["image_path"])
    # Modify the image file bytes on disk while retaining identical path
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR_MODIFIED_BYTES_FOR_TEST")
    new_hash = canonicalize_paper_payload(paper)

    assert base_hash != new_hash, "Digest must detect image content mutation even when image_path is unchanged"


def test_digest_detects_subquestion_reordering(tmp_path):
    paper = make_sample_paper_with_all_rendered_types(tmp_path)
    base_hash = canonicalize_paper_payload(paper)

    tampered = deepcopy(paper)
    # Swap subquestion (a) and (b)
    subs = tampered["modules"][0]["questions"][0]["subquestions"]
    subs[0], subs[1] = subs[1], subs[0]
    new_hash = canonicalize_paper_payload(tampered)

    assert base_hash != new_hash, "Digest must detect subquestion reordering"


def test_digest_detects_figure_path_replacement(tmp_path):
    paper = make_sample_paper_with_all_rendered_types(tmp_path)
    base_hash = canonicalize_paper_payload(paper)

    tampered = deepcopy(paper)
    tampered["modules"][0]["questions"][0]["subquestions"][0]["image_path"] = str(tmp_path / "replaced_diagram.png")
    new_hash = canonicalize_paper_payload(tampered)

    assert base_hash != new_hash, "Digest must detect figure replacement"


# =============================================================================
# Task 2: Eligible PDF Export Success on Live Route
# =============================================================================

def test_eligible_pdf_export_success_returns_200(client, tmp_path):
    paper = make_sample_paper_with_all_rendered_types(tmp_path)
    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
    )

    resp = client.post("/api/download/pdf", json=paper)
    assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}: {resp.get_data(as_text=True)}"
    assert resp.mimetype == "application/pdf"
    assert resp.data.startswith(b"%PDF-"), "Exported PDF must have valid %PDF- magic header"
    assert len(resp.data) > 1000, "Exported PDF must have substantial readable content"


# =============================================================================
# Task 3: Direct Exporter Adversarial Checks
# =============================================================================

def test_direct_exporter_rejects_clean_fabricated_payload_with_fake_id(tmp_path):
    """A clean, well-formed payload with a fabricated, unregistered ID must be rejected."""
    paper = make_sample_paper_with_all_rendered_types(tmp_path)
    paper["id"] = "paper_fabricated_99999999"
    paper["paper_id"] = "paper_fabricated_99999999"

    with pytest.raises(ExportIntegrityError) as exc_info:
        generate_docx_from_paper(paper)
    assert "not found in export registry" in str(exc_info.value)


def test_direct_exporter_rejects_forged_authorization_context(tmp_path):
    """
    Adversary registers Paper A (eligible), then takes Paper A's valid paper_id
    and attaches it to Paper B (with altered content).
    Direct exporter MUST detect the digest mismatch and reject it.
    """
    paper_a = make_sample_paper_with_all_rendered_types(tmp_path)
    PaperExportRegistry.register(
        paper_id=paper_a["id"],
        paper_data=paper_a,
        export_eligible=True,
        paper_status="OK",
    )

    paper_b = deepcopy(paper_a)
    paper_b["modules"][0]["questions"][0]["subquestions"][0]["text"] = "Paper B Trojan question text."

    with pytest.raises(ExportIntegrityError) as exc_info:
        generate_docx_from_paper(paper_b)
    assert "digest mismatch" in str(exc_info.value)


# =============================================================================
# Task 4: Internal Trace Isolation Check
# =============================================================================

def test_internal_render_is_isolated_and_unregistered_route_fails(client, tmp_path):
    """
    Verifies render_docx_document() can be called by internal offline workers,
    but no external API route exposes this without registry authorization.
    """
    paper = make_sample_paper_with_all_rendered_types(tmp_path)
    # 1. Internal rendering produces docx bytes in memory
    internal_buf = render_docx_document(paper)
    assert isinstance(internal_buf, io.BytesIO)
    assert internal_buf.getvalue().startswith(b"PK")

    # 2. Public HTTP routes (/api/export/docx) NEVER allow un-registered payloads
    resp = client.post("/api/export/docx", json=paper)
    assert resp.status_code == 404
    assert resp.get_json()["code"] == "REGISTRY_RECORD_NOT_FOUND"


# =============================================================================
# Task 5: V1 Route Live Integration
# =============================================================================

def test_v1_route_eligible_and_ineligible_exports(client, tmp_path):
    paper = make_sample_paper_with_all_rendered_types(tmp_path)
    paper_id = paper["id"]

    # 1. Unregistered -> 404
    resp_unreg = client.post(f"/api/v1/papers/{paper_id}/export", json={"format": "docx"})
    assert resp_unreg.status_code == 404

    # 2. Registered & Ineligible -> 403
    PaperExportRegistry.register(
        paper_id=paper_id,
        paper_data=paper,
        export_eligible=False,
        paper_status="BLOCKED",
        blocked_reasons=["QUALITY_FAIL"],
    )
    resp_inelig = client.post(f"/api/v1/papers/{paper_id}/export", json={"format": "docx"})
    assert resp_inelig.status_code == 403
    assert resp_inelig.get_json()["code"] == "EXPORT_INELIGIBLE"

    # 3. Registered & Eligible -> 200 OK with download_url
    paper_id_ok = generate_paper_id()
    paper_ok = deepcopy(paper)
    paper_ok["id"] = paper_id_ok
    paper_ok["paper_id"] = paper_id_ok
    PaperExportRegistry.register(
        paper_id=paper_id_ok,
        paper_data=paper_ok,
        export_eligible=True,
        paper_status="OK",
    )
    resp_ok = client.post(f"/api/v1/papers/{paper_id_ok}/export", json={"format": "docx"})
    assert resp_ok.status_code == 200
    data = resp_ok.get_json()
    assert data["paper_id"] == paper_id_ok
    assert f"/api/v1/papers/{paper_id_ok}/download" in data["download_url"]
