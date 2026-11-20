"""
tests/test_docx_export_gate_enforcement.py
==========================================
Comprehensive regression suite for Stage 2C Phase 3 Physical DOCX & PDF Export Gate Enforcement.
Covers 22 rigorous scenarios verifying dual-tier validation, registry authorization, tamper detection,
HTTP status mapping, and backward-compatible object contract preservation.
"""

from copy import deepcopy
from datetime import datetime, timezone, timedelta
import io
import json
import os
from pathlib import Path
import pytest

from aion_api import app
from core.validation.export_gate import ExportGate, ExportDecision
from core.validation.paper_export_registry import (
    PaperExportRegistry,
    PaperExportRecord,
    generate_paper_id,
    canonicalize_paper_payload,
    RegistryCollisionError,
    STATUS_VALID,
    STATUS_MISSING,
    STATUS_EXPIRED,
    STATUS_CORRUPTED,
    STATUS_ERROR,
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


def make_valid_paper(paper_id: str = None) -> dict:
    """Helper creating a fully conforming VTU examination paper payload."""
    if paper_id is None:
        paper_id = generate_paper_id()
    return {
        "id": paper_id,
        "paper_id": paper_id,
        "title": "Computer Networks Examination",
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
                "title": "Module 1: Application Layer",
                "questions": [
                    {
                        "mq_index": 1,
                        "question_number": 1,
                        "or_group": "group_1",
                        "subquestions": [
                            {
                                "slot_id": "slot_m1_q1_a",
                                "letter": "a",
                                "text": "Explain the architectural principles of the DNS distributed database system.",
                                "marks": 6,
                                "co": "CO1",
                                "bloom": "L2",
                            },
                            {
                                "slot_id": "slot_m1_q1_b",
                                "letter": "b",
                                "text": "Differentiate between persistent and non-persistent HTTP connections with timing diagrams.",
                                "marks": 4,
                                "co": "CO1",
                                "bloom": "L3",
                            },
                        ],
                    },
                    {
                        "mq_index": 2,
                        "question_number": 2,
                        "or_group": "group_1",
                        "subquestions": [
                            {
                                "slot_id": "slot_m1_q2_a",
                                "letter": "a",
                                "text": "Describe the socket programming workflow using TCP sockets in network communications.",
                                "marks": 6,
                                "co": "CO1",
                                "bloom": "L2",
                            },
                            {
                                "slot_id": "slot_m1_q2_b",
                                "letter": "b",
                                "text": "Analyze the message formats used in the SMTP electronic mail exchange protocol.",
                                "marks": 4,
                                "co": "CO1",
                                "bloom": "L3",
                            },
                        ],
                    },
                ],
            },
        ],
    }


# =============================================================================
# T1: Eligible registered unchanged paper -> 200 OK (.docx binary starting with PK)
# =============================================================================
def test_t01_eligible_registered_unchanged_paper_returns_200(client):
    paper = make_valid_paper()
    record = PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
    )
    assert record.export_eligible is True

    resp = client.post("/api/export/docx", json=paper)
    assert resp.status_code == 200
    assert resp.mimetype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert resp.data.startswith(b"PK")  # Valid docx zip magic header


# =============================================================================
# T2: Unregistered paper -> 404 Not Found
# =============================================================================
def test_t02_unregistered_paper_returns_404(client):
    paper = make_valid_paper("paper_unregistered_99999999")
    resp = client.post("/api/export/docx", json=paper)
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["code"] == "REGISTRY_RECORD_NOT_FOUND"


# =============================================================================
# T3: Ineligible registered paper -> 403 Forbidden
# =============================================================================
def test_t03_ineligible_registered_paper_returns_403(client):
    paper = make_valid_paper()
    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=False,
        paper_status="BLOCKED",
        blocked_reasons=["QUALITY_GATE_FAILED: Faithfulness below threshold."],
    )

    resp = client.post("/api/export/docx", json=paper)
    assert resp.status_code == 403
    data = resp.get_json()
    assert data["code"] == "EXPORT_INELIGIBLE"
    assert "QUALITY_GATE_FAILED" in data["blocked_reasons"][0]


# =============================================================================
# T4: Expired paper -> 403 Forbidden
# =============================================================================
def test_t04_expired_paper_returns_403(client, isolated_registry):
    paper = make_valid_paper()
    past_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    record = PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
        expires_at=past_time,
    )

    resp = client.post("/api/export/docx", json=paper)
    assert resp.status_code == 403
    data = resp.get_json()
    assert data["code"] == "REGISTRY_RECORD_EXPIRED"


# =============================================================================
# T5: Corrupted registry file -> 500 Internal Error
# =============================================================================
def test_t05_corrupted_registry_file_returns_500(client, isolated_registry):
    paper = make_valid_paper()
    record_file = isolated_registry / f"{paper['id']}.json"
    record_file.write_text("{invalid_json: true, unterminated", encoding="utf-8")

    resp = client.post("/api/export/docx", json=paper)
    assert resp.status_code == 500
    data = resp.get_json()
    assert data["code"] == "REGISTRY_RECORD_CORRUPTED"


# =============================================================================
# T6: Modified question text -> 409 Conflict
# =============================================================================
def test_t06_tampered_question_text_returns_409(client):
    paper = make_valid_paper()
    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
    )

    tampered = deepcopy(paper)
    tampered["modules"][0]["questions"][0]["subquestions"][0]["text"] = "HACKED: Altered question text."

    resp = client.post("/api/export/docx", json=tampered)
    assert resp.status_code == 409
    data = resp.get_json()
    assert data["code"] == "PAYLOAD_DIGEST_MISMATCH"


# =============================================================================
# T7: Swapped question order -> 409 Conflict
# =============================================================================
def test_t07_swapped_question_order_returns_409(client):
    paper = make_valid_paper()
    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
    )

    tampered = deepcopy(paper)
    # Swap q1 and q2 in module 0
    questions = tampered["modules"][0]["questions"]
    questions[0], questions[1] = questions[1], questions[0]

    resp = client.post("/api/export/docx", json=tampered)
    assert resp.status_code == 409
    data = resp.get_json()
    assert data["code"] == "PAYLOAD_DIGEST_MISMATCH"


# =============================================================================
# T8: Swapped module order -> 409 Conflict
# =============================================================================
def test_t08_swapped_module_order_returns_409(client):
    paper = make_valid_paper()
    mod2 = deepcopy(paper["modules"][0])
    mod2["module_index"] = 2
    mod2["title"] = "Module 2: Transport Layer"
    mod2["questions"][0]["subquestions"][0]["slot_id"] = "slot_m2_q3_a"
    mod2["questions"][0]["subquestions"][1]["slot_id"] = "slot_m2_q3_b"
    mod2["questions"][1]["subquestions"][0]["slot_id"] = "slot_m2_q4_a"
    mod2["questions"][1]["subquestions"][1]["slot_id"] = "slot_m2_q4_b"
    paper["modules"].append(mod2)

    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
    )

    tampered = deepcopy(paper)
    tampered["modules"][0], tampered["modules"][1] = tampered["modules"][1], tampered["modules"][0]

    resp = client.post("/api/export/docx", json=tampered)
    assert resp.status_code == 409
    data = resp.get_json()
    assert data["code"] == "PAYLOAD_DIGEST_MISMATCH"


# =============================================================================
# T9: Modified marks split -> 409 Conflict
# =============================================================================
def test_t09_tampered_marks_split_returns_409(client):
    paper = make_valid_paper()
    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
    )

    tampered = deepcopy(paper)
    # Change marks from 6+4 to 7+3
    tampered["modules"][0]["questions"][0]["subquestions"][0]["marks"] = 7
    tampered["modules"][0]["questions"][0]["subquestions"][1]["marks"] = 3

    resp = client.post("/api/export/docx", json=tampered)
    assert resp.status_code == 409
    data = resp.get_json()
    assert data["code"] == "PAYLOAD_DIGEST_MISMATCH"


# =============================================================================
# T10: Modified header/subject -> 409 Conflict
# =============================================================================
def test_t10_tampered_header_metadata_returns_409(client):
    paper = make_valid_paper()
    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
    )

    tampered = deepcopy(paper)
    tampered["subject"] = "Modified Fake Subject Name"

    resp = client.post("/api/export/docx", json=tampered)
    assert resp.status_code == 409
    data = resp.get_json()
    assert data["code"] == "PAYLOAD_DIGEST_MISMATCH"


# =============================================================================
# T11: Forged client export_eligible: true -> 403 Forbidden
# =============================================================================
def test_t11_client_cannot_forge_export_eligibility(client):
    paper = make_valid_paper()
    # Server authority registers it as ineligible
    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=False,
        paper_status="BLOCKED",
        blocked_reasons=["QUALITY_GATE_FAILED"],
    )

    # Client tries to pass export_eligible: true in wire payload
    spoofed = deepcopy(paper)
    spoofed["export_eligible"] = True
    spoofed["status"] = "OK"

    resp = client.post("/api/export/docx", json=spoofed)
    assert resp.status_code == 403
    data = resp.get_json()
    assert data["code"] == "EXPORT_INELIGIBLE"


# =============================================================================
# T12: Empty request body -> 400 Bad Request
# =============================================================================
def test_t12_empty_request_body_returns_400(client):
    resp = client.post("/api/export/docx", data="", content_type="application/json")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "EMPTY_REQUEST_BODY"


# =============================================================================
# T13: Malformed JSON syntax -> 400 Bad Request
# =============================================================================
def test_t13_malformed_json_syntax_returns_400(client):
    resp = client.post("/api/export/docx", data="{bad: json", content_type="application/json")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "INVALID_JSON"


# =============================================================================
# T14: Non-dict payload -> 422 Unprocessable
# =============================================================================
def test_t14_non_dict_payload_returns_422(client):
    resp = client.post("/api/export/docx", data="[1, 2, 3]", content_type="application/json")
    assert resp.status_code == 422
    data = resp.get_json()
    assert data["code"] == "INVALID_PAPER_SCHEMA"


# =============================================================================
# T15: Paper with [UNRESOLVED SLOT: ...] -> 422 Unprocessable
# =============================================================================
def test_t15_unresolved_placeholder_returns_422(client):
    paper = make_valid_paper()
    paper["modules"][0]["questions"][0]["subquestions"][0]["text"] = (
        "[UNRESOLVED SLOT: m1_q1_a] Detailed explanation could not be synthesized."
    )
    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
    )

    resp = client.post("/api/export/docx", json=paper)
    assert resp.status_code == 422
    data = resp.get_json()
    assert data["code"] == "PHYSICAL_VALIDATION_FAILED"
    assert any("UNRESOLVED_SLOT_DETECTED" in f for f in data["failures"])


# =============================================================================
# T16: Duplicate question slots -> 422 Unprocessable
# =============================================================================
def test_t16_duplicate_question_slots_returns_422(client):
    paper = make_valid_paper()
    # Duplicate slot_id in subquestions
    paper["modules"][0]["questions"][1]["subquestions"][0]["slot_id"] = "slot_m1_q1_a"
    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
    )

    resp = client.post("/api/export/docx", json=paper)
    assert resp.status_code == 422
    data = resp.get_json()
    assert data["code"] == "PHYSICAL_VALIDATION_FAILED"
    assert any("DUPLICATE_QUESTION_SLOT" in f for f in data["failures"])


# =============================================================================
# T17: Boolean or zero marks -> 422 Unprocessable
# =============================================================================
def test_t17_invalid_marks_returns_422(client):
    paper = make_valid_paper()
    paper["modules"][0]["questions"][0]["subquestions"][0]["marks"] = True  # bool
    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
    )

    resp = client.post("/api/export/docx", json=paper)
    assert resp.status_code == 422
    data = resp.get_json()
    assert data["code"] == "PHYSICAL_VALIDATION_FAILED"
    assert any("INVALID_MARKS" in f for f in data["failures"])


# =============================================================================
# T18: Direct exporter call on clean unregistered payload -> raises ExportIntegrityError
# =============================================================================
def test_t18_direct_exporter_call_unregistered_raises():
    paper = make_valid_paper("paper_unregistered_12345678")
    with pytest.raises(ExportIntegrityError) as exc_info:
        generate_docx_from_paper(paper)
    assert "not found in export registry" in str(exc_info.value) or "unregistered" in str(exc_info.value)


# =============================================================================
# T19: Duplicate ID registration collision -> fails closed without overwrite
# =============================================================================
def test_t19_registration_collision_fails_closed():
    paper = make_valid_paper()
    rec1 = PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=True,
        paper_status="OK",
    )
    assert rec1 is not None

    with pytest.raises(RegistryCollisionError):
        PaperExportRegistry.register(
            paper_id=paper["id"],
            paper_data=paper,
            export_eligible=False,
            paper_status="FAILED",
            allow_overwrite=False,
        )


# =============================================================================
# T20: PDF route /api/download/pdf enforces identical gate
# =============================================================================
def test_t20_pdf_route_enforces_gate(client):
    paper = make_valid_paper()
    # Unregistered -> 404
    resp = client.post("/api/download/pdf", json=paper)
    assert resp.status_code == 404
    assert resp.get_json()["code"] == "REGISTRY_RECORD_NOT_FOUND"

    # Ineligible -> 403
    PaperExportRegistry.register(
        paper_id=paper["id"],
        paper_data=paper,
        export_eligible=False,
        paper_status="BLOCKED",
    )
    resp = client.post("/api/download/pdf", json=paper)
    assert resp.status_code == 403
    assert resp.get_json()["code"] == "EXPORT_INELIGIBLE"


# =============================================================================
# T21: V1 route /api/v1/papers/<paper_id>/export enforces registry lookup
# =============================================================================
def test_t21_v1_papers_export_enforces_registry(client):
    paper_id = "paper_20261120_120000_abcd1234"
    # Unregistered -> 404
    resp = client.post(f"/api/v1/papers/{paper_id}/export", json={"format": "docx"})
    assert resp.status_code == 404

    # Ineligible -> 403
    paper = make_valid_paper(paper_id)
    PaperExportRegistry.register(
        paper_id=paper_id,
        paper_data=paper,
        export_eligible=False,
        paper_status="BLOCKED",
    )
    resp = client.post(f"/api/v1/papers/{paper_id}/export", json={"format": "docx"})
    assert resp.status_code == 403
    assert resp.get_json()["code"] == "EXPORT_INELIGIBLE"


# =============================================================================
# T22: Existing object-based ExportGate.evaluate() remains 100% backward compatible
# =============================================================================
def test_t22_existing_export_gate_evaluate_backward_compatible():
    from core.contracts.question import LegacyGeneratedQuestionAdapter

    q = LegacyGeneratedQuestionAdapter.adapt(
        slot_id="slot_test_1",
        question_text="Describe the TCP handshake procedure in detail.",
        marks=10,
        bloom="L2",
        co="CO1",
        module_id=1,
    )
    decision = ExportGate.evaluate(generated_slots=[q])
    assert decision.passed is True
    assert decision.status == "PASS"

    # Reject unresolved object
    q_unresolved = LegacyGeneratedQuestionAdapter.adapt(
        slot_id="slot_test_2",
        question_text="[UNRESOLVED SLOT: slot_test_2] Unsynthesized question.",
        marks=10,
        bloom="L2",
        co="CO1",
        module_id=1,
    )
    decision2 = ExportGate.evaluate(generated_slots=[q_unresolved])
    assert decision2.passed is False
    assert decision2.status == "BLOCKED"
    assert "UNRESOLVED_SLOT_DETECTED" in decision2.failures[0]
