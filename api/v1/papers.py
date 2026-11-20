"""
AION API v1 — Paper Forge Router
Handles exam paper generation, validation, and export (PDF, DOCX, HTML, JSON).
"""

import threading
from pathlib import Path
from flask import Blueprint, jsonify, request
from api.v1.jobs import create_job, update_job, get_job_state

papers_bp = Blueprint("papers_api", __name__)

ROOT = Path(__file__).parent.parent.parent.resolve()
PAPERS_DIR = ROOT / "generated_papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

_paper_store = {}


def _async_generate_paper(job_id: str, config: dict):
    update_job(job_id, status="processing", progress=15, message="Selecting questions across modules...")

    subject_code = config.get("subject_code", "BEC601")
    exam_type = config.get("exam_type", "SEE")
    total_marks = config.get("total_marks", 100)

    update_job(job_id, status="processing", progress=50, message="Formatting VTU compliant paper layout...")

    paper_id = f"PAPER_{subject_code}_{exam_type}_001"
    paper_data = {
        "paper_id": paper_id,
        "subject_code": subject_code,
        "subject_name": "Satellite Communication",
        "exam_type": exam_type,
        "total_marks": total_marks,
        "duration_hours": 3,
        "sections": [
            {
                "module": 1,
                "questions": [
                    {"q_num": "1.a", "marks": 10, "text": "Explain satellite orbits and Kepler's laws."},
                    {"q_num": "1.b", "marks": 10, "text": "OR: Derive link budget equation for satellite communication."},
                ],
            },
            {
                "module": 3,
                "questions": [
                    {"q_num": "5.a", "marks": 10, "text": "Illustrate TDMA frame structure and guard time allocation."},
                    {"q_num": "5.b", "marks": 10, "text": "OR: Compare TDMA and FDMA with respect to bandwidth efficiency."},
                ],
            },
        ],
        "formats_available": ["pdf", "docx", "html", "json"],
        "status": "ready",
    }

    _paper_store[paper_id] = paper_data

    update_job(
        job_id,
        status="completed",
        progress=100,
        message="Exam paper generated successfully",
        result=paper_data,
    )


@papers_bp.route("/papers/generate", methods=["POST"])
def generate_paper():
    config = request.get_json() or {}
    job_id = create_job("paper_generation", config)

    t = threading.Thread(target=_async_generate_paper, args=(job_id, config), daemon=True)
    t.start()

    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "workspace": "paper-forge",
    })


@papers_bp.route("/papers/<paper_id>", methods=["GET"])
def get_paper(paper_id):
    if paper_id not in _paper_store:
        return jsonify({"error": f"Paper {paper_id} not found"}), 404
    return jsonify(_paper_store[paper_id])


@papers_bp.route("/papers/<paper_id>/export", methods=["POST"])
def export_paper(paper_id):
    from core.validation.paper_export_registry import (
        validate_paper_id,
        PaperExportRegistry,
        STATUS_MISSING,
        STATUS_EXPIRED,
        STATUS_CORRUPTED,
        STATUS_ERROR,
    )

    if not validate_paper_id(paper_id):
        return jsonify({"error": f"Malformed paper_id '{paper_id}'", "code": "MALFORMED_PAPER_ID"}), 422

    record, status = PaperExportRegistry.get(paper_id)
    if status == STATUS_MISSING:
        return jsonify({"error": f"Paper '{paper_id}' not found in export registry", "code": "REGISTRY_RECORD_NOT_FOUND"}), 404
    elif status == STATUS_EXPIRED:
        return jsonify({"error": f"Export authorization for '{paper_id}' has expired", "code": "REGISTRY_RECORD_EXPIRED"}), 403
    elif status in (STATUS_CORRUPTED, STATUS_ERROR):
        return jsonify({"error": f"Error accessing registry for '{paper_id}'", "code": "REGISTRY_ACCESS_ERROR"}), 500
    elif record is None or not record.export_eligible:
        return jsonify({
            "error": f"Paper '{paper_id}' is ineligible for export",
            "code": "EXPORT_INELIGIBLE",
            "blocked_reasons": record.blocked_reasons if record else []
        }), 403

    data = request.get_json(silent=True) or {}
    fmt = data.get("format", "docx")
    return jsonify({
        "paper_id": paper_id,
        "format": fmt,
        "download_url": f"/api/v1/papers/{paper_id}/download?format={fmt}",
        "status": "ready",
    })


@papers_bp.route("/papers/<paper_id>/download", methods=["GET"])
def download_paper(paper_id):
    """
    Delivers physical document bytes for an authorized paper.
    Enforces identical authorization gates: missing (404), expired (403), ineligible (403),
    digest mismatch (409). Never falls back to unvalidated stores.
    """
    from flask import send_file
    from core.validation.paper_export_registry import (
        validate_paper_id,
        PaperExportRegistry,
        STATUS_MISSING,
        STATUS_EXPIRED,
        STATUS_CORRUPTED,
        STATUS_ERROR,
        STATUS_DIGEST_MISMATCH,
    )

    if not validate_paper_id(paper_id):
        return jsonify({"error": f"Malformed paper_id '{paper_id}'", "code": "MALFORMED_PAPER_ID"}), 422

    payload, record, status = PaperExportRegistry.get_verified_payload(paper_id)
    if status == STATUS_MISSING:
        return jsonify({"error": f"Paper '{paper_id}' not found in export registry", "code": "REGISTRY_RECORD_NOT_FOUND"}), 404
    elif status == STATUS_EXPIRED:
        return jsonify({"error": f"Export authorization for '{paper_id}' has expired", "code": "REGISTRY_RECORD_EXPIRED"}), 403
    elif status == STATUS_DIGEST_MISMATCH:
        return jsonify({
            "error": f"Paper content digest mismatch for '{paper_id}'. Registry payload has been altered.",
            "code": "DIGEST_MISMATCH"
        }), 409
    elif status in (STATUS_CORRUPTED, STATUS_ERROR):
        return jsonify({"error": f"Error accessing registry for '{paper_id}'", "code": "REGISTRY_ACCESS_ERROR"}), 500
    elif record is None or not record.export_eligible:
        return jsonify({
            "error": f"Paper '{paper_id}' is ineligible for export",
            "code": "EXPORT_INELIGIBLE",
            "blocked_reasons": record.blocked_reasons if record else []
        }), 403

    if not payload:
        return jsonify({"error": f"Paper payload not found for '{paper_id}'", "code": "PAYLOAD_NOT_FOUND"}), 404

    fmt = request.args.get("format", "docx").lower()
    subject_code = payload.get("subject_code") or "VTU"
    exam_type = payload.get("exam_type") or "IA"

    if fmt == "docx":
        from v0_1.docx_export import render_docx_document
        docx_buffer = render_docx_document(payload)
        filename = f"{subject_code}_{exam_type}_{paper_id}.docx"
        return send_file(
            docx_buffer,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            as_attachment=True,
            download_name=filename,
        )
    elif fmt == "pdf":
        from v0_1.paper_formatter import export_pdf
        filename = f"{subject_code}_{exam_type}_{paper_id}.pdf"
        out_path = export_pdf(payload, filename=filename)
        return send_file(str(out_path), mimetype="application/pdf", as_attachment=True)
    elif fmt == "json":
        return jsonify(payload), 200
    else:
        return jsonify({"error": f"Unsupported export format '{fmt}'", "code": "UNSUPPORTED_FORMAT"}), 400
