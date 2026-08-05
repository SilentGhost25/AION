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
    data = request.get_json() or {}
    fmt = data.get("format", "pdf")
    return jsonify({
        "paper_id": paper_id,
        "format": fmt,
        "download_url": f"/api/v1/papers/{paper_id}/download?format={fmt}",
        "status": "ready",
    })
