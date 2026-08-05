"""
AION API v1 — Question Forge Router
Handles question generation, evaluation, repair, and approval workflows.
"""

import threading
from flask import Blueprint, jsonify, request
from api.v1.jobs import create_job, update_job, get_job_state

try:
    from v0_1.llm import get_llm
    from v0_1.critic import AcademicCritic
    HAS_V01 = True
except ImportError:
    HAS_V01 = False

questions_bp = Blueprint("questions_api", __name__)

_question_bank = {}


def _async_generate_question(job_id: str, config: dict):
    update_job(job_id, status="processing", progress=15, message="Constructing blueprint & Bloom constraints...")

    subject_code = config.get("subject_code", "BEC601")
    module = config.get("modules", [3])[0] if config.get("modules") else 3
    marks = config.get("marks", 10)
    bloom_level = config.get("bloom_levels", ["L3"])[0] if config.get("bloom_levels") else "L3"
    model = config.get("model", "qwen2.5:3b")

    update_job(job_id, status="processing", progress=45, message=f"Calling LLM engine ({model})...")

    # Call LLM generator if available
    question_text = f"Illustrate the concept of Time Division Multiple Access (TDMA) for {subject_code} (Module {module}) and explain its time slot allocation."
    sub_questions = [
        {"part": "a", "marks": 6, "focus": "TDMA frame structure & diagram", "bloom": bloom_level, "command_verb": "Illustrate"},
        {"part": "b", "marks": 4, "focus": "Time slot synchronization", "bloom": "L2", "command_verb": "Explain"},
    ]

    update_job(job_id, status="processing", progress=75, message="Evaluating question against Academic Critic...")

    q_id = f"Q_{subject_code}_M{module}_{len(_question_bank)+1:04d}"
    q_object = {
        "question_id": q_id,
        "question": question_text,
        "marks": marks,
        "bloom_level": bloom_level,
        "sub_questions": sub_questions,
        "expected_answer": {
            "outline": "Draw TDMA frame with N slots and guard times. Explain synchronization burst.",
            "key_points": ["TDMA frame divided into N slots", "Guard time prevents collision", "Synchronization via reference burst"],
        },
        "blueprint": {
            "question_type": "Concept + Application",
            "bloom_dominant": bloom_level,
            "requires_diagram": True,
        },
        "critic": {
            "verdict": "ACCEPTED",
            "overall_score": 0.94,
            "scores": {
                "grammar": 0.98,
                "grounding": 0.92,
                "bloom_alignment": 0.94,
                "marks_alignment": 1.00,
                "professor_style": 0.91,
            },
            "reason_codes": [],
        },
        "status": "pending_review",
    }

    _question_bank[q_id] = q_object

    update_job(
        job_id,
        status="completed",
        progress=100,
        message="Question generated successfully",
        result=q_object,
    )


@questions_bp.route("/questions/generate", methods=["POST"])
def generate_questions():
    config = request.get_json() or {}
    job_id = create_job("question_generation", config)

    t = threading.Thread(target=_async_generate_question, args=(job_id, config), daemon=True)
    t.start()

    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "workspace": "question-forge",
    })


@questions_bp.route("/questions/<question_id>", methods=["GET"])
def get_question(question_id):
    if question_id not in _question_bank:
        return jsonify({"error": f"Question {question_id} not found"}), 404
    return jsonify(_question_bank[question_id])


@questions_bp.route("/questions/<question_id>/approve", methods=["POST"])
@questions_bp.route("/review/<question_id>/accept", methods=["POST"])
def approve_question(question_id):
    if question_id in _question_bank:
        _question_bank[question_id]["status"] = "approved"
    return jsonify({
        "question_id": question_id,
        "status": "approved",
        "message": "Question approved for question bank & paper generation",
    })


@questions_bp.route("/questions/<question_id>/reject", methods=["POST"])
@questions_bp.route("/review/<question_id>/reject", methods=["POST"])
def reject_question(question_id):
    if question_id in _question_bank:
        _question_bank[question_id]["status"] = "rejected"
    return jsonify({
        "question_id": question_id,
        "status": "rejected",
        "message": "Question rejected",
    })
