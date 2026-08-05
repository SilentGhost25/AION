"""
AION API v1 — Knowledge Router
Exposes concept graph nodes, subjects, and source grounding.
"""

from flask import Blueprint, jsonify, request

knowledge_bp = Blueprint("knowledge_api", __name__)


@knowledge_bp.route("/knowledge/subjects", methods=["GET"])
def get_subjects():
    subjects = [
        {"subject_code": "BEC601", "name": "Satellite Communication", "department": "ECE", "modules": 5},
        {"subject_code": "BCS401", "name": "Data Structures & Algorithms", "department": "CSE", "modules": 5},
    ]
    return jsonify({"subjects": subjects})


@knowledge_bp.route("/knowledge/graph", methods=["GET"])
def get_knowledge_graph():
    subject_code = request.args.get("subject_id", "BEC601")

    nodes = [
        {"id": "SATCOM_M3_TDMA_001", "label": "Time Division Multiple Access (TDMA)", "type": "concept", "confidence": 0.96, "importance": 0.88},
        {"id": "SATCOM_M3_FDMA_002", "label": "Frequency Division Multiple Access (FDMA)", "type": "concept", "confidence": 0.94, "importance": 0.85},
        {"id": "SATCOM_M3_GUARDTIME_003", "label": "Guard Time", "type": "concept", "confidence": 0.92, "importance": 0.78},
        {"id": "SATCOM_M3_SYNCHRONIZATION_004", "label": "Reference Burst Synchronization", "type": "concept", "confidence": 0.95, "importance": 0.82},
    ]

    edges = [
        {"source": "SATCOM_M3_TDMA_001", "target": "SATCOM_M3_FDMA_002", "relation": "compared_with"},
        {"source": "SATCOM_M3_TDMA_001", "target": "SATCOM_M3_GUARDTIME_003", "relation": "requires"},
        {"source": "SATCOM_M3_TDMA_001", "target": "SATCOM_M3_SYNCHRONIZATION_004", "relation": "requires"},
    ]

    return jsonify({
        "subject_code": subject_code,
        "nodes": nodes,
        "edges": edges,
    })


@knowledge_bp.route("/knowledge/concepts/<path:concept_id>", methods=["GET"])
def get_concept(concept_id):
    return jsonify({
        "concept_id": concept_id,
        "label": "Time Division Multiple Access (TDMA)",
        "subject_code": "BEC601",
        "module": 3,
        "importance": 0.88,
        "sources": ["Module3_MultipleAccessTechniques.pdf"],
        "definition": "TDMA allows multiple users to share the same frequency channel by dividing signal into different time slots.",
    })
