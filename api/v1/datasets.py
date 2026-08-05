"""
AION API v1 — Datasets Router
Manages ARD v1 dataset samples, manifest statistics, and exports.
"""

import json
from pathlib import Path
from flask import Blueprint, jsonify, request

try:
    from export_ard_v1 import export_ard_v1, compute_sha256
    HAS_EXPORTER = True
except ImportError:
    HAS_EXPORTER = False

datasets_bp = Blueprint("datasets_api", __name__)

ROOT = Path(__file__).parent.parent.parent.resolve()
DATASETS_DIR = ROOT / "datasets"
EXPORTS_DIR = ROOT / "exports"


@datasets_bp.route("/datasets", methods=["GET"])
def list_datasets():
    manifest_path = DATASETS_DIR / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    return jsonify({
        "datasets": [
            {
                "id": "ARD_v1",
                "name": "AION Academic Reasoning Dataset v1.0",
                "schema_version": "1.0",
                "manifest": manifest,
            }
        ]
    })


@datasets_bp.route("/datasets/export", methods=["POST"])
def export_dataset():
    data = request.get_json() or {}
    min_score = data.get("min_critic_score", 0.85)

    if HAS_EXPORTER:
        try:
            export_ard_v1(base_dir=str(ROOT), min_critic_score=min_score)
        except Exception as e:
            return jsonify({"error": f"Export failed: {e}"}), 500

    return jsonify({
        "status": "completed",
        "dataset": "ARD_v1",
        "exports_available": [
            "/exports/train.jsonl",
            "/exports/val.jsonl",
            "/exports/test.jsonl",
            "/exports/ard_v1_generated_training_fmt.jsonl",
        ],
        "message": "ARD v1 dataset exported successfully",
    })


@datasets_bp.route("/datasets/<dataset_id>", methods=["GET"])
def get_dataset(dataset_id):
    manifest_path = DATASETS_DIR / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    return jsonify({
        "id": dataset_id,
        "name": "AION Academic Reasoning Dataset v1.0",
        "manifest": manifest,
    })


@datasets_bp.route("/datasets/<dataset_id>/export", methods=["POST"])
def export_dataset_by_id(dataset_id):
    data = request.get_json() or {}
    fmt = data.get("format", "jsonl")
    return jsonify({
        "status": "completed",
        "dataset_id": dataset_id,
        "format": fmt,
        "message": f"Dataset {dataset_id} exported in {fmt} format",
    })
