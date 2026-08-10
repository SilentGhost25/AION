"""
AION API v1 — Health & Diagnostics Router
Provides server readiness, model status, and subsystem health.
"""

import os
import sys
import requests
from flask import Blueprint, jsonify

from core.config.production_model import get_production_model, get_resolution_info

health_bp = Blueprint("health_api", __name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


@health_bp.route("/health", methods=["GET"])
def get_health():
    resolution = get_resolution_info()
    ollama_ok = False
    models_count = 0
    try:
        r = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_ok = True
            models_count = len(r.json().get("models", []))
    except Exception:
        ollama_ok = False

    return jsonify({
        "status": "ready" if ollama_ok else "degraded",
        "api_version": "v1.0",
        "device": resolution["device"],
        "model": resolution["resolved_model"],
        "ollama": ollama_ok,
        "model_loaded": ollama_ok,
        "pipeline": "unified",
        "vre": True,
        "auto_healer": True,
        "models_available": models_count,
        "model_policy": "single_production_model_authority",
    })


@health_bp.route("/diagnostics", methods=["GET"])
def get_diagnostics():
    ollama_status = "unreachable"
    available_models = []
    try:
        r = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_status = "healthy"
            available_models = [m.get("name") for m in r.json().get("models", [])]
    except Exception as e:
        ollama_status = f"error: {e}"

    return jsonify({
        "system_health": "ready" if ollama_status == "healthy" else "degraded",
        "python_version": sys.version.split()[0],
        "production_model": get_production_model(),
        "ollama": {
            "status": ollama_status,
            "url": OLLAMA_URL,
            "models": available_models,
        },
        "environment": {
            "AION_MODEL": get_production_model(),
            "OLLAMA_URL": OLLAMA_URL,
        },
        "pipeline": {
            "philosophy": "Upload → Extract → Understand → Build Concept Graph → Ground → Reason → Plan → Compose → Audit → Output",
            "llm_role": "component, not architecture",
        },
    })
