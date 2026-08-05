"""
AION API v1 — Health & Diagnostics Router
Provides health checks and system diagnostics.
"""

import os
import sys
import requests
from flask import Blueprint, jsonify

health_bp = Blueprint("health_api", __name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


@health_bp.route("/health", methods=["GET"])
def get_health():
    ollama_ok = False
    models_count = 0
    try:
        r = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_ok = True
            models_count = len(r.json().get("models", []))
    except Exception:
        ollama_ok = False

    active_model = os.environ.get("AION_MODEL", "qwen2.5:3b")

    return jsonify({
        "status": "healthy" if ollama_ok else "degraded",
        "api_version": "v1.0",
        "services": {
            "aion_api": "healthy",
            "ollama": "healthy" if ollama_ok else "unreachable",
        },
        "active_model": active_model,
        "models_available": models_count,
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
        "system_health": "healthy" if ollama_status == "healthy" else "degraded",
        "python_version": sys.version.split()[0],
        "ollama": {
            "status": ollama_status,
            "url": OLLAMA_URL,
            "models": available_models,
        },
        "environment": {
            "AION_MODEL": os.environ.get("AION_MODEL", "qwen2.5:3b"),
            "OLLAMA_URL": OLLAMA_URL,
        },
    })
