# aion_web/training/backend_api.py
"""
Backend API — Flask routes consumed by the Training Studio UI.

/api/training/mode           GET / POST  — read or switch mode
/api/training/health         GET         — backend health check
/api/training/analyse        POST        — run analysis
/api/training/train          POST        — start training
/api/training/progress/<id>  GET         — SSE progress stream
/api/training/cancel/<id>    POST        — cancel job
/api/training/resolve        POST        — resolve ambiguity
/api/training/confirm        POST        — confirm course preview
"""

from __future__ import annotations

import json
from dataclasses import asdict
from flask import Blueprint, request, jsonify, Response, current_app, stream_with_context

from aion_web.training.backends.base import (
    TrainingMode, BackendError, BackendBusyError,
)
from aion_web.training.backend_registry import BackendRegistry
from aion_web.training.mode_config import ModeConfig

training_bp = Blueprint("training", __name__, url_prefix="/api/training")


def get_registry() -> BackendRegistry:
    return current_app.config["BACKEND_REGISTRY"]


def get_mode_config() -> ModeConfig:
    return current_app.config["MODE_CONFIG"]


# -- Mode management -----------------------------------------------------------

@training_bp.get("/mode")
def get_mode():
    registry = get_registry()
    mode_config = get_mode_config()
    backend = registry.get()
    return jsonify({
        "active_mode": backend.mode.value,
        "display_name": backend.display_name,
        "color": backend.color,
        "is_busy": backend.is_busy,
        "config": {
            "server_url": mode_config.server_url,
            "local_model": mode_config.local_model,
            "auto_connect": mode_config.auto_connect,
        },
    })


@training_bp.post("/mode")
def switch_mode():
    registry = get_registry()
    data = request.get_json(silent=True) or {}

    raw_mode = data.get("mode")
    if not raw_mode:
        return jsonify({"error": "mode is required"}), 400

    try:
        new_mode = TrainingMode(raw_mode)
    except ValueError:
        return jsonify({
            "error": f"Invalid mode '{raw_mode}'. "
                     f"Choose from: {[m.value for m in TrainingMode]}"
        }), 400

    # Build updated config from request
    mode_config = get_mode_config()
    if "server_url" in data:
        mode_config.server_url = data["server_url"]
    if "server_token" in data:
        mode_config.server_token = data["server_token"]
    if "local_model" in data:
        mode_config.local_model = data["local_model"]
    if "auto_connect" in data:
        mode_config.auto_connect = data["auto_connect"]
    mode_config.active_mode = new_mode.value

    try:
        backend = registry.switch(new_mode, mode_config)
    except BackendBusyError as e:
        return jsonify({"error": str(e), "busy": True}), 409
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "switched_to": new_mode.value,
        "display_name": backend.display_name,
        "color": backend.color,
    })


# -- Health --------------------------------------------------------------------

@training_bp.get("/health")
def get_health():
    try:
        registry = get_registry()
        backend = registry.get()
        return jsonify(backend.health_check())
    except Exception as e:
        return jsonify({"healthy": False, "details": str(e)}), 500


# -- Core Operations -----------------------------------------------------------

@training_bp.post("/analyse")
def run_analyse():
    registry = get_registry()
    backend = registry.get()
    data = request.get_json(silent=True) or {}
    
    file_paths = data.get("file_paths")
    if not isinstance(file_paths, list):
        return jsonify({"error": "file_paths list is required"}), 400
        
    subject_code = data.get("subject_code", "")
    
    try:
        output = backend.analyse(file_paths, subject_code)
        return jsonify(asdict(output))
    except BackendError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal error during analysis: {e}"}), 500


@training_bp.post("/train")
def run_train():
    registry = get_registry()
    backend = registry.get()
    data = request.get_json(silent=True) or {}
    
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
        
    subject_code = data.get("subject_code")
    if not subject_code:
        return jsonify({"error": "subject_code is required"}), 400
        
    try:
        handle = backend.train(session_id, subject_code)
        return jsonify(asdict(handle))
    except BackendError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal error starting training: {e}"}), 500


@training_bp.get("/progress/<job_id>")
def get_progress_stream(job_id):
    registry = get_registry()
    backend = registry.get()
    
    @stream_with_context
    def generate():
        try:
            for event in backend.get_progress(job_id):
                payload = {
                    "message": event.message,
                    "metrics": {
                        "fraction": event.fraction,
                        **(event.metrics or {})
                    },
                    "stage": event.stage,
                    "terminal": event.is_terminal,
                    "level": "ERROR" if event.is_error else "INFO"
                }
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            payload = {
                "message": f"Progress stream failed internally: {e}",
                "metrics": {"fraction": 1.0},
                "stage": "error",
                "terminal": True,
                "level": "ERROR"
            }
            yield f"data: {json.dumps(payload)}\n\n"
            
    return Response(generate(), mimetype="text/event-stream")


@training_bp.post("/cancel/<job_id>")
def run_cancel(job_id):
    registry = get_registry()
    backend = registry.get()
    try:
        success = backend.cancel(job_id)
        return jsonify({"cancelled": success})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.post("/resolve")
def run_resolve():
    registry = get_registry()
    backend = registry.get()
    data = request.get_json(silent=True) or {}
    
    session_id = data.get("session_id")
    ambiguity_id = data.get("ambiguity_id")
    action = data.get("action")
    value = data.get("value")
    
    if not all([session_id, ambiguity_id, action]):
        return jsonify({"error": "session_id, ambiguity_id, action are required"}), 400
        
    try:
        res = backend.resolve_ambiguity(session_id, ambiguity_id, action, value)
        return jsonify(res)
    except BackendError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.post("/confirm")
def run_confirm():
    registry = get_registry()
    backend = registry.get()
    data = request.get_json(silent=True) or {}
    
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
        
    try:
        success = backend.confirm_course(session_id)
        return jsonify({"success": success})
    except BackendError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
