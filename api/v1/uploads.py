"""
AION API v1 — Uploads Router
Handles PDF and academic document uploads.
"""

import os
import uuid
import time
from pathlib import Path
from flask import Blueprint, jsonify, request

uploads_bp = Blueprint("uploads_api", __name__)

ROOT = Path(__file__).parent.parent.parent.resolve()
UPLOAD_DIR = ROOT / "workspace" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".pptx", ".md"}


@uploads_bp.route("/uploads", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    file_obj = request.files["file"]
    if not file_obj or file_obj.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file_obj.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file extension '{ext}'"}), 400

    upload_id = uuid.uuid4().hex[:12]
    saved_filename = f"{upload_id}{ext}"
    saved_path = UPLOAD_DIR / saved_filename

    file_obj.save(saved_path)

    return jsonify({
        "upload_id": upload_id,
        "original_filename": file_obj.filename,
        "saved_path": str(saved_path),
        "file_size": saved_path.stat().st_size,
        "uploaded_at": time.time(),
        "status": "ready",
    })


@uploads_bp.route("/uploads", methods=["GET"])
def list_uploads():
    uploads = []
    if UPLOAD_DIR.exists():
        for p in UPLOAD_DIR.glob("*"):
            if p.is_file():
                uploads.append({
                    "upload_id": p.stem,
                    "filename": p.name,
                    "size_bytes": p.stat().st_size,
                    "modified_at": p.stat().st_mtime,
                })
    return jsonify({"uploads": uploads})
