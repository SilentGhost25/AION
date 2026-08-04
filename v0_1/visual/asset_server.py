"""
AION Visual: Asset Server Route Registration
Serves extracted figure images securely over Flask API.
"""

from __future__ import annotations

import os
from pathlib import Path
from flask import send_from_directory, abort, jsonify


def register_asset_routes(app, base_dir: str = "extracted_output/assets"):
    """
    Registers /api/assets/<filename> route on Flask app.
    """
    assets_path = Path(base_dir).resolve()
    assets_path.mkdir(parents=True, exist_ok=True)

    @app.route("/api/assets/<path:filename>", methods=["GET"])
    def serve_asset(filename):
        file_p = assets_path / filename
        if not file_p.exists():
            return jsonify({"error": f"Asset not found: {filename}"}), 404
        return send_from_directory(assets_path, filename)

    print(f"[ASSET_SERVER] Asset route registered -> /api/assets/<filename> (serving from {assets_path})")
