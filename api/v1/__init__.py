"""
AION API v1 Package Initialization
Bundles all API v1 domain sub-routers under a single Flask Blueprint.
"""

from flask import Blueprint, jsonify

from api.v1.jobs import jobs_bp
from api.v1.health import health_bp
from api.v1.dashboard import dashboard_bp
from api.v1.uploads import uploads_bp
from api.v1.training import training_bp
from api.v1.questions import questions_bp
from api.v1.papers import papers_bp
from api.v1.critic import critic_bp
from api.v1.datasets import datasets_bp
from api.v1.models import models_bp
from api.v1.knowledge import knowledge_bp

api_v1_bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")


@api_v1_bp.route("", methods=["GET"])
@api_v1_bp.route("/", methods=["GET"])
def api_v1_index():
    return jsonify({
        "name": "AION API Gateway",
        "version": "v1.0",
        "status": "online",
        "documentation": "AION Intelligence & Backend REST API v1",
        "workspaces": {
            "health": "/api/v1/health",
            "diagnostics": "/api/v1/diagnostics",
            "dashboard": "/api/v1/dashboard/summary",
            "uploads": "/api/v1/uploads",
            "training": "/api/v1/training/analyze",
            "question_forge": "/api/v1/questions/generate",
            "paper_forge": "/api/v1/papers/generate",
            "critic": "/api/v1/critic/evaluate",
            "datasets": "/api/v1/datasets",
            "models": "/api/v1/models",
            "knowledge": "/api/v1/knowledge/subjects",
        },
    })

# Register sub-blueprints
api_v1_bp.register_blueprint(jobs_bp)
api_v1_bp.register_blueprint(health_bp)
api_v1_bp.register_blueprint(dashboard_bp)
api_v1_bp.register_blueprint(uploads_bp)
api_v1_bp.register_blueprint(training_bp)
api_v1_bp.register_blueprint(questions_bp)
api_v1_bp.register_blueprint(papers_bp)
api_v1_bp.register_blueprint(critic_bp)
api_v1_bp.register_blueprint(datasets_bp)
api_v1_bp.register_blueprint(models_bp)
api_v1_bp.register_blueprint(knowledge_bp)
