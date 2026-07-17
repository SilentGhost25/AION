# learning_engine/learning_api.py
"""
FastAPI Router for AION Learning Engine.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from learning_engine.orchestrator import LearningOrchestrator

logger = logging.getLogger("aion.learning.api")
router = APIRouter(prefix="/learning", tags=["Learning Engine"])

# Cache orchestrator instances per subject
_orchestrators: Dict[str, LearningOrchestrator] = {}


def get_orchestrator(subject_code: str, academic_root: str = "academic") -> LearningOrchestrator:
    key = f"{subject_code}_{academic_root}"
    if key not in _orchestrators:
        _orchestrators[key] = LearningOrchestrator(
            subject_code=subject_code,
            academic_root=academic_root,
        )
    return _orchestrators[key]


class EpochRequest(BaseModel):
    subject_code: str
    epoch: int
    academic_root: str = "academic"


@router.post("/epoch")
async def run_learning_epoch(req: EpochRequest):
    try:
        orch = get_orchestrator(req.subject_code, req.academic_root)
        report = orch.run_epoch(req.epoch)
        return {"status": "success", "report": report.to_dict()}
    except Exception as e:
        logger.exception("Failed to run learning epoch")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_learning_status(subject_code: str, academic_root: str = "academic"):
    try:
        orch = get_orchestrator(subject_code, academic_root)
        iq_details = orch.calculate_iq()
        latest_report = orch.progress_tracker.get_latest()
        return {
            "subject_code": subject_code,
            "academic_iq": iq_details.iq_score,
            "details": {
                "concept_understanding": iq_details.concept_understanding,
                "relationships": iq_details.relationships,
                "question_quality": iq_details.question_quality,
                "answer_quality": iq_details.answer_quality,
                "examiner_style": iq_details.examiner_style,
                "confidence": iq_details.confidence,
            },
            "latest_report": latest_report.to_dict() if latest_report else None,
        }
    except Exception as e:
        logger.exception("Failed to get learning status")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports")
async def get_epoch_reports(subject_code: str, academic_root: str = "academic"):
    try:
        orch = get_orchestrator(subject_code, academic_root)
        reports = orch.progress_tracker.get_all()
        return {"reports": [r.to_dict() for r in reports]}
    except Exception as e:
        logger.exception("Failed to get epoch reports")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/concept")
async def get_concept_memory(subject_code: str, concept_id: str, academic_root: str = "academic"):
    try:
        orch = get_orchestrator(subject_code, academic_root)
        orch.bootstrap()
        entry = orch.concept_memory.get(concept_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found in memory.")
        return {"entry": entry.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get concept memory")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_progress(subject_code: str, academic_root: str = "academic"):
    try:
        orch = get_orchestrator(subject_code, academic_root)
        orch.progress_tracker.clear()
        orch.progress_tracker.save()
        return {"status": "success", "message": "Progress history cleared."}
    except Exception as e:
        logger.exception("Failed to clear progress")
        raise HTTPException(status_code=500, detail=str(e))
