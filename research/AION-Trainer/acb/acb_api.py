# AION-Trainer/acb/acb_api.py
"""
ACB API Router — exposes FastAPI endpoints for the Academic Course Builder.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from acb.acb_pipeline import ACBPipeline
from acb.concept import Concept

router = APIRouter(prefix="/acb", tags=["Academic Course Builder"])


class IngestRequest(BaseModel):
    subject_code: str
    academic_root: str
    department: str = "AIML"
    semester: int = 4


class VerifyConceptRequest(BaseModel):
    status: str                         # verified | needs_verification
    definition: Optional[str] = None
    explanation: Optional[str] = None
    prerequisites: Optional[List[str]] = None


@router.post("/ingest")
def run_acb_ingest(req: IngestRequest):
    """
    Triggers the complete ACB ingestion, deduplication, confidence validation,
    completeness scoring, and report generation pipeline.
    """
    try:
        pipeline = ACBPipeline(
            subject_code=req.subject_code,
            academic_root=req.academic_root,
            department=req.department,
            semester=req.semester,
        )
        results = pipeline.run()
        if results.get("status") == "failed":
            raise HTTPException(status_code=400, detail=results.get("error", "Ingestion failed"))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/concepts/{subject_code}")
def get_subject_concepts(subject_code: str, academic_root: str):
    """
    Get all deduplicated concepts extracted for the given subject.
    """
    try:
        pipeline = ACBPipeline(subject_code=subject_code, academic_root=academic_root)
        concepts = pipeline.concept_store.concepts_for_subject(subject_code)
        if not concepts:
            concepts = pipeline.concept_store.all_concepts()
        return [c.to_dict() for c in concepts]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/concepts/{subject_code}/{concept_id}")
def get_concept_details(subject_code: str, concept_id: str, academic_root: str):
    """
    Get details of a specific concept by ID.
    """
    try:
        pipeline = ACBPipeline(subject_code=subject_code, academic_root=academic_root)
        concept = pipeline.concept_store.get(concept_id)
        if not concept:
            raise HTTPException(status_code=404, detail="Concept not found")
        return concept.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/concepts/{subject_code}/{concept_id}/verify")
def verify_concept(
    subject_code: str,
    concept_id: str,
    academic_root: str,
    req: VerifyConceptRequest,
):
    """
    Enable manual overrides of concept fields, statuses, or dependencies by faculty.
    """
    try:
        pipeline = ACBPipeline(subject_code=subject_code, academic_root=academic_root)
        concept = pipeline.concept_store.get(concept_id)
        if not concept:
            raise HTTPException(status_code=404, detail="Concept not found")
        
        concept.status = req.status
        if req.definition is not None:
            concept.definition = req.definition
        if req.explanation is not None:
            concept.explanation = req.explanation
        if req.prerequisites is not None:
            concept.prerequisites = req.prerequisites
            
        concept.touch()
        pipeline.concept_store.save()
        return {"status": "success", "concept": concept.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/{subject_code}")
def get_intelligence_report(subject_code: str, academic_root: str):
    """
    Retrieve the Course Intelligence Report JSON metadata and markdown path.
    """
    try:
        pipeline = ACBPipeline(subject_code=subject_code, academic_root=academic_root)
        report_json_path = pipeline.db_dir / "course_intelligence_report.json"
        report_md_path = pipeline.db_dir / "course_intelligence_report.md"
        
        if not report_json_path.exists() or not report_md_path.exists():
            raise HTTPException(status_code=404, detail="Report not generated yet. Run ingestion first.")
            
        import json
        with open(report_json_path, encoding="utf-8") as f:
            meta = json.load(f)
            
        return {
            "metadata": meta,
            "markdown_content": report_md_path.read_text(encoding="utf-8"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/{subject_code}")
def get_subject_sources(subject_code: str, academic_root: str):
    """
    Retrieve quality profiles of all files registered in the source registry.
    """
    try:
        pipeline = ACBPipeline(subject_code=subject_code, academic_root=academic_root)
        sources = pipeline.source_registry.all_sources()
        return [s.to_dict() for s in sources if s.subject_code == subject_code]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
