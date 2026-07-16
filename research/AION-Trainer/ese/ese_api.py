# AION-Trainer/ese/ese_api.py
"""
ESE API Router — exposes FastAPI endpoints for the Examiner Simulation Engine.
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from acb.acb_pipeline import ACBPipeline
from ese.examiner_simulation_engine import ExaminerSimulationEngine
from ese.exam_blueprint import ExamBlueprint
from ese.chief_examiner import ChiefExaminerReport

router = APIRouter(prefix="/ese", tags=["Examiner Simulation Engine"])


class GeneratePaperRequest(BaseModel):
    subject_code: str
    subject_name: str
    academic_root: str
    semester: int = 4
    department: str = "AIML"
    previously_asked: Optional[List[str]] = None
    include_optional: bool = True


class OverrideSlotRequest(BaseModel):
    academic_root: str
    slot_id: str
    overridden_text: str
    semester: int = 4
    department: str = "AIML"


class RegenerateSlotRequest(BaseModel):
    academic_root: str
    slot_id: str
    semester: int = 4
    department: str = "AIML"


def get_exams_dir(academic_root: str, department: str, semester: int, subject_code: str) -> Path:
    pipeline = ACBPipeline(
        subject_code=subject_code,
        academic_root=academic_root,
        department=department,
        semester=semester,
    )
    exams_dir = pipeline.db_dir / "exams"
    exams_dir.mkdir(parents=True, exist_ok=True)
    return exams_dir


def load_paper_file(exams_dir: Path, blueprint_id: str) -> Dict[str, Any]:
    file_path = exams_dir / f"{blueprint_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Exam blueprint not found")
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def save_paper_file(exams_dir: Path, blueprint_id: str, data: Dict[str, Any]):
    file_path = exams_dir / f"{blueprint_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


@router.post("/generate")
def generate_question_paper(req: GeneratePaperRequest):
    """
    Triggers ESE to blueprint, plan, discover, rank, realize, and validate
    a complete question paper matching VTU guidelines.
    """
    try:
        pipeline = ACBPipeline(
            subject_code=req.subject_code,
            academic_root=req.academic_root,
            department=req.department,
            semester=req.semester,
        )
        
        engine = ExaminerSimulationEngine(pipeline.concept_store)
        blueprint, report, metadata = engine.generate_paper(
            subject_code=req.subject_code,
            subject_name=req.subject_name,
            semester=req.semester,
            previously_asked=req.previously_asked,
            include_optional=req.include_optional,
        )

        # Structure saving dictionary
        serialized_metadata = {sid: meta.to_dict() for sid, meta in metadata.items()}
        paper_data = {
            "blueprint": blueprint.to_dict(),
            "report": report.__dict__,
            "metadata": serialized_metadata,
            "is_promoted": False,
        }

        exams_dir = get_exams_dir(req.academic_root, req.department, req.semester, req.subject_code)
        save_paper_file(exams_dir, blueprint.blueprint_id, paper_data)

        return {
            "status": "success",
            "blueprint": blueprint.to_dict(),
            "report": {
                "passed": report.passed,
                "overall_quality": report.overall_quality,
                "flags": [f.__dict__ for f in report.flags],
                "slots_to_regenerate": report.slots_to_regenerate,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper/{subject_code}/{blueprint_id}")
def get_exam_paper(
    subject_code: str,
    blueprint_id: str,
    academic_root: str,
    department: str = "AIML",
    semester: int = 4,
):
    """
    Retrieve details of a generated exam paper blueprint.
    """
    try:
        exams_dir = get_exams_dir(academic_root, department, semester, subject_code)
        paper_data = load_paper_file(exams_dir, blueprint_id)
        return paper_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/paper/{subject_code}/{blueprint_id}/override")
def override_question_slot(
    subject_code: str,
    blueprint_id: str,
    req: OverrideSlotRequest,
):
    """
    Apply manual overrides by the user to a specific slot's question text.
    Recalculates grammar and VTU checks, and chief examiner balance gates.
    """
    try:
        exams_dir = get_exams_dir(req.academic_root, req.department, req.semester, subject_code)
        paper_data = load_paper_file(exams_dir, blueprint_id)

        blueprint_dict = paper_data["blueprint"]
        metadata_dict = paper_data["metadata"]

        # 1. Update text inside slot
        slots = blueprint_dict.get("slots", [])
        slot = next((s for s in slots if s["slot_id"] == req.slot_id), None)
        if not slot:
            raise HTTPException(status_code=404, detail="Slot not found")
        
        slot["question_text"] = req.overridden_text
        slot["filled"] = True

        # 2. Recalculate validation rules for the override text
        pipeline = ACBPipeline(
            subject_code=subject_code,
            academic_root=req.academic_root,
            department=req.department,
            semester=req.semester,
        )
        engine = ExaminerSimulationEngine(pipeline.concept_store)

        grammar_issues = engine.grammar_validator.validate(req.overridden_text)
        vtu_issues = engine.vtu_validator.validate(
            req.overridden_text, slot["bloom_level"], slot["marks"], slot.get("diagram_required", False)
        )

        # 3. Update slot metadata
        meta_id = slot.get("question_metadata_id") or ""
        meta = metadata_dict.get(req.slot_id)
        if meta:
            meta["realized_text"] = req.overridden_text
            meta["grammar_issues"] = [i.__dict__ for i in grammar_issues]
            meta["vtu_issues"] = [i.__dict__ for i in vtu_issues]
            meta["status"] = "verified"
        else:
            # Create a new stub if not found
            metadata_dict[req.slot_id] = {
                "metadata_id": meta_id,
                "slot_id": req.slot_id,
                "concept_id": slot["concept_id"],
                "concept_name": slot["concept_name"],
                "bloom_level": slot["bloom_level"],
                "marks": slot["marks"],
                "question_type": slot["question_type"],
                "realized_text": req.overridden_text,
                "grammar_issues": [i.__dict__ for i in grammar_issues],
                "vtu_issues": [i.__dict__ for i in vtu_issues],
                "status": "verified"
            }

        # 4. Re-evaluate overall paper report
        blueprint_obj = ExamBlueprint.from_dict(blueprint_dict)
        blueprint_obj.compute_distributions()
        report = engine.chief_examiner.evaluate_paper(blueprint_obj)

        # 5. Save back
        paper_data["blueprint"] = blueprint_obj.to_dict()
        paper_data["report"] = report.__dict__
        save_paper_file(exams_dir, blueprint_id, paper_data)

        return {
            "status": "success",
            "blueprint": blueprint_obj.to_dict(),
            "report": report.__dict__,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/paper/{subject_code}/{blueprint_id}/regenerate-slot")
def regenerate_question_slot(
    subject_code: str,
    blueprint_id: str,
    req: RegenerateSlotRequest,
):
    """
    Reruns ESE generation specifically for a single slot, ensuring novelty.
    """
    try:
        exams_dir = get_exams_dir(req.academic_root, req.department, req.semester, subject_code)
        paper_data = load_paper_file(exams_dir, blueprint_id)

        blueprint_dict = paper_data["blueprint"]
        metadata_dict = paper_data["metadata"]

        # Parse blueprint object
        blueprint_obj = ExamBlueprint.from_dict(blueprint_dict)
        slot = next((s for s in blueprint_obj.slots if s.slot_id == req.slot_id), None)
        if not slot:
            raise HTTPException(status_code=404, detail="Slot not found")

        # Collect text history to guarantee novelty
        current_texts = [s.question_text for s in blueprint_obj.slots if s.question_text and s.slot_id != req.slot_id]

        # Init Engine
        pipeline = ACBPipeline(
            subject_code=subject_code,
            academic_root=req.academic_root,
            department=req.department,
            semester=req.semester,
        )
        engine = ExaminerSimulationEngine(pipeline.concept_store)

        # Clear and regenerate slot
        slot.filled = False
        slot.question_text = ""
        
        # Populate
        meta_dict: Dict[str, Any] = {}
        engine._populate_single_slot(slot, meta_dict, current_texts)

        # Merge new metadata back
        if req.slot_id in meta_dict:
            metadata_dict[req.slot_id] = meta_dict[req.slot_id].to_dict()

        # Re-evaluate paper
        blueprint_obj.compute_distributions()
        report = engine.chief_examiner.evaluate_paper(blueprint_obj)

        # Save back
        paper_data["blueprint"] = blueprint_obj.to_dict()
        paper_data["report"] = report.__dict__
        save_paper_file(exams_dir, blueprint_id, paper_data)

        return {
            "status": "success",
            "blueprint": blueprint_obj.to_dict(),
            "report": report.__dict__,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/paper/{subject_code}/{blueprint_id}/promote")
def promote_exam_paper(
    subject_code: str,
    blueprint_id: str,
    academic_root: str,
    department: str = "AIML",
    semester: int = 4,
):
    """
    Flags the generated paper as promoted (approved for final printing).
    """
    try:
        exams_dir = get_exams_dir(academic_root, department, semester, subject_code)
        paper_data = load_paper_file(exams_dir, blueprint_id)

        paper_data["is_promoted"] = True
        save_paper_file(exams_dir, blueprint_id, paper_data)

        return {"status": "success", "blueprint_id": blueprint_id, "is_promoted": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
