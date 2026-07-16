# AION-Trainer/training_studio/studio_api.py
"""
Studio API Router — exposes FastAPI endpoints for the Training Studio.
"""

from __future__ import annotations

import os
import uuid
import shutil
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from training_studio.studio_session import TrainingStudioSession
from training_studio.preview.course_preview_builder import CoursePreviewBuilder
from training_studio.classifier.document_classifier import DocumentType
from acb.syllabus_parser import SyllabusParser
from server.job_queue import JobQueue, Job

logger = logging.getLogger("aion.studio.api")

router = APIRouter(prefix="/studio", tags=["Training Studio"])

# In-memory session store
sessions: Dict[str, TrainingStudioSession] = {}


class ResolveRequest(BaseModel):
    ambiguity_id: str
    option_index: int


class TrainRequest(BaseModel):
    academic_root: str
    epochs: int = 3
    learning_rate: float = 5e-5


@router.post("/upload")
async def upload_files(
    subject_code: str = Form(...),
    academic_root: str = Form(...),
    department: str = Form("AIML"),
    semester: int = Form(4),
    files: List[UploadFile] = File(...),
):
    """
    Creates a new Training Studio session and saves the uploaded files.
    """
    try:
        session_id = str(uuid_session_id())
        session = TrainingStudioSession(session_id=session_id)
        
        # Load existing syllabus if available
        subj_dir = Path(academic_root) / department / f"semester_{semester}" / subject_code
        if not subj_dir.exists():
            subj_dir = Path(academic_root) / subject_code
        
        syllabus_dir = subj_dir / "syllabus"
        syllabus = None
        if syllabus_dir.exists():
            syllabus_files = list(syllabus_dir.glob("*.pdf")) + list(syllabus_dir.glob("*.docx")) + list(syllabus_dir.glob("*.txt"))
            if syllabus_files:
                parser = SyllabusParser()
                try:
                    syllabus = parser.parse_file(str(syllabus_files[0]), subject_code=subject_code)
                    session.syllabus = syllabus
                    session.module_mapper.syllabus = syllabus
                except Exception as e:
                    logger.warning(f"Failed to pre-load syllabus: {e}")

        # Create session upload directory
        upload_dir = Path("scratch/studio_uploads") / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            session.add_file(str(file_path))

        session.result.subject_code = subject_code
        session.result.department = department
        session.result.semester = semester
        
        sessions[session_id] = session

        return {"session_id": session_id, "total_files": len(files)}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/analyse")
def analyse_session(session_id: str):
    """
    Starts analysis for the uploaded documents.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        result = session.start_analysis()

        # Dynamic Syllabus Detection:
        # If no syllabus was pre-loaded, check if one of the uploaded files is a syllabus
        if not session.syllabus:
            syllabus_fa = next((fa for fa in result.file_analyses if fa.document_type == DocumentType.SYLLABUS), None)
            if syllabus_fa:
                logger.info(f"Syllabus file detected: {syllabus_fa.filename}. Parsing dynamically...")
                parser = SyllabusParser()
                try:
                    syllabus = parser.parse_file(syllabus_fa.file_path, subject_code=result.subject_code)
                    session.syllabus = syllabus
                    session.module_mapper.syllabus = syllabus
                    
                    # Re-run analysis now that syllabus is loaded (so module mapping can run)
                    result = session.start_analysis()
                except Exception as e:
                    logger.warning(f"Dynamic syllabus parsing failed: {e}")

        return result.to_dict()
    except Exception as e:
        logger.error(f"Analysis failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/status")
def get_session_status(session_id: str):
    """
    Gets the current analysis status.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.result.to_dict()


@router.get("/session/{session_id}/preview")
def get_course_preview(session_id: str):
    """
    Returns the course tree preview.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    builder = CoursePreviewBuilder()
    return builder.build_tree(session.result)


@router.post("/session/{session_id}/resolve")
def resolve_session_ambiguity(session_id: str, req: ResolveRequest):
    """
    Resolves a specific ambiguity.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    success = session.resolve_ambiguity(req.ambiguity_id, req.option_index)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to resolve ambiguity")
    
    return session.result.to_dict()


@router.post("/session/{session_id}/train")
def trigger_training(session_id: str, req: TrainRequest):
    """
    Triggers model training if the session is ready.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.result.compute_readiness()
    if not session.result.train_enabled:
        raise HTTPException(status_code=400, detail="Cannot train. Unresolved errors remain or analysis incomplete.")

    try:
        # Move files from scratch/studio_uploads to the official academic folders
        target_subj_dir = Path(req.academic_root) / session.result.department / f"semester_{session.result.semester}" / session.result.subject_code
        if not target_subj_dir.exists():
            target_subj_dir = Path(req.academic_root) / session.result.subject_code
        
        type_mappings = {
            DocumentType.TEXTBOOK: "textbooks",
            DocumentType.NOTES: "notes",
            DocumentType.QUESTION_BANK: "question_bank",
            DocumentType.PREVIOUS_PAPER: "previous_papers",
            DocumentType.ANSWER_KEY: "answer_keys",
            DocumentType.SYLLABUS: "syllabus",
        }

        for fa in session.result.file_analyses:
            if fa.status != "complete":
                continue
            subfolder = type_mappings.get(fa.document_type)
            if not subfolder:
                continue
            
            dest_dir = target_subj_dir / subfolder
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            src_file = Path(fa.file_path)
            dest_file = dest_dir / src_file.name
            
            if src_file.exists():
                shutil.copy(src_file, dest_file)
                # Update file_path in result to point to official storage
                fa.file_path = str(dest_file)

        # Submit Job to pipeline queue
        job_queue = JobQueue(os.path.join(req.academic_root, "jobs.db"))
        job = job_queue.submit(
            subject=session.result.subject_code,
            job_type="train",
            resource="gpu",
            params={
                "epochs": req.epochs,
                "learning_rate": req.learning_rate,
                "department": session.result.department,
                "semester": session.result.semester,
            }
        )

        return {
            "status": "success",
            "message": "Training job submitted successfully",
            "job": {
                "job_id": job.id,
                "subject_code": job.subject,
                "status": job.status,
                "created_at": job.created_at,
            }
        }
    except Exception as e:
        logger.error(f"Failed to submit training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def uuid_session_id() -> str:
    return str(uuid.uuid4())[:8]
