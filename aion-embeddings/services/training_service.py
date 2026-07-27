import logging
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from core.learning.pair_generator import TrainingPairGenerator
from core.learning.replay_buffer import ReplayBuffer
from core.learning.dataset_builder import DatasetBuilder
from core.learning.embedding_trainer import EmbeddingTrainer
from core.learning.evaluator import EmbeddingEvaluator
from core.learning.model_registry import ModelRegistry
from core.learning.hot_reload import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])

class UploadRequest(BaseModel):
    filename: str
    subject: str

class UploadResponse(BaseModel):
    job_id: str
    status: str

# Job tracking (In a real app, use Redis/DB)
jobs_db = {}

@router.post("/upload")
async def upload_file(
    request: UploadRequest,
    background_tasks: BackgroundTasks
) -> UploadResponse:
    job_id = f"{request.subject}_{datetime.now().isoformat()}"
    
    jobs_db[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Waiting to process"
    }
    
    background_tasks.add_task(
        process_upload_pipeline,
        job_id=job_id,
        filename=request.filename,
        subject=request.subject
    )
    
    return UploadResponse(job_id=job_id, status="queued")

async def process_upload_pipeline(job_id: str, filename: str, subject: str):
    """
    The complete learning pipeline orchestrated via FastAPI background tasks.
    """
    try:
        # Mocking Stage 1 for the Facade
        jobs_db[job_id]["status"] = "extracting"
        jobs_db[job_id]["message"] = "Mock extraction..."
        # In the real facade, you'd pull `questions` from the legacy pipeline.
        # questions = legacy_pipeline.extract(...)
        questions = [{"text": "What is normalization?", "answer": "Normalization is the process of organizing data to minimize redundancy.", "marks": 5}]
        
        jobs_db[job_id]["status"] = "generating_pairs"
        jobs_db[job_id]["progress"] = 20
        
        pair_gen = TrainingPairGenerator()
        pairs = pair_gen.from_question_extraction(questions, subject)
        
        jobs_db[job_id]["status"] = "storing_buffer"
        jobs_db[job_id]["progress"] = 40
        replay_buffer = ReplayBuffer()
        replay_buffer.add_pairs(pairs)
        
        jobs_db[job_id]["status"] = "building_dataset"
        jobs_db[job_id]["progress"] = 50
        dataset_builder = DatasetBuilder(replay_buffer)
        dataset = dataset_builder.build_training_dataset(pairs, subject=subject, replay_ratio=0.3)
        
        jobs_db[job_id]["status"] = "training"
        jobs_db[job_id]["progress"] = 60
        trainer = EmbeddingTrainer(output_dir="data/models")
        # Train with 1 epoch for quick demonstration, normally 3
        checkpoint_path = trainer.train(dataset, subject, epochs=1, batch_size=2)
        
        jobs_db[job_id]["status"] = "evaluating"
        jobs_db[job_id]["progress"] = 80
        evaluator = EmbeddingEvaluator()
        eval_score, is_acceptable, report = evaluator.evaluate(checkpoint_path)
        
        if not is_acceptable:
            jobs_db[job_id]["status"] = "eval_failed"
            jobs_db[job_id]["message"] = f"Evaluation failed. Score: {eval_score:.4f}"
            return
        
        jobs_db[job_id]["status"] = "registering"
        jobs_db[job_id]["progress"] = 90
        registry = ModelRegistry()
        model_id = registry.register_model(checkpoint_path, subject, eval_score)
        
        jobs_db[job_id]["status"] = "reloading"
        jobs_db[job_id]["progress"] = 95
        embedding_service = EmbeddingService()
        embedding_service.switch_model(subject, model_id)
        
        jobs_db[job_id]["status"] = "complete"
        jobs_db[job_id]["message"] = f"Complete! Model {model_id} is active."
        jobs_db[job_id]["progress"] = 100
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        jobs_db[job_id]["status"] = "error"
        jobs_db[job_id]["message"] = str(e)

@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    return jobs_db[job_id]

@router.get("/models")
async def list_models(subject: str = None):
    registry = ModelRegistry()
    return registry.list_models(subject=subject)

@router.post("/models/{model_id}/activate")
async def activate_model(model_id: str):
    registry = ModelRegistry()
    registry.rollback(model_id)
    return {"status": "activated", "model_id": model_id}
