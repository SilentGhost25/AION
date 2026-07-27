import json
import yaml
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from serving.embedder import AIonEmbedder
from vector_store.indexer import AIonIndex

app = FastAPI(
    title="AION Embeddings API",
    description="Educational embedding and retrieval system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global instances (lazy loaded)
_embedder: Optional[AIonEmbedder] = None
_index: Optional[AIonIndex] = None

def get_embedder() -> AIonEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = AIonEmbedder()
    return _embedder

def get_index() -> AIonIndex:
    global _index
    if _index is None:
        _index = AIonIndex()
    return _index

def load_config():
    with open("config/aion_config.yaml") as f:
        return yaml.safe_load(f)

def save_config(config: dict):
    with open("config/aion_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

# ── Request / Response Models ──────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: List[str]
    subject: Optional[str] = None
    normalize: bool = True

class SearchRequest(BaseModel):
    query: str
    subject: Optional[str] = None
    top_k: int = 5
    score_threshold: float = 0.60

class FeedbackRequest(BaseModel):
    query: str
    wrong_result: str
    expected: Optional[str] = None
    subject: Optional[str] = None

class ConfigUpdateRequest(BaseModel):
    updates: Dict[str, Any]  # dot-notation keys → values

class TrainRequest(BaseModel):
    subject: Optional[str] = None
    config_override: Optional[Dict[str, Any]] = None
    run_name: Optional[str] = None

class SwitchSubjectRequest(BaseModel):
    subject: str

class BuildIndexRequest(BaseModel):
    subject: Optional[str] = None

# ── Core Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "model_info": get_embedder().get_model_info()
    }

@app.post("/embed")
async def embed(request: EmbedRequest):
    """Generate embeddings for texts."""
    embedder = get_embedder()
    
    if request.subject:
        embedder.switch_subject(request.subject)
    
    embeddings = embedder.embed(
        request.texts,
        normalize=request.normalize
    )
    
    return {
        "embeddings": embeddings.tolist(),
        "model_info": embedder.get_model_info(),
        "count": len(request.texts)
    }

@app.post("/search")
async def search(request: SearchRequest):
    """Search the vector index."""
    index = get_index()
    
    results = index.search(
        query=request.query,
        subject=request.subject,
        top_k=request.top_k,
        score_threshold=request.score_threshold
    )
    
    return {
        "query": request.query,
        "subject": request.subject,
        "results": results,
        "count": len(results)
    }

# ── Admin Endpoints ────────────────────────────────────────────────────────

@app.post("/admin/switch-subject")
async def switch_subject(request: SwitchSubjectRequest):
    """Hot-swap the active embedding model."""
    embedder = get_embedder()
    result = embedder.switch_subject(request.subject)
    return result

@app.get("/admin/config")
async def get_config():
    """Return the full current config."""
    return load_config()

@app.post("/admin/config")
async def update_config(request: ConfigUpdateRequest):
    """
    Update config values at runtime using dot-notation.
    e.g. {"training.learning_rate": 0.00001}
    """
    config = load_config()
    
    for key, value in request.updates.items():
        keys = key.split(".")
        cfg = config
        for k in keys[:-1]:
            if k not in cfg:
                raise HTTPException(400, f"Config key not found: {k}")
            cfg = cfg[k]
        cfg[keys[-1]] = value
    
    save_config(config)
    
    return {
        "status": "updated",
        "changes": request.updates,
        "message": "Config saved. Some changes require model reload to take effect."
    }

@app.post("/admin/train")
async def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger a training run asynchronously.
    Returns immediately; training runs in background.
    """
    def run_training(subject, config_override, run_name):
        import subprocess
        import sys
        
        cmd = [sys.executable, "training/train.py"]
        if subject:
            cmd.extend(["--subject", subject])
        
        subprocess.run(cmd, check=True)
    
    background_tasks.add_task(
        run_training,
        request.subject,
        request.config_override,
        request.run_name
    )
    
    return {
        "status": "training_started",
        "subject": request.subject or "base",
        "message": "Training job started in background. Check W&B for progress."
    }

@app.post("/admin/build-index")
async def build_index(request: BuildIndexRequest, background_tasks: BackgroundTasks):
    """Rebuild the FAISS index for one or all subjects."""
    index = get_index()
    
    def _build(subject):
        index.build_index(subject=subject)
    
    background_tasks.add_task(_build, request.subject)
    
    return {
        "status": "index_build_started",
        "subject": request.subject or "all",
        "message": "Index building started in background."
    }

@app.post("/admin/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit a 'wrong result' feedback.
    This becomes a hard negative in the next training run.
    """
    feedback_dir = Path("data/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = feedback_dir / "wrong_results.jsonl"
    
    record = {
        "query": request.query,
        "wrong_result": request.wrong_result,
        "expected": request.expected,
        "subject": request.subject,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(feedback_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    
    return {
        "status": "feedback_recorded",
        "message": "This will be used as a hard negative in the next training run."
    }

@app.get("/admin/adapters")
async def list_adapters():
    """List all available subject adapters and their status."""
    adapter_dir = Path("adapters")
    adapters = []
    
    for subject_dir in adapter_dir.iterdir():
        if subject_dir.is_dir():
            has_model = (subject_dir / "model_latest").exists()
            versions = sorted([d.name for d in subject_dir.iterdir() if d.name.startswith("model_2")])
            
            adapters.append({
                "subject": subject_dir.name,
                "has_model": has_model,
                "versions": versions,
                "version_count": len(versions)
            })
    
    config = load_config()
    return {
        "adapters": adapters,
        "current_subject": config["adapters"]["current_subject"]
    }

@app.get("/admin/index-stats")
async def index_stats():
    """Return stats about the current vector indices."""
    index = get_index()
    stats = {}
    
    for subject, faiss_index in index.indices.items():
        stats[subject] = {
            "vector_count": faiss_index.ntotal,
            "dimension": faiss_index.d
        }
    
    return {"indices": stats, "total_chunks": len(index.chunk_texts)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
