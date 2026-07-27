import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# IMPORTANT: Run this from aion-embeddings root so core modules resolve correctly
from core.graph import build_graph

app = FastAPI(title="AION VTU Generator API")
graph_app = build_graph()

class GenerateRequest(BaseModel):
    subject: str
    subject_code: str
    exam_type: str = "SEE"
    duration_minutes: int = 180
    modules_to_cover: List[int] = [1, 2, 3, 4, 5]
    bloom_focus: List[str] = ["L2", "L3", "L4"]

class ReviewRequest(BaseModel):
    verdict: str  # "approve", "revise", "reject"
    feedback: Optional[str] = None

@app.post("/generate")
def generate_paper(req: GenerateRequest):
    run_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": run_id}}
    
    initial_state = {
        "run_id": run_id,
        "subject": req.subject,
        "subject_code": req.subject_code,
        "exam_type": req.exam_type,
        "duration_minutes": req.duration_minutes,
        "modules_to_cover": req.modules_to_cover,
        "bloom_focus": req.bloom_focus,
        "generation_attempt": 0,
        "status": "draft"
    }
    
    # Run the graph until it hits the interrupt (HOD review)
    for output in graph_app.stream(initial_state, config):
        pass
        
    return {"run_id": run_id, "message": "Generation started and queued for review"}

@app.get("/pending")
def get_pending_reviews():
    # Since we are using SqliteSaver, we can theoretically query the state.
    # In LangGraph, we can get state by thread_id. But since we don't know thread_ids easily without a DB of runs, 
    # we would normally have our own table of runs. 
    # For the sake of this prototype, we'll return a mock empty list or require the UI to track run_ids.
    # In a full prod app, we'd query our relational DB for runs where status == "pending_review".
    return {"pending_runs": ["Track run_ids manually for now"]}

@app.get("/paper/{run_id}")
def get_paper(run_id: str):
    config = {"configurable": {"thread_id": run_id}}
    state = graph_app.get_state(config)
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Run not found")
    
    val = state.values
    return {
        "status": val.get("status"),
        "draft_paper": val.get("draft_paper"),
        "validation_errors": val.get("validation_errors"),
        "generation_errors": val.get("generation_errors")
    }

@app.post("/review/{run_id}")
def review_paper(run_id: str, req: ReviewRequest):
    config = {"configurable": {"thread_id": run_id}}
    state = graph_app.get_state(config)
    
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if state.values.get("status") != "pending_review":
        raise HTTPException(status_code=400, detail="Run is not pending review")
        
    # Update state with feedback
    graph_app.update_state(
        config,
        {
            "review_verdict": req.verdict,
            "hod_feedback": req.feedback,
            "status": "review_submitted" # Temporary status update to unblock
        },
        as_node="review"
    )
    
    # Resume the graph
    for output in graph_app.stream(None, config, input=None):
        pass
        
    new_state = graph_app.get_state(config)
    return {"message": "Review processed", "new_status": new_state.values.get("status")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
