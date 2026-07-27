import json
import logging
import hashlib
from datetime import datetime
from typing import TypedDict, Optional, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError
import instructor
from openai import OpenAI

from .schemas import VTUQuestionPaper

logger = logging.getLogger(__name__)

class PaperState(TypedDict, total=False):
    run_id: str
    created_at: str
    created_by: str
    assigned_to_hod: str
    
    subject: str
    subject_code: str
    exam_type: Literal["CIE", "SEE"]
    duration_minutes: int
    syllabus_version: str
    modules_to_cover: List[int]
    bloom_focus: List[str]
    
    context_chunks: List[str]
    concept_graph: Optional[dict]
    sample_good_questions: List[dict]
    
    draft_paper: Optional[VTUQuestionPaper]
    generation_attempt: int
    generation_errors: List[str]
    validation_errors: List[str]
    
    review_verdict: Optional[Literal["approve", "revise", "reject"]]
    hod_feedback: Optional[str]
    
    status: Literal["draft", "generating", "validating", "pending_review", "approved", "rejected", "archived", "failed_circuit_breaker", "failed_validation"]
    error_message: Optional[str]

# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 1: RAG RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_rich_context(state: PaperState) -> dict:
    """Mock retrieval for now until FAISS/Neo4j are fully wired."""
    return {
        "context_chunks": [
            f"Syllabus coverage for {state.get('subject_name', state.get('subject', ''))} modules {state.get('modules_to_cover', [])}",
            "Key topics include advanced algorithms, database tuning, and distributed systems."
        ],
        "concept_graph": {},
        "sample_good_questions": [],
        "status": "generating"
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 2: GENERATE DRAFT WITH INSTRUCTOR (Guaranteed Schema)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_draft(state: PaperState) -> dict:
    """Generate or fix a draft paper."""
    
    attempt = state.get('generation_attempt', 0)
    if attempt >= 3:
        logger.error(f"Generation failed after {attempt} attempts")
        return {
            "status": "failed_circuit_breaker",
            "error_message": "LLM failed to generate valid paper after 3 attempts.",
            "generation_attempt": attempt
        }
    
    client = instructor.from_openai(
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
        mode=instructor.Mode.JSON_SCHEMA
    )

    if state.get("draft_paper") and state.get("validation_errors"):
        prompt = f"""
        Fix the following errors in this existing question paper:
        {json.dumps(state['validation_errors'])}
        
        Existing Paper:
        {state['draft_paper'].model_dump_json()}
        
        Return the corrected full paper.
        """
    elif state.get("draft_paper") and state.get("hod_feedback"):
        prompt = f"""
        Revise this question paper according to the reviewer feedback:
        {state['hod_feedback']}
        
        Existing Paper:
        {state['draft_paper'].model_dump_json()}
        """
    else:
        prompt = f"""You are an expert university examiner creating a VTU CBCS question paper.
Subject: {state.get('subject')} ({state.get('subject_code')})
Exam Type: {state.get('exam_type')}
Duration: {state.get('duration_minutes')} minutes
Modules: {', '.join(str(m) for m in state.get('modules_to_cover', []))}

RULES:
1. Exactly 5 modules, each with 20 marks total.
2. Ensure proper Bloom's taxonomy tags.
3. Total paper = 100 marks.
"""
    
    try:
        generated_paper = client.chat.completions.create(
            model="llama3.1:8b",
            response_model=VTUQuestionPaper,
            messages=[{"role": "user", "content": prompt}],
            max_retries=3
        )
        return {
            "draft_paper": generated_paper,
            "validation_errors": [],
            "hod_feedback": None,
            "generation_attempt": attempt + 1,
            "status": "validating"
        }
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return {
            "generation_errors": [f"LLM error: {str(e)}"],
            "generation_attempt": attempt + 1,
            "status": "generating" # Loop back or fail out based on attempt counter
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 3: DISCRIMINATOR
# ═══════════════════════════════════════════════════════════════════════════════

def discriminator(state: PaperState) -> dict:
    """Fast, cheap sanity check to catch garbage hallucinations."""
    if not state.get('draft_paper'):
        return {"validation_errors": ["No paper generated"], "status": "failed_validation"}
        
    client = instructor.from_openai(
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
        mode=instructor.Mode.JSON_SCHEMA
    )
    
    try:
        verdict = client.chat.completions.create(
            model="llama3.2:1b",
            response_model=bool,
            messages=[{
                "role": "user",
                "content": f"""
                Is this a valid, sensible university examination question paper?
                Answer only True or False.
                
                {state['draft_paper'].model_dump_json()}
                """
            }]
        )
        if not verdict:
            return {"validation_errors": ["Paper failed sanity check by Discriminator"], "status": "failed_validation"}
    except Exception as e:
        logger.warning(f"Discriminator skipped due to error: {e}")
        
    return {"status": "validating"}

# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 4: VALIDATION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def validate_paper(state: PaperState) -> dict:
    """Extra validations beyond what Pydantic catches."""
    paper = state.get('draft_paper')
    if not paper:
        return {"validation_errors": ["No paper present"], "status": "failed_validation"}
        
    errors = []
    # (Pydantic catches most, here we just do basic content checks)
    for mod in paper.modules:
        for q in mod.questions:
            for part in q.parts:
                if len(part.text) < 20:
                    errors.append(f"Q{q.question_number}{part.part_letter}: Text too short ({len(part.text)} chars)")
    
    if errors:
        return {
            "validation_errors": errors,
            "status": "failed_validation"
        }
    
    return {
        "validation_errors": [],
        "status": "pending_review"
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 5: HOD REVIEW GATE
# ═══════════════════════════════════════════════════════════════════════════════

def human_review_gate(state: PaperState) -> dict:
    """
    Execution pauses *before* this node using LangGraph interrupts.
    """
    match state.get("review_verdict"):
        case "approve":
            return {"status": "approved"}
        case "revise":
            return {"status": "generating"}
        case "reject":
            return {"status": "rejected"}
            
    # Default if no verdict yet (it shouldn't reach here if resumed properly, but just in case)
    return {"status": "pending_review"}

# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 6: SAVE APPROVED
# ═══════════════════════════════════════════════════════════════════════════════

def save_approved_paper(state: PaperState) -> dict:
    """Saves approved paper to gold standard."""
    paper = state['draft_paper']
    paper_id = hashlib.md5(json.dumps(paper.model_dump()).encode()).hexdigest()
    
    import os
    os.makedirs("data", exist_ok=True)
    with open("data/gold_standard_papers.jsonl", "a") as f:
        f.write(json.dumps({
            "paper_id": paper_id,
            "paper": paper.model_dump(),
            "approved_at": datetime.now().isoformat()
        }) + "\n")
    
    return {"status": "archived"}

# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD THE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    workflow = StateGraph(PaperState)
    
    workflow.add_node("retrieve", retrieve_rich_context)
    workflow.add_node("generate", generate_draft)
    workflow.add_node("discriminator", discriminator)
    workflow.add_node("validate", validate_paper)
    workflow.add_node("review", human_review_gate)
    workflow.add_node("save_gold", save_approved_paper)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    
    def conditional_generate(s):
        if s.get("status") == "failed_circuit_breaker":
            return END
        elif s.get("status") == "generating":
            return "generate"  # if generation threw error but retries remain
        return "discriminator"

    workflow.add_conditional_edges("generate", conditional_generate)
    
    workflow.add_conditional_edges(
        "discriminator",
        lambda s: s.get("status"),
        {
            "failed_validation": "generate",
            "validating": "validate"
        }
    )
    
    workflow.add_conditional_edges(
        "validate",
        lambda s: s.get("status"),
        {
            "failed_validation": "generate",
            "pending_review": "review"
        }
    )
    
    workflow.add_conditional_edges(
        "review",
        lambda s: s.get("status"),
        {
            "generating": "generate",
            "approved": "save_gold",
            "rejected": END,
            "pending_review": "review" # Should not happen unless improperly resumed
        }
    )
    
    workflow.add_edge("save_gold", END)
    
    import os
    os.makedirs("data", exist_ok=True)
    memory = SqliteSaver.from_conn_string("data/langgraph_checkpoints.db")
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["review"]
    )
    
    return app
