from pydantic import BaseModel, Field
from typing import Literal

class TrainingPair(BaseModel):
    """A training example for the embedding model"""
    pair_id: str
    
    # Question or concept
    anchor: str = Field(..., min_length=10, description="The question or prompt")
    
    # Correct/related content
    positive: str = Field(..., min_length=20, description="Relevant answer or context")
    
    # Contrasting content (for negative sampling)
    negative: str | None = Field(None, description="Irrelevant or wrong content")
    
    # Metadata for filtering/weighting
    subject: str = Field(..., description="Academic subject")
    bloom_level: Literal["L1", "L2", "L3", "L4", "L5", "L6"] = Field(...)
    source: Literal["textbook", "question_paper", "hod_feedback", "synthetic"] = Field(...)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class TrainingDataset(BaseModel):
    """A collection of training pairs"""
    dataset_id: str
    pairs: list[TrainingPair]
    total_pairs: int
    created_at: str
    sources: dict[str, int]  # {source: count}
