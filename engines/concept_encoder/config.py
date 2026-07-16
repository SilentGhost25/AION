"""
Configuration for concept_encoder.
"""
from pydantic import BaseModel

class EngineConfig(BaseModel):
    enabled: bool = True
