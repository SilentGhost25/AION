"""
Configuration for academic_selfrag_engine.
"""
from pydantic import BaseModel

class EngineConfig(BaseModel):
    enabled: bool = True
