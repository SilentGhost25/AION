"""
Configuration for question_generation_engine.
"""
from pydantic import BaseModel

class EngineConfig(BaseModel):
    enabled: bool = True
