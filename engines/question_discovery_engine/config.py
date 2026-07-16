"""
Configuration for question_discovery_engine.
"""
from pydantic import BaseModel

class EngineConfig(BaseModel):
    enabled: bool = True
