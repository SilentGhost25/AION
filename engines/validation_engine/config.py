"""
Configuration for validation_engine.
"""
from pydantic import BaseModel

class EngineConfig(BaseModel):
    enabled: bool = True
