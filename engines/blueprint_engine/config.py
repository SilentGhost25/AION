"""
Configuration for blueprint_engine.
"""
from pydantic import BaseModel

class EngineConfig(BaseModel):
    enabled: bool = True
