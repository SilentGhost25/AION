"""
Configuration for academic_tokenizer.
"""
from pydantic import BaseModel

class EngineConfig(BaseModel):
    enabled: bool = True
