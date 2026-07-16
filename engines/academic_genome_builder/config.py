"""
Configuration for academic_genome_builder.
"""
from pydantic import BaseModel

class EngineConfig(BaseModel):
    enabled: bool = True
