"""
Configuration for answer_graph_builder.
"""
from pydantic import BaseModel

class EngineConfig(BaseModel):
    enabled: bool = True
