"""
AION v2 Contracts Module
========================
Strict stage contracts governing data flow between pipeline stages.
"""

from .request_contract import GenerationRequest
from .pipeline_trace import PipelineTrace

__all__ = ["GenerationRequest", "PipelineTrace"]
