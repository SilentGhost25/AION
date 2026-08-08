"""
AION Self-Healing Pipeline (SHP) Package
=========================================
Provides deterministic recovery and quality gates across all pipeline stages.
"""

from .error_knowledge import ErrorKnowledgeBase, ErrorRecord, Severity, HealingRule, RecoveryAction
from .health_monitor import SystemHealthMonitor, HealthStatus
from .file_diagnostics import FileDiagnostics, FileProfile
from .pipeline_planner import PipelinePlanner, ExtractionPlan
from .content_healer import ContentHealer, HealedContent
from .retrieval_healer import RetrievalHealer, RetrievalResult
from .output_repair import OutputRepair
from .shp_pipeline import SHPPipeline, SHPPipelineResult

__all__ = [
    "ErrorKnowledgeBase",
    "ErrorRecord",
    "Severity",
    "HealingRule",
    "RecoveryAction",
    "SystemHealthMonitor",
    "HealthStatus",
    "FileDiagnostics",
    "FileProfile",
    "PipelinePlanner",
    "ExtractionPlan",
    "ContentHealer",
    "HealedContent",
    "RetrievalHealer",
    "RetrievalResult",
    "OutputRepair",
    "SHPPipeline",
    "SHPPipelineResult",
]
