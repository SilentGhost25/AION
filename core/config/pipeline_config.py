"""
AION Pipeline Configuration
===========================
Defines default quality thresholds and operational feature flags for AION v2.1 PC.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    min_extraction_confidence: float = float(os.environ.get("AION_MIN_EXTRACTION_CONFIDENCE", "0.70"))
    min_grounding_score: float = float(os.environ.get("AION_MIN_GROUNDING_SCORE", "0.80"))
    strict_validation: bool = os.environ.get("AION_STRICT_VALIDATION", "true").lower() == "true"
    strict_grounding: bool = os.environ.get("AION_STRICT_GROUNDING", "true").lower() == "true"
    strict_rendering: bool = os.environ.get("AION_STRICT_RENDERING", "true").lower() == "true"
    use_unified: bool = os.environ.get("AION_USE_UNIFIED", "true").lower() == "true"
    vre_enabled: bool = os.environ.get("AION_VRE_ENABLED", "true").lower() == "true"
    mke_enabled: bool = os.environ.get("AION_MKE_ENABLED", "true").lower() == "true"
    auto_heal: bool = os.environ.get("AION_AUTO_HEAL", "true").lower() == "true"


PIPELINE_CONFIG = PipelineConfig()
