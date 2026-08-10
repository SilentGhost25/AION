"""
AION Core Integrity Package
===========================
Provides multi-signal encoding analysis, prompt injection detection,
evidence quarantine & healing, equation validation, and safe byte decoding.
"""

from .encoding_gate import CorruptionReport, EncodingGate, CORRUPTION_THRESHOLDS
from .prompt_safety_gate import SafetyReport, PromptSafetyGate, PROMPT_INJECTION_PATTERNS
from .quarantine import QuarantineState, QuarantineDecision, EvidenceQuarantineLayer, QuarantineHealer
from .equation_gate import EquationReport, EquationIntegrityGate
from .safe_decoder import SafeDecoder

__all__ = [
    "CorruptionReport",
    "EncodingGate",
    "CORRUPTION_THRESHOLDS",
    "SafetyReport",
    "PromptSafetyGate",
    "PROMPT_INJECTION_PATTERNS",
    "QuarantineState",
    "QuarantineDecision",
    "EvidenceQuarantineLayer",
    "QuarantineHealer",
    "EquationReport",
    "EquationIntegrityGate",
    "SafeDecoder",
]
