"""
AION Error Knowledge Base
=========================
Every error has a structured record.
Every fix is deterministic and auditable.
No silent catch-all exception handlers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


# ── Recovery Actions ──────────────────────────────────────────────────────────

class RecoveryAction(str, Enum):
    RETRY            = "retry"
    LOWER_THRESHOLD  = "lower_threshold"
    SPLIT_CHUNKS     = "split_chunks"
    SWITCH_EXTRACTOR = "switch_extractor"
    DISABLE_STAGE    = "disable_stage"
    REDUCE_CONTEXT   = "reduce_context"
    REGENERATE       = "regenerate"
    REPAIR_MARKS     = "repair_marks"
    STOP             = "stop"
    DEGRADE          = "degrade"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR    = "ERROR"
    WARNING  = "WARNING"
    INFO     = "INFO"


# ── Error Record ──────────────────────────────────────────────────────────────

@dataclass
class ErrorRecord:
    """One instance of an observed error."""
    error_id:      str
    rule_id:       str
    stage:         str
    message:       str
    severity:      Severity
    timestamp:     str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    context:       dict = field(default_factory=dict)
    resolved:      bool  = False
    resolution:    str   = ""
    elapsed_ms:    float = 0.0


@dataclass
class HealingRule:
    """A deterministic recovery rule."""
    rule_id:     str
    trigger:     str
    action:      RecoveryAction
    max_retries: int = 1
    fallback:    RecoveryAction = RecoveryAction.STOP
    description: str = ""


# ── Rule Definitions ──────────────────────────────────────────────────────────

HEALING_RULES: dict[str, HealingRule] = {

    "SH-001": HealingRule(
        rule_id     = "SH-001",
        trigger     = "Two extractors running simultaneously",
        action      = RecoveryAction.SWITCH_EXTRACTOR,
        max_retries = 1,
        fallback    = RecoveryAction.DEGRADE,
        description = "Choose highest-confidence extractor. Disable all others.",
    ),

    "SH-010": HealingRule(
        rule_id     = "SH-010",
        trigger     = "Document.get() called on dataclass",
        action      = RecoveryAction.RETRY,
        max_retries = 1,
        fallback    = RecoveryAction.STOP,
        description = "Migration mismatch: use document.text instead of document.get('text').",
    ),

    "SH-014": HealingRule(
        rule_id     = "SH-014",
        trigger     = "Chunk size > 500 words",
        action      = RecoveryAction.SPLIT_CHUNKS,
        max_retries = 1,
        fallback    = RecoveryAction.DEGRADE,
        description = "Split oversized chunks into 250-word pieces with 30-word overlap.",
    ),

    "SH-020": HealingRule(
        rule_id     = "SH-020",
        trigger     = "0 chunks accepted after validation",
        action      = RecoveryAction.LOWER_THRESHOLD,
        max_retries = 2,
        fallback    = RecoveryAction.STOP,
        description = "Relax academic threshold 0.70 → 0.55 → 0.40. Retry validation.",
    ),

    "SH-021": HealingRule(
        rule_id     = "SH-021",
        trigger     = "Module > 50000 words (segmentation failure)",
        action      = RecoveryAction.SPLIT_CHUNKS,
        max_retries = 1,
        fallback    = RecoveryAction.DEGRADE,
        description = "Force re-segmentation into 3000-word blocks.",
    ),

    "SH-030": HealingRule(
        rule_id     = "SH-030",
        trigger     = "PDF artifacts in extracted text",
        action      = RecoveryAction.RETRY,
        max_retries = 1,
        fallback    = RecoveryAction.DISABLE_STAGE,
        description = "Run PDF artifact filter on extracted text. Re-validate.",
    ),

    "SH-032": HealingRule(
        rule_id     = "SH-032",
        trigger     = "Question references PDF/xref/stream keywords",
        action      = RecoveryAction.REGENERATE,
        max_retries = 1,
        fallback    = RecoveryAction.STOP,
        description = "Regenerate question with stricter academic-only prompt.",
    ),

    "SH-041": HealingRule(
        rule_id     = "SH-041",
        trigger     = "Sub-question marks do not sum to question total",
        action      = RecoveryAction.REPAIR_MARKS,
        max_retries = 1,
        fallback    = RecoveryAction.STOP,
        description = "Redistribute marks from template. Never from LLM.",
    ),

    "SH-050": HealingRule(
        rule_id     = "SH-050",
        trigger     = "LLM timeout",
        action      = RecoveryAction.REDUCE_CONTEXT,
        max_retries = 2,
        fallback    = RecoveryAction.DEGRADE,
        description = "Halve context size. Retry. If still timeout, use fallback template.",
    ),

    "SH-060": HealingRule(
        rule_id     = "SH-060",
        trigger     = "Ollama not running",
        action      = RecoveryAction.RETRY,
        max_retries = 2,
        fallback    = RecoveryAction.STOP,
        description = "Attempt to start ollama serve. Wait 8 seconds. Retry health check.",
    ),

    "SH-070": HealingRule(
        rule_id     = "SH-070",
        trigger     = "Model not found in Ollama",
        action      = RecoveryAction.SWITCH_EXTRACTOR,
        max_retries = 1,
        fallback    = RecoveryAction.STOP,
        description = "Fall back to next available model in device profile.",
    ),

    "SH-080": HealingRule(
        rule_id     = "SH-080",
        trigger     = "Extraction word count increased 30%+ on second run",
        action      = RecoveryAction.DISABLE_STAGE,
        max_retries = 0,
        fallback    = RecoveryAction.DEGRADE,
        description = "Duplicate extraction detected. Disable second extractor. Use first result.",
    ),
}


class ErrorKnowledgeBase:
    """
    Stores error records and resolves healing rules.
    Keeps a session log for audit.
    """

    def __init__(self):
        self._records: list[ErrorRecord] = []
        self._counters: dict[str, int]  = {}

    def record(
        self,
        rule_id:  str,
        stage:    str,
        message:  str,
        severity: Severity = Severity.ERROR,
        context:  dict     = None,
    ) -> ErrorRecord:
        rec = ErrorRecord(
            error_id = f"ERR-{len(self._records)+1:04d}",
            rule_id  = rule_id,
            stage    = stage,
            message  = message,
            severity = severity,
            context  = context or {},
        )
        self._records.append(rec)
        self._counters[rule_id] = self._counters.get(rule_id, 0) + 1
        print(f"[SHP-KB] {rec.error_id} [{rule_id}] {severity.value}: {message}")
        return rec

    def resolve(self, record: ErrorRecord, resolution: str) -> None:
        record.resolved   = True
        record.resolution = resolution
        print(f"[SHP-KB] {record.error_id} RESOLVED: {resolution}")

    def get_rule(self, rule_id: str) -> Optional[HealingRule]:
        return HEALING_RULES.get(rule_id)

    def fire_count(self, rule_id: str) -> int:
        return self._counters.get(rule_id, 0)

    def session_log(self) -> list[dict]:
        return [
            {
                "error_id":   r.error_id,
                "rule_id":    r.rule_id,
                "stage":      r.stage,
                "severity":   r.severity.value,
                "message":    r.message,
                "resolved":   r.resolved,
                "resolution": r.resolution,
            }
            for r in self._records
        ]

    def has_critical(self) -> bool:
        return any(
            r.severity == Severity.CRITICAL and not r.resolved
            for r in self._records
        )
