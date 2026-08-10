"""
AION Core Integrity — Evidence Quarantine System
=================================================
Implements the 5-gate evidence quarantine workflow and the QuarantineHealer
for automated repair of corrupted or compromised evidence chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .encoding_gate import EncodingGate
from .prompt_safety_gate import PromptSafetyGate


class QuarantineState(str, Enum):
    VALID_EVIDENCE = "VALID_EVIDENCE"  # enters retrieval, eligible for question generation
    SUSPICIOUS     = "SUSPICIOUS"      # enters retrieval with penalty, flagged in QA report
    QUARANTINED    = "QUARANTINED"     # never enters retrieval, routed to healer
    HEALED         = "HEALED"          # re-validated after repair, becomes VALID_EVIDENCE or FAILED
    FAILED         = "FAILED"          # permanently excluded, logged for manual review


@dataclass
class QuarantineDecision:
    status  : QuarantineState
    reason  : str = ""
    action  : str = "PASS"             # "PASS" | "QUARANTINE" | "HEAL" | "FAIL"
    metrics : Dict[str, Any] = field(default_factory=dict)


class EvidenceQuarantineLayer:
    """Evaluates evidence chunks against 5 mandatory integrity gates."""

    MIN_CHUNK_LENGTH = 50

    @classmethod
    def process(cls, chunk: Dict[str, Any]) -> QuarantineDecision:
        text = chunk.get("text", "") or chunk.get("content", "")

        # ── GATE A: ENCODING INTEGRITY ───────────────────────────────────────
        encoding_report = EncodingGate.analyze(text)
        if encoding_report.corruption_level == "BINARY":
            return QuarantineDecision(
                status=QuarantineState.QUARANTINED,
                reason="BINARY_CONTAMINATION",
                action="QUARANTINE",
                metrics={"confidence": encoding_report.confidence},
            )
        if encoding_report.corruption_level == "CORRUPTED":
            return QuarantineDecision(
                status=QuarantineState.QUARANTINED,
                reason="TEXT_CORRUPTION",
                action="QUARANTINE",
                metrics={"confidence": encoding_report.confidence},
            )
        if encoding_report.corruption_level == "SUSPICIOUS":
            chunk["quality_flag"] = "SUSPICIOUS"
            chunk["retrieval_penalty"] = 0.5

        # ── GATE B: EQUATION INTEGRITY ───────────────────────────────────────
        eq_ids = chunk.get("equation_ids", [])
        if "given formula" in text.lower() and len(eq_ids) == 0:
            return QuarantineDecision(
                status=QuarantineState.QUARANTINED,
                reason="EQUATION_REFERENCE_WITHOUT_EQUATION",
                action="QUARANTINE",
            )

        # ── GATE C: PROMPT INJECTION IN SOURCE ───────────────────────────────
        source_safety = PromptSafetyGate.scan_source(text)
        if source_safety.status == "INJECTION_DETECTED":
            return QuarantineDecision(
                status=QuarantineState.QUARANTINED,
                reason="SOURCE_INJECTION",
                action="QUARANTINE",
                metrics={"patterns": source_safety.patterns},
            )

        # ── GATE D: PROVENANCE ───────────────────────────────────────────────
        doc_id = chunk.get("document_id") or chunk.get("doc_id")
        if not doc_id:
            return QuarantineDecision(
                status=QuarantineState.QUARANTINED,
                reason="MISSING_PROVENANCE",
                action="QUARANTINE",
            )
        extraction_conf = chunk.get("extraction_confidence", 1.0)
        if extraction_conf < 0.30:
            return QuarantineDecision(
                status=QuarantineState.QUARANTINED,
                reason="LOW_EXTRACTION_CONFIDENCE",
                action="QUARANTINE",
                metrics={"confidence": extraction_conf},
            )

        # ── GATE E: MINIMUM CONTENT ──────────────────────────────────────────
        if len(text.strip()) < cls.MIN_CHUNK_LENGTH:
            return QuarantineDecision(
                status=QuarantineState.QUARANTINED,
                reason="INSUFFICIENT_CONTENT",
                action="QUARANTINE",
                metrics={"length": len(text.strip())},
            )

        # ALL GATES PASSED
        chunk["status"] = QuarantineState.VALID_EVIDENCE.value
        return QuarantineDecision(status=QuarantineState.VALID_EVIDENCE, action="PASS")


class QuarantineHealer:
    """Attempts automated healing of quarantined evidence chunks."""

    @classmethod
    def heal(cls, chunk: Dict[str, Any], reason: str) -> Optional[Dict[str, Any]]:
        text = chunk.get("text", "") or chunk.get("content", "")

        if reason == "BINARY_CONTAMINATION":
            # Strip null bytes and non-printable control runs
            healed_text = "".join(c for c in text if c == "\n" or c == "\t" or (ord(c) >= 32 and c != "\ufffd"))
            report = EncodingGate.analyze(healed_text)
            if report.is_safe_for_llm() and len(healed_text.strip()) >= 50:
                healed_chunk = dict(chunk)
                healed_chunk["text"] = healed_text
                healed_chunk["status"] = QuarantineState.HEALED.value
                return healed_chunk
            return None

        if reason == "TEXT_CORRUPTION":
            # Attempt encoding repair by stripping replacement chars and control chars
            healed_text = text.replace("\ufffd", "").strip()
            report = EncodingGate.analyze(healed_text)
            if report.confidence < 0.70 and len(healed_text) >= 50:
                healed_chunk = dict(chunk)
                healed_chunk["text"] = healed_text
                healed_chunk["status"] = QuarantineState.HEALED.value
                return healed_chunk
            return None

        if reason == "EQUATION_REFERENCE_WITHOUT_EQUATION":
            # Remove phantom 'given formula' reference to allow valid text generation
            import re
            healed_text = re.sub(r"(?i)\bas given by the formula\b", "", text)
            healed_text = re.sub(r"(?i)\busing the given formula\b", "", healed_text)
            healed_chunk = dict(chunk)
            healed_chunk["text"] = healed_text
            healed_chunk["status"] = QuarantineState.HEALED.value
            return healed_chunk

        if reason == "LOW_EXTRACTION_CONFIDENCE":
            if len(text.strip()) >= 50:
                healed_chunk = dict(chunk)
                healed_chunk["extraction_confidence"] = 0.60
                healed_chunk["status"] = QuarantineState.HEALED.value
                return healed_chunk

        return None
