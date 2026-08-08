"""
AION Execution Auditor
======================
Sits alongside PipelineTrace.
Verifies that every stage obeyed its contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from .contracts import (
    RawFile, ExtractionResult, CleanedContent, ChunkedContent,
    RetrievedEvidence, GenerationRequest, PaperDraft, FinalPaper,
    ValidationVerdict, PipelineHealth
)


class AuditVerdict(str, Enum):
    COMPLIANT = "COMPLIANT"
    DEGRADED  = "DEGRADED"
    VIOLATED  = "VIOLATED"
    BYPASSED  = "BYPASSED"


@dataclass
class StageAudit:
    stage_name:     str
    verdict:        AuditVerdict
    contract_in:    str
    contract_out:   str
    confidence:     float
    fallbacks_used: list[str] = field(default_factory=list)
    violations:     list[str] = field(default_factory=list)
    notes:          list[str] = field(default_factory=list)
    elapsed_ms:     float     = 0.0
    timestamp:      str       = field(
        default_factory=lambda: datetime.now().isoformat()
    )


@dataclass
class AuditReport:
    doc_id:             str
    pipeline_path:      str
    stage_audits:       list[StageAudit] = field(default_factory=list)
    overall_verdict:    AuditVerdict = AuditVerdict.COMPLIANT
    overall_confidence: float        = 1.0
    trust_score:        int          = 100
    exportable:         bool         = True
    blocking_issues:    list[str]    = field(default_factory=list)

    def add(self, audit: StageAudit):
        self.stage_audits.append(audit)
        self._recompute()

    def _recompute(self):
        if not self.stage_audits:
            return

        verdicts = [a.verdict for a in self.stage_audits]
        if AuditVerdict.VIOLATED in verdicts:
            self.overall_verdict = AuditVerdict.VIOLATED
        elif AuditVerdict.BYPASSED in verdicts:
            self.overall_verdict = AuditVerdict.BYPASSED
        elif AuditVerdict.DEGRADED in verdicts:
            self.overall_verdict = AuditVerdict.DEGRADED
        else:
            self.overall_verdict = AuditVerdict.COMPLIANT

        import math
        confs = [a.confidence for a in self.stage_audits if a.confidence > 0]
        if confs:
            self.overall_confidence = round(
                math.exp(sum(math.log(c) for c in confs) / len(confs)), 4
            )

        violations = sum(1 for a in self.stage_audits if a.verdict == AuditVerdict.VIOLATED)
        bypasses   = sum(1 for a in self.stage_audits if a.verdict == AuditVerdict.BYPASSED)
        degraded   = sum(1 for a in self.stage_audits if a.verdict == AuditVerdict.DEGRADED)
        self.trust_score = max(0, 100 - violations * 20 - bypasses * 15 - degraded * 8)
        self.exportable  = self.trust_score >= 40

    def to_dict(self) -> dict:
        return {
            "doc_id":             self.doc_id,
            "pipeline_path":      self.pipeline_path,
            "overall_verdict":    self.overall_verdict.value,
            "overall_confidence": self.overall_confidence,
            "trust_score":        self.trust_score,
            "exportable":         self.exportable,
            "blocking_issues":    self.blocking_issues,
            "stages": [
                {
                    "stage":      a.stage_name,
                    "verdict":    a.verdict.value,
                    "confidence": a.confidence,
                    "fallbacks":  a.fallbacks_used,
                    "violations": a.violations,
                }
                for a in self.stage_audits
            ],
        }

    def print_summary(self):
        status = "✓" if self.exportable else "✗"
        print(f"\n[AUDITOR] {status} Trust={self.trust_score}/100 | "
              f"Confidence={self.overall_confidence:.0%} | "
              f"Verdict={self.overall_verdict.value}")
        for a in self.stage_audits:
            icon = {"COMPLIANT": "✓", "DEGRADED": "⚠", "VIOLATED": "✗", "BYPASSED": "○"}
            print(f"  {icon.get(a.verdict.value, '?')} {a.stage_name:<25} "
                  f"conf={a.confidence:.0%} {a.verdict.value}")
            for v in a.violations:
                print(f"      ✗ {v}")
            for f in a.fallbacks_used:
                print(f"      ⚠ {f}")


class ExecutionAuditor:
    """
    Verifies contract compliance at every pipeline stage.
    """

    def __init__(self, doc_id: str):
        self.report = AuditReport(
            doc_id        = doc_id,
            pipeline_path = "unknown",
        )
        self._stage_order: list[str] = []
        self._cleaner_ran  = False

    def audit_extraction(
        self,
        result:     Optional[ExtractionResult],
        pipeline:   str,
        elapsed_ms: float = 0,
    ) -> StageAudit:
        audit = StageAudit(
            stage_name   = "S1_EXTRACTION",
            verdict      = AuditVerdict.COMPLIANT,
            contract_in  = "RawFile",
            contract_out = "ExtractionResult",
            confidence   = 0.0,
            elapsed_ms   = elapsed_ms,
        )

        if result is None:
            audit.verdict = AuditVerdict.VIOLATED
            audit.violations.append("Extraction returned None")
            audit.confidence = 0.0
        else:
            if not isinstance(result, ExtractionResult):
                audit.verdict = AuditVerdict.VIOLATED
                audit.violations.append(
                    f"Wrong contract: expected ExtractionResult, got {type(result).__name__}"
                )
            else:
                audit.confidence = result.confidence
                if result.confidence < 0.6:
                    audit.verdict = AuditVerdict.DEGRADED
                    audit.fallbacks_used.append(
                        f"Low extraction confidence: {result.confidence:.0%}"
                    )

        if pipeline in ("legacy", "fallback"):
            audit.verdict = AuditVerdict.DEGRADED
            audit.fallbacks_used.append(f"Used legacy pipeline: {pipeline}")

        self._stage_order.append("S1_EXTRACTION")
        self.report.add(audit)
        return audit

    def audit_cleaning(
        self,
        result:               Optional[CleanedContent],
        ran_before_validator: bool = True,
        elapsed_ms:           float = 0,
    ) -> StageAudit:
        audit = StageAudit(
            stage_name   = "S2_CLEANING",
            verdict      = AuditVerdict.COMPLIANT,
            contract_in  = "ExtractionResult",
            contract_out = "CleanedContent",
            confidence   = 0.0,
            elapsed_ms   = elapsed_ms,
        )

        self._cleaner_ran = result is not None

        if result is None:
            audit.verdict = AuditVerdict.BYPASSED
            audit.violations.append("Cleaner was not executed")
            audit.confidence = 0.0
        else:
            if not ran_before_validator:
                audit.verdict = AuditVerdict.VIOLATED
                audit.violations.append(
                    "CRITICAL: Cleaner ran AFTER validator."
                )
            else:
                audit.confidence = result.retention_rate
                if result.retention_rate < 0.5:
                    audit.verdict = AuditVerdict.DEGRADED
                    audit.fallbacks_used.append(
                        f"Heavy cleaning: only {result.retention_rate:.0%} text retained"
                    )
                else:
                    audit.confidence = min(1.0, result.retention_rate + 0.1)

        self._stage_order.append("S2_CLEANING")
        self.report.add(audit)
        return audit

    def audit_chunking(
        self,
        result:     Optional[ChunkedContent],
        elapsed_ms: float = 0,
    ) -> StageAudit:
        audit = StageAudit(
            stage_name   = "S3_CHUNKING",
            verdict      = AuditVerdict.COMPLIANT,
            contract_in  = "CleanedContent",
            contract_out = "ChunkedContent",
            confidence   = 0.0,
            elapsed_ms   = elapsed_ms,
        )

        if not self._cleaner_ran:
            audit.verdict = AuditVerdict.VIOLATED
            audit.violations.append("Chunker received uncleaned content.")

        if result is None:
            audit.verdict = AuditVerdict.VIOLATED
            audit.violations.append("Chunking returned None")
        else:
            total  = result.total_chunks
            thresh = result.threshold_used
            audit.confidence = min(1.0, total / 10)

            if total == 0:
                audit.verdict = AuditVerdict.VIOLATED
                audit.violations.append("Zero chunks produced.")
                self.report.blocking_issues.append("ZERO_CHUNKS")
            elif thresh < 0.60:
                audit.verdict = AuditVerdict.DEGRADED
                audit.fallbacks_used.append(
                    f"Validator threshold relaxed to {thresh:.0%}"
                )
            elif total < 5:
                audit.verdict = AuditVerdict.DEGRADED
                audit.fallbacks_used.append(f"Only {total} chunks — very thin content")

        self._stage_order.append("S3_CHUNKING")
        self.report.add(audit)
        return audit

    def audit_retrieval(
        self,
        result:        Optional[RetrievedEvidence],
        used_fallback: bool = False,
        fallback_desc: str  = "",
        elapsed_ms:    float = 0,
    ) -> StageAudit:
        audit = StageAudit(
            stage_name   = "S4_RETRIEVAL",
            verdict      = AuditVerdict.COMPLIANT,
            contract_in  = "ChunkedContent",
            contract_out = "RetrievedEvidence",
            confidence   = 0.0,
            elapsed_ms   = elapsed_ms,
        )

        if result is None:
            audit.verdict = AuditVerdict.VIOLATED
            audit.violations.append("Retrieval returned None — no evidence for generation")
            self.report.blocking_issues.append("NO_EVIDENCE")
        else:
            n_modules = len(result.evidence_by_module)
            scores    = [e.evidence_score for e in result.evidence_by_module.values()]
            avg_score = sum(scores) / max(1, len(scores))
            audit.confidence = avg_score

            if n_modules == 0:
                audit.verdict = AuditVerdict.VIOLATED
                audit.violations.append("Retrieval returned zero modules")
                self.report.blocking_issues.append("NO_EVIDENCE")
            elif used_fallback:
                audit.verdict = AuditVerdict.DEGRADED
                audit.fallbacks_used.append(
                    fallback_desc or "Retrieval used fallback text"
                )
            elif avg_score < 0.5:
                audit.verdict = AuditVerdict.DEGRADED
                audit.fallbacks_used.append(
                    f"Low evidence quality: avg_score={avg_score:.0%}"
                )

        self._stage_order.append("S4_RETRIEVAL")
        self.report.add(audit)
        return audit

    def audit_generation(
        self,
        n_questions_requested: int,
        n_questions_produced:  int,
        n_fallbacks:           int,
        elapsed_ms:            float = 0,
    ) -> StageAudit:
        audit = StageAudit(
            stage_name   = "S6_GENERATION",
            verdict      = AuditVerdict.COMPLIANT,
            contract_in  = "GenerationRequest",
            contract_out = "GeneratedQuestion",
            confidence   = 0.0,
            elapsed_ms   = elapsed_ms,
        )

        if n_questions_produced == 0:
            audit.verdict = AuditVerdict.VIOLATED
            audit.violations.append("LLM produced zero questions")
            self.report.blocking_issues.append("NO_QUESTIONS")
        else:
            fill_rate = n_questions_produced / max(1, n_questions_requested)
            audit.confidence = fill_rate
            if n_fallbacks > 0:
                audit.verdict = AuditVerdict.DEGRADED
                audit.fallbacks_used.append(
                    f"{n_fallbacks} questions used fallback templates"
                )
            if fill_rate < 0.8:
                audit.verdict = AuditVerdict.DEGRADED
                audit.fallbacks_used.append(
                    f"Only {fill_rate:.0%} of requested questions generated"
                )

        self._stage_order.append("S6_GENERATION")
        self.report.add(audit)
        return audit

    def audit_critic(
        self,
        n_validated: int,
        n_passed:    int,
        n_repaired:  int,
        n_failed:    int,
        elapsed_ms:  float = 0,
    ) -> StageAudit:
        audit = StageAudit(
            stage_name   = "S7_CRITIC",
            verdict      = AuditVerdict.COMPLIANT,
            contract_in  = "GeneratedQuestion",
            contract_out = "ValidatedQuestion",
            confidence   = 0.0,
            elapsed_ms   = elapsed_ms,
        )

        if n_validated == 0:
            audit.verdict = AuditVerdict.BYPASSED
            audit.violations.append("Critic ran on zero questions")
        else:
            pass_rate = (n_passed + n_repaired) / n_validated
            audit.confidence = pass_rate

            if n_failed == n_validated:
                audit.verdict = AuditVerdict.VIOLATED
                audit.violations.append(
                    f"Critic rejected ALL {n_failed} questions."
                )
                self.report.blocking_issues.append("ALL_QUESTIONS_REJECTED")
            elif pass_rate < 0.5:
                audit.verdict = AuditVerdict.DEGRADED
                audit.fallbacks_used.append(
                    f"Only {pass_rate:.0%} of questions passed critic"
                )

        self._stage_order.append("S7_CRITIC")
        self.report.add(audit)
        return audit

    def audit_final_paper(
        self,
        paper: Optional[FinalPaper],
    ) -> StageAudit:
        audit = StageAudit(
            stage_name   = "S_FINAL",
            verdict      = AuditVerdict.COMPLIANT,
            contract_in  = "PaperDraft",
            contract_out = "FinalPaper",
            confidence   = 0.0,
        )

        if paper is None:
            audit.verdict = AuditVerdict.VIOLATED
            audit.violations.append("Final paper is None — pipeline aborted")
        else:
            audit.confidence = paper.qa_score / 100
            if not paper.exportable:
                audit.verdict = AuditVerdict.VIOLATED
                audit.violations.append(
                    f"Paper health score {paper.health.score}/100 < 40."
                )
                self.report.blocking_issues.append(
                    f"PAPER_NOT_EXPORTABLE (health={paper.health.score})"
                )
            elif paper.qa_score < 60:
                audit.verdict = AuditVerdict.DEGRADED
                audit.fallbacks_used.append(
                    f"Low QA score: {paper.qa_score}/100"
                )

        unified_stages = {
            "S1_EXTRACTION", "S2_CLEANING", "S3_CHUNKING",
            "S4_RETRIEVAL", "S6_GENERATION", "S7_CRITIC"
        }
        ran = set(self._stage_order)
        missing = unified_stages - ran
        if missing:
            self.report.pipeline_path = "incomplete"
            audit.violations.append(f"Stages not executed: {missing}")
        else:
            self.report.pipeline_path = "unified"

        self._stage_order.append("S_FINAL")
        self.report.add(audit)
        self.report.print_summary()
        return audit

    def is_safe_to_generate(self) -> tuple[bool, str]:
        if "ZERO_CHUNKS" in self.report.blocking_issues:
            return False, "Zero valid chunks. No academic content available."
        if "NO_EVIDENCE" in self.report.blocking_issues:
            return False, "Retrieval produced no grounded evidence."
        if self.report.overall_verdict == AuditVerdict.VIOLATED:
            violated = [
                a.stage_name for a in self.report.stage_audits
                if a.verdict == AuditVerdict.VIOLATED
            ]
            return False, f"Contract violations at stages: {violated}"
        return True, "Pipeline verified. Safe to generate."

    def is_safe_to_export(self) -> tuple[bool, str]:
        if self.report.blocking_issues:
            return False, f"Blocking issues: {self.report.blocking_issues}"
        if not self.report.exportable:
            return False, f"Trust score {self.report.trust_score}/100 below threshold."
        return True, "Paper verified. Safe to export."
