"""
AION Core Extraction — Chunk Validation Reporter
=================================================
Produces detailed diagnostic breakdown reports replacing single acceptance counts.
Categorizes valid, suspicious, recoverable, quarantined, and invalid chunks by rejection reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .contracts import ChunkStatus, EvidenceChunk, RejectionReason


@dataclass
class ChunkValidationReport:
    total_chunks        : int = 0
    valid_chunks        : int = 0
    suspicious_chunks   : int = 0
    recoverable_chunks  : int = 0
    quarantined_chunks  : int = 0
    invalid_chunks      : int = 0

    rejection_breakdown : Dict[RejectionReason, int] = field(default_factory=dict)
    per_module_coverage : Dict[int, int] = field(default_factory=dict)

    primary_root_cause   : Optional[RejectionReason] = None
    secondary_root_cause : Optional[RejectionReason] = None

    recovery_attempted  : bool = False
    recovered_count     : int = 0

    hard_stop_triggered : bool = False
    hard_stop_reason    : str = ""
    recommended_action  : str = ""

    def get_retrieval_eligible_count(self) -> int:
        return self.valid_chunks + self.suspicious_chunks + self.recoverable_chunks

    @classmethod
    def from_chunks(cls, chunks: List[EvidenceChunk]) -> ChunkValidationReport:
        report = cls(total_chunks=len(chunks))
        breakdown: Dict[RejectionReason, int] = {}
        module_counts: Dict[int, int] = {}

        for c in chunks:
            if c.status == ChunkStatus.VALID:
                report.valid_chunks += 1
            elif c.status == ChunkStatus.SUSPICIOUS:
                report.suspicious_chunks += 1
            elif c.status == ChunkStatus.RECOVERABLE:
                report.recoverable_chunks += 1
            elif c.status == ChunkStatus.QUARANTINED:
                report.quarantined_chunks += 1
            elif c.status == ChunkStatus.INVALID:
                report.invalid_chunks += 1

            if c.is_retrieval_eligible():
                mod_id = int(c.module_id) if c.module_id and str(c.module_id).isdigit() else 1
                module_counts[mod_id] = module_counts.get(mod_id, 0) + 1

            for reason in c.rejection_reasons:
                breakdown[reason] = breakdown.get(reason, 0) + 1

        report.rejection_breakdown = breakdown
        report.per_module_coverage = module_counts

        # Primary root cause determination
        if breakdown:
            sorted_reasons = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
            report.primary_root_cause = sorted_reasons[0][0]
            if len(sorted_reasons) > 1:
                report.secondary_root_cause = sorted_reasons[1][0]

        return report

    def format_log_report(self) -> str:
        eligible = self.get_retrieval_eligible_count()
        lines = [
            "════════════════════════════════════════════════",
            "EXTRACTION QA REPORT",
            "════════════════════════════════════════════════",
            f"Total chunks              : {self.total_chunks}",
            f"Valid                     : {self.valid_chunks}",
            f"Suspicious (penalized)    : {self.suspicious_chunks}",
            f"Recoverable (healed)      : {self.recoverable_chunks}",
            f"Quarantined               : {self.quarantined_chunks}",
            f"Invalid (excluded)        : {self.invalid_chunks}",
            f"Retrieval Eligible        : {eligible}",
            "────────────────────────────────────────────────",
            "Rejection Breakdown:",
        ]

        if not self.rejection_breakdown:
            lines.append("  (No rejections recorded)")
        else:
            tot = max(self.total_chunks, 1)
            for reason, cnt in self.rejection_breakdown.items():
                pct = (cnt / tot) * 100.0
                lines.append(f"  {reason.value:<24} : {cnt:>4} ({pct:>5.1f}%)")

        lines.extend([
            "────────────────────────────────────────────────",
            "Per-Module Coverage:",
        ])
        for mod in range(1, 6):
            cnt = self.per_module_coverage.get(mod, 0)
            status_tag = "" if cnt >= 5 else "  ← INSUFFICIENT"
            lines.append(f"  Module {mod} : {cnt} valid chunks{status_tag}")

        lines.extend([
            "────────────────────────────────────────────────",
            f"Primary Root Cause : {self.primary_root_cause.value if self.primary_root_cause else 'NONE'}",
            f"Hard Stop Triggered: {'YES' if self.hard_stop_triggered else 'NO'}",
        ])
        if self.hard_stop_triggered:
            lines.append(f"Reason             : {self.hard_stop_reason}")
            lines.append(f"Recommended Action : {self.recommended_action}")

        lines.append("════════════════════════════════════════════════")
        return "\n".join(lines)
