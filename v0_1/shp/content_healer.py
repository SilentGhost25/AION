"""
AION SHP Stage 3 — Content Healer & Safe Repair Policy
======================================================
Runs after extraction, before chunking.
Diagnoses and repairs content quality issues.

Safe Repair Policy Classifications:
1. ENCODING_REPAIR           -> SAFE
2. SYMBOL_NORMALIZATION      -> SAFE IF CONFIDENCE HIGH
3. EQUATION_RECONSTRUCTION   -> REQUIRES VALIDATION
4. SEMANTIC_GUESS            -> FORBIDDEN (Triggers clean HALT)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .error_knowledge import ErrorKnowledgeBase, Severity


THRESHOLD_STEPS = [0.70, 0.55, 0.40]
MAX_CHUNK_WORDS  = 500
TARGET_CHUNK     = 300
OVERLAP          = 30
MAX_MODULE_WORDS = 50_000


class RepairCategory(str, Enum):
    ENCODING_REPAIR = "ENCODING_REPAIR"
    SYMBOL_NORMALIZATION = "SYMBOL_NORMALIZATION"
    EQUATION_RECONSTRUCTION = "EQUIVALENT_RECONSTRUCTION"
    SEMANTIC_GUESS = "SEMANTIC_GUESS"


@dataclass
class HealedContent:
    chunks:          list[str]
    chunk_metas:     list[dict]
    thresholds_used: list[float]
    repairs:         list[str]
    accepted_rate:   float


class ContentHealer:
    """
    Stage 3: Repair content issues before LLM sees any text.
    All repairs are deterministic. No LLM calls.
    Enforces zero semantic guessing.
    """

    def __init__(self, kb: ErrorKnowledgeBase):
        self.kb = kb

    def heal(
        self,
        raw_text:   str,
        file_path:  str = "",
        chunk_size: int = TARGET_CHUNK,
        overlap:    int = OVERLAP,
    ) -> HealedContent:
        repairs = []

        word_count = len(raw_text.split())
        if word_count > MAX_MODULE_WORDS:
            rec = self.kb.record(
                "SH-021", "S3_HEAL",
                f"Module is {word_count} words (max {MAX_MODULE_WORDS})",
                Severity.WARNING,
                {"word_count": word_count},
            )
            raw_text = self._force_truncate(raw_text, MAX_MODULE_WORDS)
            repairs.append(f"SH-021 [{RepairCategory.ENCODING_REPAIR}]: Truncated {word_count}w → {MAX_MODULE_WORDS}w")
            self.kb.resolve(rec, "Truncated to max module size")

        from v0_1.cleaner import semantic_clean
        cleaned = semantic_clean(raw_text)
        if len(cleaned) < len(raw_text) * 0.9:
            repairs.append(f"SH-030 [{RepairCategory.ENCODING_REPAIR}]: Removed PDF artifacts")

        chunks = self._chunk(cleaned, chunk_size, overlap)

        oversized = [c for c in chunks if len(c.split()) > MAX_CHUNK_WORDS]
        if oversized:
            rec = self.kb.record(
                "SH-014", "S3_HEAL",
                f"{len(oversized)} chunks exceed {MAX_CHUNK_WORDS} words",
                Severity.WARNING,
            )
            chunks = self._split_oversized(chunks, TARGET_CHUNK, overlap)
            repairs.append(f"SH-014 [{RepairCategory.SYMBOL_NORMALIZATION}]: Split {len(oversized)} oversized chunks")
            self.kb.resolve(rec, "Chunks split to target size")

        valid_chunks, valid_metas, threshold_used, accepted_rate = (
            self._validate_adaptive(chunks)
        )

        if not valid_chunks:
            repairs.append("All chunks rejected even after threshold relaxation")
            print("[HEALER] Trigger: EXTRACTION_REJECTED")
            print("[HEALER] Strategy: ADAPTIVE_RELAXATION -> FAILED")
            print("[HEALER] Policy: SEMANTIC_GUESS_FORBIDDEN -> STOP")
            print("[HEALER] Result: BLOCKED")
        else:
            print(f"[HEALER] Trigger: HEALING_COMPLETE")
            print(f"[HEALER] Strategy: ADAPTIVE_THRESHOLD_{threshold_used}")
            print(f"[HEALER] Result: PASS (Accepted: {len(valid_chunks)}/{len(chunks)}, Rate: {accepted_rate:.2f})")

        return HealedContent(
            chunks          = valid_chunks,
            chunk_metas     = valid_metas,
            thresholds_used = [threshold_used],
            repairs         = repairs,
            accepted_rate   = accepted_rate,
        )

    def _chunk(self, text: str, size: int, overlap: int) -> list[str]:
        words  = text.split()
        chunks = []
        step   = max(1, size - overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i+size])
            if len(chunk.split()) >= 25:
                chunks.append(chunk)
        return chunks

    def _split_oversized(
        self, chunks: list[str], size: int, overlap: int
    ) -> list[str]:
        result = []
        for chunk in chunks:
            if len(chunk.split()) > MAX_CHUNK_WORDS:
                result.extend(self._chunk(chunk, size, overlap))
            else:
                result.append(chunk)
        return result

    def _force_truncate(self, text: str, max_words: int) -> str:
        return " ".join(text.split()[:max_words])

    def _validate_adaptive(
        self, chunks: list[str]
    ) -> tuple[list[str], list[dict], float, float]:
        from v0_1.validator import ContentValidator

        for threshold in THRESHOLD_STEPS:
            validator = ContentValidator(min_quality=threshold, min_academic=1)
            report    = validator.validate_batch(chunks)
            valid     = [
                s.clean_text or s.text
                for s in report.scores if s.passed
            ]

            if valid:
                rate = report.pass_rate
                if threshold < THRESHOLD_STEPS[0]:
                    rec = self.kb.record(
                        "SH-020", "S3_HEAL",
                        f"Used relaxed threshold {threshold} to accept {len(valid)} chunks",
                        Severity.WARNING,
                        {"threshold": threshold, "accepted": len(valid)},
                    )
                    self.kb.resolve(
                        rec, f"{len(valid)} chunks accepted at threshold {threshold}"
                    )
                print(f"[SHP-S3] {len(valid)}/{len(chunks)} chunks accepted "
                      f"at threshold {threshold}")
                return valid, [{"module": 1}] * len(valid), threshold, rate

            print(f"[SHP-S3] 0/{len(chunks)} at threshold {threshold} — relaxing")

            self.kb.record(
                "SH-020", "S3_HEAL",
                f"0 chunks at threshold {threshold}",
                Severity.WARNING,
                {"threshold": threshold, "total_chunks": len(chunks)},
            )

        self.kb.record(
            "SH-020", "S3_HEAL",
            "All chunks rejected at all thresholds",
            Severity.ERROR,
        )
        return [], [], THRESHOLD_STEPS[-1], 0.0
