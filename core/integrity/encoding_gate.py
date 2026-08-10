"""
AION Core Integrity — Encoding Gate
====================================
Multi-signal corruption analysis for raw extracted text.
Identifies binary contamination, control characters, replacement character rates,
Shannon entropy anomalies, and non-printable binary runs.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List


@dataclass
class CorruptionReport:
    """
    Multi-signal corruption analysis.
    A single threshold is insufficient — uses all signals together.
    """
    text_length          : int
    replacement_chars    : int          # U+FFFD count
    control_chars        : int          # ord < 32, excluding \n \r \t
    null_bytes           : int          # \x00
    nonprintable_ratio   : float        # (nonprintable / total) ∈ [0.0, 1.0]
    printable_ratio      : float        # (printable / total)
    estimated_entropy    : float        # Shannon entropy in bits/char
    has_binary_runs      : bool         # ≥ 4 consecutive non-printable bytes
    has_instruction_leak : bool         # prompt-like text detected
    language_anomaly     : bool         # non-subject-language detected

    # Derived verdict
    corruption_level     : str          # "CLEAN"|"SUSPICIOUS"|"CORRUPTED"|"BINARY"
    signals_triggered    : List[str] = field(default_factory=list)
    confidence           : float = 0.0  # [0.0, 1.0] that text is corrupted

    def is_clean(self) -> bool:
        return self.corruption_level == "CLEAN"

    def is_safe_for_llm(self) -> bool:
        return self.corruption_level in {"CLEAN", "SUSPICIOUS"}


# ── SIGNAL THRESHOLDS ─────────────────────────────────────────────────────────
CORRUPTION_THRESHOLDS = {
    "replacement_char_abs"     : 1,       # even 1 is a warning
    "replacement_char_rate"    : 0.001,   # > 0.1% → CORRUPTED
    "control_char_rate"        : 0.005,   # > 0.5% → CORRUPTED
    "null_byte_abs"            : 1,       # any null → BINARY
    "nonprintable_rate_warn"   : 0.02,    # > 2% → SUSPICIOUS
    "nonprintable_rate_fail"   : 0.05,    # > 5% → CORRUPTED
    "printable_rate_min"       : 0.90,    # < 90% printable → CORRUPTED
    "entropy_academic_max"     : 5.5,     # normal academic text ~3.5–4.8
    "entropy_binary_min"       : 7.0,     # binary data typically > 7.0
    "binary_run_length"        : 4,       # ≥ 4 consecutive non-printable
}


class EncodingGate:
    """Multi-signal encoding analysis gate."""

    @classmethod
    def analyze(cls, text: str) -> CorruptionReport:
        if not text:
            return CorruptionReport(
                text_length=0,
                replacement_chars=0,
                control_chars=0,
                null_bytes=0,
                nonprintable_ratio=0.0,
                printable_ratio=1.0,
                estimated_entropy=0.0,
                has_binary_runs=False,
                has_instruction_leak=False,
                language_anomaly=False,
                corruption_level="CLEAN",
                signals_triggered=[],
                confidence=0.0,
            )

        # ── STEP 1: IMMEDIATE HARD GATES ─────────────────────────────────────
        null_bytes = text.count("\x00")
        if null_bytes > 0:
            return CorruptionReport(
                text_length=len(text),
                replacement_chars=text.count("\ufffd"),
                control_chars=sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t\x0b\x0c"),
                null_bytes=null_bytes,
                nonprintable_ratio=1.0,
                printable_ratio=0.0,
                estimated_entropy=8.0,
                has_binary_runs=True,
                has_instruction_leak=False,
                language_anomaly=False,
                corruption_level="BINARY",
                signals_triggered=["NULL_BYTE"],
                confidence=1.0,
            )

        # ── STEP 2: CHARACTER ANALYSIS ───────────────────────────────────────
        total_chars       = len(text)
        replacement_chars = text.count("\ufffd")

        control_chars = sum(
            1 for c in text
            if ord(c) < 32 and c not in "\n\r\t\x0b\x0c"
        )

        nonprintable = sum(
            1 for c in text
            if not c.isprintable() and c not in "\n\r\t "
        )

        printable = total_chars - nonprintable

        # ── STEP 3: RATIO COMPUTATION ─────────────────────────────────────────
        replacement_rate  = replacement_chars / max(total_chars, 1)
        control_rate      = control_chars      / max(total_chars, 1)
        nonprintable_rate = nonprintable       / max(total_chars, 1)
        printable_rate    = printable          / max(total_chars, 1)

        # ── STEP 4: SHANNON ENTROPY ───────────────────────────────────────────
        char_freq = Counter(text)
        entropy = -sum(
            (n / total_chars) * math.log2(n / total_chars)
            for n in char_freq.values()
            if n > 0
        )

        # ── STEP 5: BINARY RUN DETECTION ─────────────────────────────────────
        has_binary_runs = False
        run_count = 0
        for char in text:
            if not char.isprintable() and char not in "\n\r\t ":
                run_count += 1
                if run_count >= CORRUPTION_THRESHOLDS["binary_run_length"]:
                    has_binary_runs = True
                    break
            else:
                run_count = 0

        # ── STEP 6: SIGNAL AGGREGATION ───────────────────────────────────────
        signals: List[str] = []

        if replacement_chars >= CORRUPTION_THRESHOLDS["replacement_char_abs"]:
            signals.append("REPLACEMENT_CHAR")
        if replacement_rate > CORRUPTION_THRESHOLDS["replacement_char_rate"]:
            signals.append("HIGH_REPLACEMENT_RATE")
        if control_rate > CORRUPTION_THRESHOLDS["control_char_rate"]:
            signals.append("HIGH_CONTROL_CHAR_RATE")
        if nonprintable_rate > CORRUPTION_THRESHOLDS["nonprintable_rate_fail"]:
            signals.append("HIGH_NONPRINTABLE_RATE")
        if printable_rate < CORRUPTION_THRESHOLDS["printable_rate_min"]:
            signals.append("LOW_PRINTABLE_RATE")
        if entropy > CORRUPTION_THRESHOLDS["entropy_binary_min"]:
            signals.append("BINARY_ENTROPY")
        if has_binary_runs:
            signals.append("BINARY_RUNS")

        # ── STEP 7: VERDICT ───────────────────────────────────────────────────
        n = len(signals)

        if "NULL_BYTE" in signals or "BINARY_ENTROPY" in signals:
            level = "BINARY"
            confidence = 1.0
        elif n >= 3 or "HIGH_REPLACEMENT_RATE" in signals or "BINARY_RUNS" in signals:
            level = "CORRUPTED"
            confidence = min(0.95, 0.60 + n * 0.10)
        elif n >= 1:
            level = "SUSPICIOUS"
            confidence = min(0.80, 0.40 + n * 0.15)
        else:
            level = "CLEAN"
            confidence = 0.0

        return CorruptionReport(
            text_length       = total_chars,
            replacement_chars = replacement_chars,
            control_chars     = control_chars,
            null_bytes        = null_bytes,
            nonprintable_ratio = nonprintable_rate,
            printable_ratio   = printable_rate,
            estimated_entropy = entropy,
            has_binary_runs   = has_binary_runs,
            has_instruction_leak = False,
            language_anomaly  = False,
            corruption_level  = level,
            signals_triggered = signals,
            confidence        = confidence,
        )
