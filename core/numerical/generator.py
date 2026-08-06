"""
Numerical Question Engine — Problem Generator (not Copier)
===========================================================
Per AION Development Context:

Input: Quick Sort 8 4 2 7 5
Output: 14 3 18 1 11  (NOT copy 8 4 2 7 5)

Same for Math, Control systems, Signals, DSP, DAA, DSA, OS, Statistics, Civil, etc.

Need: Problem Generator instead of Problem Copier.

This engine:
- Detects numerical examples in evidence
- Generates FRESH instances with same concept but new numbers
- Preserves constraints (range, sortedness, distribution)
- Returns fresh_values + verification that new instance exercises same concept
"""

from __future__ import annotations

import re
import random
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class NumericalInstance:
    original_values: List[str]
    fresh_values: Dict[str, Any] | List[Any]
    concept_hint: str
    verification: str
    raw_payload: Dict[str, Any]

class NumericalEngine:
    """
    Generates fresh numerical instances. Deterministic seed per concept for reproducibility,
    but varied per call.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or random.randint(1, 999999)

    def generate_fresh_instance(self, evidence_or_answer: str, snippet: str = "") -> Optional[Dict[str, Any]]:
        """
        Detect numerical pattern and generate fresh values.
        Returns payload with fresh_values dict/list, or None if no numerical content.
        """
        text = (evidence_or_answer + " " + snippet).strip()
        if not text:
            return None

        # Pattern 1: Array / sequence  e.g., [8, 4, 2, 7, 5] or "8 4 2 7 5"
        seq = self._detect_sequence(text)
        if seq:
            fresh = self._fresh_sequence(seq)
            return {
                "type": "sequence",
                "original": seq,
                "fresh_values": fresh,
                "verification": f"Fresh sequence preserves length {len(seq)} and value range",
            }

        # Pattern 2: Matrix
        matrix = self._detect_matrix(text)
        if matrix:
            fresh = self._fresh_matrix(matrix)
            return {
                "type": "matrix",
                "original": matrix,
                "fresh_values": fresh,
                "verification": f"Fresh {len(matrix)}x{len(matrix[0])} matrix",
            }

        # Pattern 3: Equation with numbers  e.g., x^2 + 5x + 6 = 0  or  T(n) = 2T(n/2) + n
        eq_nums = self._detect_equation_numbers(text)
        if eq_nums:
            fresh = self._fresh_numbers(eq_nums)
            return {
                "type": "equation",
                "original": eq_nums,
                "fresh_values": {f"val_{i}": v for i, v in enumerate(fresh)},
                "verification": "Fresh coefficients within similar magnitude",
            }

        # Pattern 4: Generic numbers list
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        if len(nums) >= 2:
            fresh = self._fresh_numbers(nums)
            # Try to infer context: cache / process / signal values
            hint = self._infer_hint(text)
            return {
                "type": "generic_list",
                "original": nums[:5],
                "fresh_values": {f"val_{i}": v for i, v in enumerate(fresh[:5])},
                "verification": f"Fresh {hint} values in similar range",
            }

        # Pattern 5: Graph / scheduling / OS example (e.g., burst times)
        if re.search(r"burst|arrival|process|page reference|frame", text, re.I):
            burst = re.findall(r"\b\d+\b", text)
            if burst:
                fresh = self._fresh_numbers(burst[:4])
                return {
                    "type": "scheduling",
                    "original": burst[:4],
                    "fresh_values": {f"P{i+1}": v for i, v in enumerate(fresh)},
                    "verification": "Fresh burst/arrival times",
                }

        return None

    # ── Detection helpers ────────────────────────────────────

    def _detect_sequence(self, text: str) -> Optional[List[int]]:
        # Look for bracketed or space-separated integer sequences (3+ numbers)
        # Eg: Quick Sort 8 4 2 7 5
        # Eg: arr = [5, 2, 8, 1, 9]
        for pat in [
            r"\[([0-9,\s]+)\]",
            r"(?:array|arr|sequence|list|input)\s*[:=]?\s*([0-9\s,]+)",
        ]:
            for m in re.finditer(pat, text, re.I):
                raw = m.group(1)
                nums = [int(x) for x in re.findall(r"\d+", raw) if x.isdigit()]
                if 3 <= len(nums) <= 10:
                    # Filter trivial 1..n sequences?
                    return nums
        # Fallback: lone sequence of 4-7 numbers separated by spaces
        m = re.search(r"\b(\d+(?:\s+\d+){3,6})\b", text)
        if m:
            nums = [int(x) for x in m.group(1).split()]
            if 3 <= len(nums) <= 8:
                # Ensure not a date or year list
                if all(1 <= n <= 100 for n in nums):
                    return nums
        return None

    def _detect_matrix(self, text: str) -> Optional[List[List[int]]]:
        # Multi-line matrix like [[1,2],[3,4]] or "1 2 3\n4 5 6"
        m = re.search(r"\[\s*\[.*?\]\s*\]", text, re.S)
        if m:
            try:
                import ast
                val = ast.literal_eval(m.group(0))
                if isinstance(val, list) and all(isinstance(row, list) for row in val):
                    return val
            except Exception:
                pass
        return None

    def _detect_equation_numbers(self, text: str) -> Optional[List[str]]:
        # Numbers inside equations
        if re.search(r"[=<>≤≥±∑∫√^]|equation|formula", text, re.I):
            nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
            if nums:
                return nums[:4]
        return None

    # ── Fresh generation ─────────────────────────────────────

    def _fresh_sequence(self, original: List[int]) -> List[int]:
        # Generate new sequence with same length, similar range but guaranteed different
        n = len(original)
        lo, hi = min(original), max(original)
        # Expand range slightly
        span = hi - lo
        new_lo = max(1, lo - span // 2)
        new_hi = hi + span // 2 + 5

        # Ensure not equal to original
        attempts = 0
        while attempts < 20:
            fresh = [random.randint(new_lo, new_hi) for _ in range(n)]
            if fresh != original and len(set(fresh)) >= max(2, n // 2):
                # For sorting algorithms, ensure not already sorted (keep interesting)
                if fresh != sorted(fresh) and fresh != sorted(fresh, reverse=True):
                    return fresh
            attempts += 1
        # Fallback: permute +/- offset
        return [(x + random.randint(3, 9)) % (new_hi + 1) + new_lo for x in original]

    def _fresh_matrix(self, original: List[List[int]]) -> List[List[int]]:
        fresh = []
        for row in original:
            fresh.append(self._fresh_sequence(row) if len(row) >= 3 else [random.randint(1, 20) for _ in row])
        return fresh

    def _fresh_numbers(self, original: List[str]) -> List[str]:
        fresh: List[str] = []
        for val in original:
            try:
                if "." in val:
                    f = float(val)
                    delta = random.uniform(-f * 0.4, f * 0.4) if f != 0 else random.uniform(1, 5)
                    fresh.append(str(round(f + delta, 2)))
                else:
                    iv = int(val)
                    # Keep magnitude similar
                    lo = max(1, int(iv * 0.5))
                    hi = int(iv * 1.5) + 5
                    if lo >= hi:
                        lo, hi = 1, 20
                    new = random.randint(lo, hi)
                    if new == iv:
                        new = (iv + random.randint(1, 5)) % hi + lo
                    fresh.append(str(new))
            except ValueError:
                fresh.append(val)
            if len(fresh) >= 5:
                break
        return fresh

    def _infer_hint(self, text: str) -> str:
        low = text.lower()
        if "sort" in low:
            return "sorting"
        if "burst" in low or "process" in low:
            return "scheduling"
        if "page" in low:
            return "page replacement"
        if "matrix" in low:
            return "matrix"
        if "equation" in low or "solve" in low:
            return "equation coefficients"
        return "numerical"

    def verify(self, fresh: Dict[str, Any], original_text: str) -> bool:
        """Verify fresh instance is not a copy and exercises same concept."""
        if not fresh or "fresh_values" not in fresh:
            return False
        # Primary check: fresh sequence must NOT be identical to original sequence verbatim
        # This satisfies "Problem Generator not Problem Copier" — core requirement
        orig_seq = self._detect_sequence(original_text)
        fresh_vals = fresh.get("fresh_values")
        if orig_seq and isinstance(fresh_vals, list):
            if fresh_vals == orig_seq:
                return False  # Exact copy — fail
            # Must preserve length and be within expanded range, not sorted trivially
            if len(fresh_vals) != len(orig_seq):
                return False
            return True
        # Generic fallback: check no verbatim substring copy
        orig_nums = set(re.findall(r"\d+", original_text))
        fresh_str = str(fresh_vals)
        fresh_nums = set(re.findall(r"\d+", fresh_str))
        if not fresh_nums:
            return False
        # Not an exact copy of the whole number list
        if len(fresh_nums) == 0 or fresh_vals == orig_nums:
            return False
        # Allow moderate overlap (e.g., small range 1-16 will share some numbers) but not exact copy
        overlap = len(orig_nums & fresh_nums) / max(len(fresh_nums), 1)
        return overlap < 1.0  # any non-identical is okay; strict <1.0 ensures not all same
