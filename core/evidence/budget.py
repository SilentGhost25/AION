"""
AION Core Evidence — Stratified Validation Budget
=================================================
Calculates validation budgets proportional to document size with module
and content-type stratification as specified in Part IV.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

logger = logging.getLogger("AION.EvidenceBudget")


@dataclass
class ValidationBudget:
    selected_chunks : List[Any]
    total_chunks    : int
    coverage_ratio  : float
    per_module_min  : int
    stratified      : bool = True


class StratifiedValidationBudget:
    """Computes stratified validation budget across modules and content types."""

    MAX_VALIDATION_CHUNKS : int = 5000   # absolute ceiling
    MIN_VALIDATION_CHUNKS : int = 500    # minimum regardless of size
    MIN_PER_MODULE        : int = 50     # hard minimum per module
    MIN_PER_CONTENT_TYPE  : int = 20     # hard minimum per content type

    @classmethod
    def compute(
        cls,
        all_chunks: List[Any],
        modules: Sequence[Any],
        content_types: Sequence[Any]
    ) -> ValidationBudget:
        total_chunks = len(all_chunks)
        if total_chunks == 0:
            return ValidationBudget(selected_chunks=[], total_chunks=0, coverage_ratio=0.0, per_module_min=cls.MIN_PER_MODULE)

        # STEP 1 — PROPORTIONAL BUDGET (30% proportional)
        proportional = int(total_chunks * 0.30)
        budget = max(cls.MIN_VALIDATION_CHUNKS, min(proportional, cls.MAX_VALIDATION_CHUNKS))
        budget = min(budget, total_chunks)

        module_list = list(modules) if modules else [1]
        num_modules = max(len(module_list), 1)

        # STEP 2 — PER-MODULE ALLOCATION
        per_module_base = budget // num_modules
        per_module = max(per_module_base, cls.MIN_PER_MODULE)

        if per_module * num_modules > budget:
            budget = min(per_module * num_modules, total_chunks)

        # STEP 3 — STRATIFIED SELECTION
        selected: List[Any] = []
        selected_ids = set()

        for mod in module_list:
            mod_chunks = [
                c for c in all_chunks
                if getattr(c, "module_id", None) == mod or getattr(c, "module", None) == mod
            ]
            if not mod_chunks:
                mod_chunks = all_chunks

            for ct in (content_types or ["TEXT"]):
                type_chunks = [
                    c for c in mod_chunks
                    if getattr(c, "content_type", "TEXT") == ct or str(getattr(c, "content_type", "TEXT")) == str(ct)
                ]
                if not type_chunks:
                    continue

                n_from_type = max(
                    cls.MIN_PER_CONTENT_TYPE,
                    int(per_module * len(type_chunks) / max(len(mod_chunks), 1))
                )

                # Prioritize higher confidence
                sorted_type = sorted(
                    type_chunks,
                    key=lambda c: getattr(c, "confidence", 1.0),
                    reverse=True
                )
                sampled = sorted_type[:n_from_type]
                for chunk in sampled:
                    cid = getattr(chunk, "chunk_id", id(chunk))
                    if cid not in selected_ids:
                        selected.append(chunk)
                        selected_ids.add(cid)

        # Fill remaining budget if needed
        if len(selected) < budget:
            remaining = [c for c in all_chunks if getattr(c, "chunk_id", id(c)) not in selected_ids]
            needed = budget - len(selected)
            selected.extend(remaining[:needed])

        coverage = len(selected) / total_chunks if total_chunks > 0 else 0.0

        logger.info(f"[BUDGET] Total chunks      : {total_chunks}")
        logger.info(f"[BUDGET] Validation budget : {len(selected)}")
        logger.info(f"[BUDGET] Coverage          : {coverage:.1%}")
        logger.info(f"[BUDGET] Per-module min    : {cls.MIN_PER_MODULE}")
        logger.info(f"[BUDGET] Stratified        : YES")

        return ValidationBudget(
            selected_chunks=selected,
            total_chunks=total_chunks,
            coverage_ratio=coverage,
            per_module_min=cls.MIN_PER_MODULE,
            stratified=True
        )
