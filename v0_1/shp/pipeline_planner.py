"""
AION SHP Stage 2 — Pipeline Planner
=====================================
Chooses ONE extraction pipeline per document.
Prevents duplicate extraction (Root Cause 1 fix).

Rule: Extract once. Cache. Never extract again.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .file_diagnostics import FileProfile
from .error_knowledge  import ErrorKnowledgeBase, Severity


@dataclass
class ExtractionPlan:
    """Locked pipeline plan for a single document."""
    doc_id:         str
    pipeline:       str
    chunk_size:     int = 300
    chunk_overlap:  int = 30
    extract_math:   bool = False
    extract_images: bool = False
    extract_tables: bool = False
    cache_path:     str = ""
    locked:         bool = False

    def lock(self):
        self.locked = True
        print(f"[SHP-S2] Plan locked: {self.pipeline} | "
              f"chunk={self.chunk_size}w overlap={self.chunk_overlap}w")


class PipelinePlanner:
    """
    Stage 2: Decide the extraction pipeline before any extractor runs.
    Prevents the dual-extraction bug (SH-001).
    """

    CACHE_DIR = Path("workspace/cache/plans")

    def __init__(self, kb: ErrorKnowledgeBase):
        self.kb = kb
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._active_plans: dict[str, ExtractionPlan] = {}

    def plan(self, profile: FileProfile) -> ExtractionPlan:
        doc_id = self._doc_id(profile.path)

        cached = self._load_cache(doc_id)
        if cached:
            print(f"[SHP-S2] Using cached plan for {doc_id}")
            return cached

        if doc_id in self._active_plans:
            rec = self.kb.record(
                "SH-001", "S2_PLAN",
                f"Duplicate extraction attempt for {doc_id}",
                Severity.WARNING,
            )
            plan = self._active_plans[doc_id]
            self.kb.resolve(rec, "Returned existing plan")
            return plan

        pipeline   = profile.recommended_pipeline
        chunk_size = self._choose_chunk_size(profile)

        plan = ExtractionPlan(
            doc_id         = doc_id,
            pipeline       = pipeline,
            chunk_size     = chunk_size,
            chunk_overlap  = max(20, chunk_size // 10),
            extract_math   = profile.contains_math,
            extract_images = profile.contains_images,
            extract_tables = profile.contains_tables,
            cache_path     = str(self.CACHE_DIR / f"{doc_id}.json"),
        )
        plan.lock()

        self._active_plans[doc_id] = plan
        self._save_cache(plan)

        return plan

    def has_cached_extraction(self, file_path: str) -> bool:
        doc_id = self._doc_id(file_path)
        result_path = Path("workspace/cache") / f"{doc_id}_result.json"
        return result_path.exists()

    def load_cached_extraction(self, file_path: str) -> Optional[dict]:
        doc_id = self._doc_id(file_path)
        result_path = Path("workspace/cache") / f"{doc_id}_result.json"
        if result_path.exists():
            try:
                data = json.loads(result_path.read_text())
                print(f"[SHP-S2] Loaded cached extraction for {doc_id}")
                return data
            except Exception:
                pass
        return None

    def save_extraction(self, file_path: str, result: dict) -> None:
        doc_id = self._doc_id(file_path)
        result_path = Path("workspace/cache") / f"{doc_id}_result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[SHP-S2] Extraction cached: {doc_id}")

    def _choose_chunk_size(self, profile: FileProfile) -> int:
        if profile.contains_math:
            return 250
        if profile.estimated_words > 100_000:
            return 300
        if profile.estimated_words < 5_000:
            return 200
        return 300

    def _doc_id(self, file_path: str) -> str:
        p = Path(file_path)
        try:
            stat   = p.stat()
            raw    = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        except Exception:
            raw    = file_path
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _save_cache(self, plan: ExtractionPlan) -> None:
        cache = Path(plan.cache_path)
        cache.write_text(json.dumps({
            "doc_id":        plan.doc_id,
            "pipeline":      plan.pipeline,
            "chunk_size":    plan.chunk_size,
            "chunk_overlap": plan.chunk_overlap,
            "extract_math":  plan.extract_math,
        }))

    def _load_cache(self, doc_id: str) -> Optional[ExtractionPlan]:
        path = self.CACHE_DIR / f"{doc_id}.json"
        if path.exists():
            try:
                d = json.loads(path.read_text())
                plan = ExtractionPlan(**d)
                plan.lock()
                return plan
            except Exception:
                pass
        return None
