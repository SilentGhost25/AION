"""
AION Self-Healing Pipeline (SHP) Orchestrator
===============================================
Orchestrates Stages 0 through 5 with deterministic recovery.

Pipeline Stages:
  Stage 0 — HealthMonitor: Verify dependencies (Ollama, models, dirs)
  Stage 1 — FileDiagnostics: Profile file and select pipeline
  Stage 2 — PipelinePlanner: Lock pipeline, check cache (prevent SH-001)
  Stage 3 — ContentHealer: Filter PDF artifacts, chunk, adaptive validate (SH-014, SH-020, SH-021)
  Stage 4 — RetrievalHealer: Retrieve & verify evidence via GroundingGate
  Stage 5 — OutputRepair: Validate & repair generated questions (SH-032, SH-041)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .error_knowledge   import ErrorKnowledgeBase, ErrorRecord, Severity
from .health_monitor    import SystemHealthMonitor, HealthStatus
from .file_diagnostics  import FileDiagnostics, FileProfile
from .pipeline_planner  import PipelinePlanner, ExtractionPlan
from .content_healer    import ContentHealer, HealedContent
from .retrieval_healer  import RetrievalHealer, RetrievalResult
from .output_repair     import OutputRepair


@dataclass
class SHPPipelineResult:
    success:        bool
    status_code:    int
    healed_content: Optional[HealedContent]  = None
    retrieval:      Optional[RetrievalResult] = None
    health:         Optional[HealthStatus]   = None
    plan:           Optional[ExtractionPlan]  = None
    errors:         list[dict]               = field(default_factory=list)
    repairs_count:  int                      = 0
    message:        str                      = ""


class SHPPipeline:
    """
    Self-Healing Pipeline Orchestrator for AION.
    Wraps document processing with Stage 0-5 self-healing safety nets.
    """

    def __init__(self, root_dir: Path = None):
        self.kb               = ErrorKnowledgeBase()
        self.health_monitor   = SystemHealthMonitor(self.kb, root_dir)
        self.diagnostics      = FileDiagnostics()
        self.planner          = PipelinePlanner(self.kb)
        self.content_healer   = ContentHealer(self.kb)
        self.retrieval_healer = RetrievalHealer(self.kb)
        self.output_repair    = OutputRepair(self.kb)

    def run_pre_generation(
        self,
        file_path:  str,
        notes_text: str = "",
        query:      str = "engineering concepts",
        module_id:  str = "module_1",
    ) -> SHPPipelineResult:
        """
        Run Stages 0 through 4 (Pre-generation pipeline).
        Returns clean, validated evidence chunks ready for LLM generation.
        """
        health = self.health_monitor.check()
        if not health.healthy:
            return SHPPipelineResult(
                success     = False,
                status_code = 500,
                health      = health,
                errors      = self.kb.session_log(),
                message     = f"System health check failed: {health.fatal}",
            )

        if not file_path and notes_text:
            cleaned_chunks = [notes_text]
            ret_res = self.retrieval_healer.retrieve_and_heal(query, cleaned_chunks, module_id=module_id)
            return SHPPipelineResult(
                success     = ret_res.can_generate,
                status_code = 200 if ret_res.can_generate else 400,
                retrieval   = ret_res,
                health      = health,
                errors      = self.kb.session_log(),
                message     = "Inline notes preprocessed successfully.",
            )

        profile = self.diagnostics.diagnose(file_path, self.kb)
        if profile.fatal:
            return SHPPipelineResult(
                success     = False,
                status_code = 400,
                health      = health,
                errors      = self.kb.session_log(),
                message     = profile.fatal,
            )

        plan = self.planner.plan(profile)

        cached = self.planner.load_cached_extraction(file_path)
        if cached and "raw_text" in cached:
            raw_text = cached["raw_text"]
        else:
            from v0_1.extractor import extract
            doc = extract(file_path)
            raw_text = doc.raw_text
            self.planner.save_extraction(file_path, {"raw_text": raw_text})

        healed = self.content_healer.heal(
            raw_text   = raw_text,
            file_path  = file_path,
            chunk_size = plan.chunk_size,
            overlap    = plan.chunk_overlap,
        )

        if not healed.chunks:
            return SHPPipelineResult(
                success        = False,
                status_code    = 400,
                healed_content = healed,
                health         = health,
                plan           = plan,
                errors         = self.kb.session_log(),
                message        = "Content healing failed: 0 chunks accepted.",
            )

        ret_res = self.retrieval_healer.retrieve_and_heal(
            query     = query,
            chunks    = healed.chunks,
            metas     = healed.chunk_metas,
            module_id = module_id,
        )

        repairs_count = len(healed.repairs) + len(ret_res.repairs)

        return SHPPipelineResult(
            success        = ret_res.can_generate,
            status_code    = 200 if ret_res.can_generate else 400,
            healed_content = healed,
            retrieval      = ret_res,
            health         = health,
            plan           = plan,
            errors         = self.kb.session_log(),
            repairs_count  = repairs_count,
            message        = "Pre-generation pipeline complete.",
        )

    def repair_question(
        self,
        question:    str,
        evidence:    list[str],
        bloom_level: int = 2,
        verb:        str = "Explain",
        topic:       str = "the concept",
        subject:     str = "engineering",
        slot_id:     str = "",
    ) -> tuple[str, bool]:
        """Stage 5: Repair a generated question before output rendering."""
        return self.output_repair.repair(
            question    = question,
            evidence    = evidence,
            bloom_level = bloom_level,
            verb        = verb,
            topic       = topic,
            subject     = subject,
            slot_id     = slot_id,
        )
