# AION-Trainer/training_studio/studio_session.py
"""
Studio Session — handles the per-upload session state, manages files,
triggers analysis, and handles resolution of ambiguities.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Any

from training_studio.analyser.analysis_result import SessionAnalysisResult, FileAnalysis, Ambiguity
from training_studio.analyser.analysis_pipeline import AnalysisPipeline

logger = logging.getLogger("aion.studio.session")


class TrainingStudioSession:
    """
    Manages the lifecycle of a training studio session.
    Persists uploaded file lists, runs the pipeline, and resolves ambiguities.
    """

    def __init__(self, session_id: Optional[str] = None, syllabus=None, llm_client=None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.syllabus = syllabus
        self.llm = llm_client
        self.file_paths: List[str] = []
        self.result: SessionAnalysisResult = SessionAnalysisResult(session_id=self.session_id)

    def add_file(self, file_path: str):
        if file_path not in self.file_paths:
            self.file_paths.append(file_path)

    def remove_file(self, file_id: str):
        # Find file path by ID
        target_fa = next((fa for fa in self.result.file_analyses if fa.file_id == file_id), None)
        if target_fa and target_fa.file_path in self.file_paths:
            self.file_paths.remove(target_fa.file_path)
        
        # Prune from result
        self.result.file_analyses = [fa for fa in self.result.file_analyses if fa.file_id != file_id]
        self.result.total_files = len(self.file_paths)
        
        # Prune related ambiguities
        self.result.ambiguities = [a for a in self.result.ambiguities if a.file_id != file_id]
        self.result.compute_readiness()

    def start_analysis(self, progress_callback: Optional[callable] = None) -> SessionAnalysisResult:
        pipeline = AnalysisPipeline(
            syllabus=self.syllabus,
            llm_client=self.llm,
            progress_callback=progress_callback
        )
        self.result = pipeline.run(
            self.file_paths,
            session_id=self.session_id,
            subject_code=self.result.subject_code,
            department=self.result.department,
            semester=self.result.semester
        )
        return self.result

    def resolve_ambiguity(self, ambiguity_id: str, option_index: int) -> bool:
        """
        Applies a resolution option to an ambiguity and triggers
        readiness re-evaluation.
        """
        ambiguity = next((a for a in self.result.ambiguities if a.ambiguity_id == ambiguity_id), None)
        if not ambiguity or option_index >= len(ambiguity.options):
            return False

        option = ambiguity.options[option_index]
        action = option.get("action")
        val = option.get("value")

        logger.info(f"[Session] Resolving ambiguity {ambiguity_id}: {action} -> {val}")

        # Apply the selected action
        if action == "confirm_type":
            fa = self._find_file(ambiguity.file_id)
            if fa:
                fa.document_type = val
                fa.type_needs_confirmation = False

        elif action == "reassign_module":
            fa = self._find_file(ambiguity.file_id)
            if fa:
                for mapping in fa.module_mappings:
                    mapping["assigned_module"] = val
                # Prune chapter module ambiguities for this file
                fa.ambiguous_chapters = []
                self.result.ambiguities = [
                    a for a in self.result.ambiguities
                    if not (a.file_id == fa.file_id and "chapter" in a.title.lower())
                ]

        elif action == "confirm_chapter_module":
            fa = self._find_file(ambiguity.file_id)
            if fa:
                chapter_title = ambiguity.title.replace("Ambiguous chapter: '", "").replace("'", "")
                for ch in fa.module_mappings:
                    if ch.get("chapter_title") == chapter_title:
                        ch["assigned_module"] = val
                # Remove from ambiguous list
                fa.ambiguous_chapters = [
                    c for c in fa.ambiguous_chapters
                    if c.get("chapter_title") != chapter_title
                ]

        elif action == "skip_chapter":
            fa = self._find_file(ambiguity.file_id)
            if fa:
                chapter_title = ambiguity.title.replace("Ambiguous chapter: '", "").replace("'", "")
                # Remove chapter from mappings
                fa.module_mappings = [
                    ch for ch in fa.module_mappings
                    if ch.get("chapter_title") != chapter_title
                ]
                fa.ambiguous_chapters = [
                    c for c in fa.ambiguous_chapters
                    if c.get("chapter_title") != chapter_title
                ]

        elif action == "set_subject":
            self.result.subject_code = val
            from training_studio.classifier.subject_detector import SUBJECT_NAMES
            self.result.subject_name = SUBJECT_NAMES.get(val, val)
            
            # Resolve other subject conflicts
            for fa in self.result.file_analyses:
                fa.subject_code = val
                fa.subject_needs_confirmation = False
            self.result.ambiguities = [
                a for a in self.result.ambiguities
                if "subject" not in a.title.lower()
            ]

        elif action == "assign_subject":
            fa = self._find_file(ambiguity.file_id)
            if fa:
                fa.subject_code = self.result.subject_code
                fa.subject_needs_confirmation = False

        elif action == "remove_file":
            self.remove_file(val)
            # Early return since remove_file already prunes and computes readiness
            return True

        # Mark ambiguity as resolved
        ambiguity.resolved = True
        ambiguity.resolution = option

        # Re-evaluate session readiness
        self.result.compute_readiness()
        return True

    def _find_file(self, file_id: str) -> Optional[FileAnalysis]:
        return next((fa for fa in self.result.file_analyses if fa.file_id == file_id), None)
