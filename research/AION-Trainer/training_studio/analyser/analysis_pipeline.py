# AION-Trainer/training_studio/analyser/analysis_pipeline.py
"""
Analysis Pipeline — the Analyse stage orchestrator.

No neural training. No Academic Genome construction.
Just fast, safe classification and preview generation.

Returns a SessionAnalysisResult that the UI renders as:
    1. File list with status badges
    2. Ambiguity review screen
    3. Course Preview tree
"""

from __future__ import annotations

import time
import logging
import re
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Tuple

from training_studio.analyser.analysis_result import (
    FileAnalysis, SessionAnalysisResult, AmbiguitySeverity, ModulePreview,
)
from training_studio.analyser.ambiguity_detector import AmbiguityDetector
from training_studio.classifier.document_classifier import DocumentClassifier, DocumentType
from training_studio.classifier.subject_detector import SubjectDetector
from training_studio.classifier.module_mapper import ModuleMapper
from document_intelligence.opendataloader_provider import OpenDataLoaderProvider
from document_intelligence.cache_manager import CacheManager
from document_intelligence.structure_builder import StructureBuilder
from document_intelligence.layout_analyzer import LayoutAnalyzer

logger = logging.getLogger("aion.studio.analysis")


class AnalysisPipeline:
    """
    Runs the full Analyse stage for a set of uploaded files.

    Designed to be fast (< 2 minutes for a typical upload set).
    Heavy operations (concept extraction, genome building) are deferred
    to the Train stage.

    progress_callback: optional callable(message, fraction) for live UI updates
    """

    def __init__(
        self,
        syllabus=None,
        llm_client=None,
        progress_callback: Optional[Callable] = None,
    ):
        self.syllabus = syllabus
        self.progress_callback = progress_callback
        self.doc_classifier = DocumentClassifier()
        self.subject_detector = SubjectDetector(llm_client=llm_client)
        self.module_mapper = ModuleMapper(syllabus)
        self.ambiguity_detector = AmbiguityDetector()
        
        self.doc_provider = OpenDataLoaderProvider()
        self.cache_manager = CacheManager()
        self.structure_builder = StructureBuilder()
        self.layout_analyzer = LayoutAnalyzer()

    def run(
        self,
        file_paths: List[str],
        session_id: str = "",
        subject_code: str = "",
        department: str = "",
        semester: int = 0
    ) -> SessionAnalysisResult:
        import uuid
        result = SessionAnalysisResult(
            session_id=session_id or str(uuid.uuid4())[:8],
            total_files=len(file_paths),
            subject_code=subject_code,
            department=department,
            semester=semester
        )
        self._emit("Starting analysis...", 0.0)

        # Phase 1: Classify + detect subject for each file
        file_analyses = []
        for i, file_path in enumerate(file_paths):
            self._emit(f"Analysing {Path(file_path).name}...", (i / len(file_paths)) * 0.6)
            fa = self._analyse_file(file_path)
            file_analyses.append(fa)
            result.file_analyses.append(fa)

        # Phase 2: Reconcile subject across all files
        self._emit("Reconciling subject detection...", 0.65)
        self._reconcile_subject(result)

        # Phase 3: Count file types
        self._count_file_types(result)

        # Phase 4: Build module previews
        self._emit("Building module preview...", 0.75)
        result.module_previews = self._build_module_previews(result)

        # Phase 5: Detect ambiguities
        self._emit("Detecting ambiguities...", 0.85)
        result.ambiguities = self.ambiguity_detector.detect(result)

        # Phase 6: Compute readiness
        result.analysis_complete = True
        result.compute_readiness()

        self._emit("Analysis complete.", 1.0)
        return result

    def _analyse_file(self, file_path: str) -> FileAnalysis:
        start = time.time()
        filename = Path(file_path).name
        fa = FileAnalysis(
            filename=filename,
            file_path=file_path,
            file_size_bytes=Path(file_path).stat().st_size if Path(file_path).exists() else 0,
            status="analysing",
        )

        try:
            # Step 0: Document Intelligence
            doc = self.cache_manager.load_cached_document(file_path)
            if not doc:
                doc = self.doc_provider.load(file_path)
                # Apply processors
                self.structure_builder.enrich_document(doc, {})
                self.layout_analyzer.enrich_document(doc)
                self.cache_manager.save_document(doc)
                
            fa.document = doc  # Optional, if downstream needs the raw doc

            # Step 1: Document type classification
            type_result = self.doc_classifier.classify_document(doc)
            fa.document_type = type_result.document_type
            fa.type_confidence = type_result.confidence
            fa.type_needs_confirmation = type_result.needs_confirmation
            fa.type_alternatives = type_result.suggested_alternatives
            fa.type_signals = type_result.signals_found

            # Step 2: Subject detection
            subject_result = self.subject_detector.detect_document(doc)
            fa.subject_code = subject_result.subject_code
            fa.subject_name = subject_result.subject_name
            fa.subject_confidence = subject_result.confidence
            fa.subject_needs_confirmation = subject_result.needs_confirmation

            # Step 3: Module mapping (only for textbooks and notes)
            if fa.document_type in (DocumentType.TEXTBOOK, DocumentType.NOTES):
                if doc.toc:
                    mapping_result = self.module_mapper.map_document(doc)
                    fa.module_mappings = [
                        {
                            "chapter_title": m.chapter_title,
                            "assigned_module": m.assigned_module,
                            "confidence": m.confidence,
                            "matching_topics": m.matching_topics,
                        }
                        for m in mapping_result.mappings
                    ]
                    fa.ambiguous_chapters = [
                        {
                            "chapter_title": m.chapter_title,
                            "assigned_module": m.assigned_module,
                            "confidence": m.confidence,
                            "ambiguity_reason": m.ambiguity_reason,
                            "alternative_module": m.alternative_module,
                        }
                        for m in mapping_result.ambiguous_chapters
                    ]

            # Step 4: Quick concept estimation (title-level, not full extraction)
            fa.estimated_concept_count, fa.sample_concepts = (
                self._estimate_concepts(doc)
            )

            fa.status = "complete"

        except Exception as e:
            fa.status = "error"
            fa.error_message = str(e)
            logger.error(f"[AnalysisPipeline] Failed to analyse {filename}: {e}")

        fa.analysis_time_seconds = round(time.time() - start, 2)
        return fa

    def _reconcile_subject(self, result: SessionAnalysisResult):
        """
        If all files agree on subject, set session subject.
        If they disagree, leave resolution to ambiguity review.
        """
        completed = [fa for fa in result.file_analyses if fa.status == "complete"]
        if not completed:
            return

        subject_votes: Dict[str, int] = {}
        for fa in completed:
            if fa.subject_code and fa.subject_code != "UNKNOWN":
                subject_votes[fa.subject_code] = subject_votes.get(fa.subject_code, 0) + 1

        if not subject_votes:
            return

        top_subject = max(subject_votes, key=subject_votes.get)
        top_votes = subject_votes[top_subject]

        if top_votes / len(completed) >= 0.6:
            result.subject_code = top_subject
            from training_studio.classifier.subject_detector import SUBJECT_NAMES
            result.subject_name = SUBJECT_NAMES.get(top_subject, top_subject)

    def _count_file_types(self, result: SessionAnalysisResult):
        for fa in result.file_analyses:
            if fa.document_type == DocumentType.TEXTBOOK:
                result.books_detected += 1
            elif fa.document_type == DocumentType.NOTES:
                result.notes_detected += 1
            elif fa.document_type == DocumentType.QUESTION_BANK:
                result.qb_detected += 1
            elif fa.document_type == DocumentType.PREVIOUS_PAPER:
                result.pyq_detected += 1

    def _build_module_previews(
        self, result: SessionAnalysisResult
    ) -> List[ModulePreview]:
        if not self.syllabus:
            return []

        previews = []
        for mod in self.syllabus.modules:
            contributing_files = []
            concepts_from_module: List[str] = []

            for fa in result.file_analyses:
                if fa.document_type in (DocumentType.TEXTBOOK, DocumentType.NOTES):
                    module_chapters = [
                        m["chapter_title"] for m in fa.module_mappings
                        if m["assigned_module"] == mod.module_number
                    ]
                    if module_chapters:
                        contributing_files.append(fa.filename)
                        concepts_from_module.extend(mod.topics[:3])

            # Use syllabus topics as concept preview
            sample_concepts = list(dict.fromkeys(
                concepts_from_module + mod.topics[:5]
            ))[:8]

            confidence = 0.95 if contributing_files else 0.30

            previews.append(ModulePreview(
                module_number=mod.module_number,
                title=mod.title,
                concept_count=len(mod.topics) if contributing_files else 0,
                confidence=confidence,
                sample_concepts=sample_concepts,
                sources=contributing_files,
            ))

        return previews

    def _estimate_concepts(self, doc: "AcademicDocument") -> Tuple[int, List[str]]:
        """Lightweight preview estimation of concepts."""
        try:
            text = doc.markdown[:5000] # Just the first 5000 characters for estimation
            count = len(doc.sections) * 3 if doc.sections else 10

            # Extracted concepts: find capitalized topic words
            words = re.findall(r"\b[A-Z][a-zA-Z]{3,15}\b", text)
            stops = {"Chapter", "Module", "University", "Syllabus", "Question", "Answer", "Notes", "Lecture", "VTU", "Marks", "Time", "Refer"}
            concepts = list(dict.fromkeys(w for w in words if w not in stops))[:5]

            return max(5, count), concepts
        except Exception:
            return 10, ["Introduction", "Basic Concepts"]

    def _emit(self, message: str, fraction: float):
        if self.progress_callback:
            self.progress_callback(message, fraction)
