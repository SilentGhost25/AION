# AION-Trainer/training_studio/analyser/ambiguity_detector.py
"""
Ambiguity Detector — finds conflicts across all uploaded files.

Detects:
    Module mismatch: file named Module1 contains Module 2 content
    Subject conflict: two files classified to different subjects
    Duplicate content: two files contain the same concepts
    Low-confidence type: file type classification is uncertain
    Low-confidence module: chapter belongs to multiple modules equally
"""

from __future__ import annotations

import logging
from typing import List

from training_studio.analyser.analysis_result import (
    FileAnalysis, Ambiguity, AmbiguitySeverity, SessionAnalysisResult,
)
from training_studio.classifier.document_classifier import DocumentType

logger = logging.getLogger("aion.studio.ambiguity")


class AmbiguityDetector:
    def detect(self, result: SessionAnalysisResult) -> List[Ambiguity]:
        ambiguities: List[Ambiguity] = []

        ambiguities.extend(self._detect_type_uncertainty(result.file_analyses))
        ambiguities.extend(self._detect_subject_conflicts(result.file_analyses))
        ambiguities.extend(self._detect_module_mismatches(result.file_analyses))
        ambiguities.extend(self._detect_ambiguous_chapters(result.file_analyses))

        logger.info(
            f"[AmbiguityDetector] Found {len(ambiguities)} ambiguities "
            f"({sum(1 for a in ambiguities if a.severity == AmbiguitySeverity.ERROR)} errors)"
        )
        return ambiguities

    def _detect_type_uncertainty(self, files: List[FileAnalysis]) -> List[Ambiguity]:
        ambiguities = []
        for fa in files:
            if fa.type_needs_confirmation:
                alternatives = fa.type_alternatives or []
                options = [
                    {"label": f"Yes, it is a {fa.document_type}",
                     "action": "confirm_type", "value": fa.document_type},
                ] + [
                    {"label": f"No, it is a {alt}",
                     "action": "confirm_type", "value": alt}
                    for alt in alternatives
                ]
                ambiguities.append(Ambiguity(
                    severity=AmbiguitySeverity.WARNING,
                    file_id=fa.file_id,
                    filename=fa.filename,
                    title=f"Document type uncertain: {fa.filename}",
                    description=(
                        f"AION classified this as '{fa.document_type}' "
                        f"({fa.type_confidence:.0%} confidence). "
                        f"Please confirm the document type."
                    ),
                    options=options,
                ))
        return ambiguities

    def _detect_subject_conflicts(self, files: List[FileAnalysis]) -> List[Ambiguity]:
        """Flag if two files are classified to different subjects."""
        ambiguities = []
        subjects = {fa.subject_code for fa in files if fa.subject_code and fa.subject_code != "UNKNOWN"}

        if len(subjects) > 1:
            subject_list = sorted(subjects)
            options = [
                {"label": f"All files belong to {s}",
                 "action": "set_subject", "value": s}
                for s in subject_list
            ]
            ambiguities.append(Ambiguity(
                severity=AmbiguitySeverity.ERROR,
                title="Multiple subjects detected across uploaded files",
                description=(
                    f"Files appear to belong to different subjects: "
                    f"{', '.join(subject_list)}. "
                    f"AION trains one subject at a time. "
                    f"Please confirm which subject these files belong to, "
                    f"or remove files from the wrong subject."
                ),
                options=options,
            ))

        # Subject-unknown files
        for fa in files:
            if fa.subject_needs_confirmation or fa.subject_code == "UNKNOWN":
                ambiguities.append(Ambiguity(
                    severity=AmbiguitySeverity.WARNING,
                    file_id=fa.file_id,
                    filename=fa.filename,
                    title=f"Subject not detected: {fa.filename}",
                    description=(
                        f"AION could not confidently determine which subject "
                        f"this file belongs to "
                        f"(confidence: {fa.subject_confidence:.0%}). "
                        f"Please select the correct subject."
                    ),
                    options=[
                        {"label": "Assign to session subject",
                         "action": "assign_subject", "value": "session"},
                        {"label": "Remove from training",
                         "action": "remove_file", "value": fa.file_id},
                    ],
                ))
        return ambiguities

    def _detect_module_mismatches(self, files: List[FileAnalysis]) -> List[Ambiguity]:
        """
        Detect the key scenario described in the spec:
        a file named 'Module1.pdf' that actually contains Module 2 content.
        """
        import re
        ambiguities = []

        for fa in files:
            if fa.document_type not in (DocumentType.NOTES,):
                continue

            # Extract module number from filename
            filename_module_match = re.search(
                r"module\s*[-_]?\s*(\d)", fa.filename, re.IGNORECASE
            )
            if not filename_module_match:
                continue

            filename_module = int(filename_module_match.group(1))

            # Check if majority of content maps to a different module
            if not fa.module_mappings:
                continue

            assigned_modules = [m["assigned_module"] for m in fa.module_mappings]
            if not assigned_modules:
                continue

            majority_module = max(set(assigned_modules), key=assigned_modules.count)

            if majority_module != filename_module:
                ambiguities.append(Ambiguity(
                    severity=AmbiguitySeverity.WARNING,
                    file_id=fa.file_id,
                    filename=fa.filename,
                    title=f"Module mismatch: {fa.filename}",
                    description=(
                        f"The file is named 'Module {filename_module}' but its content "
                        f"appears to belong to Module {majority_module} based on the syllabus. "
                        f"Move automatically?"
                    ),
                    options=[
                        {"label": f"Yes, treat as Module {majority_module}",
                         "action": "reassign_module", "value": majority_module},
                        {"label": f"No, keep as Module {filename_module}",
                         "action": "reassign_module", "value": filename_module},
                    ],
                ))
        return ambiguities

    def _detect_ambiguous_chapters(self, files: List[FileAnalysis]) -> List[Ambiguity]:
        """Chapters the module mapper couldn't confidently assign."""
        ambiguities = []
        for fa in files:
            for chapter in fa.ambiguous_chapters:
                ambiguities.append(Ambiguity(
                    severity=AmbiguitySeverity.INFO,
                    file_id=fa.file_id,
                    filename=fa.filename,
                    title=f"Ambiguous chapter: '{chapter.get('chapter_title', '')}'",
                    description=(
                        f"{chapter.get('ambiguity_reason', 'Low confidence assignment')} "
                        f"(assigned Module {chapter.get('assigned_module')}, "
                        f"confidence {chapter.get('confidence', 0):.0%})"
                    ),
                    options=[
                        {"label": f"Keep in Module {chapter.get('assigned_module')}",
                         "action": "confirm_chapter_module",
                         "value": chapter.get("assigned_module")},
                        {"label": f"Move to Module {chapter.get('alternative_module')}",
                         "action": "confirm_chapter_module",
                         "value": chapter.get("alternative_module")},
                        {"label": "Skip this chapter",
                         "action": "skip_chapter", "value": None},
                    ],
                ))
        return ambiguities
