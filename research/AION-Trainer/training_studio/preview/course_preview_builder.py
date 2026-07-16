# AION-Trainer/training_studio/preview/course_preview_builder.py
"""
Course Preview Builder — builds a structured course tree representation
from the session analysis results for rendering in the desktop UI.
"""

from __future__ import annotations

from typing import Dict, List, Any
from training_studio.analyser.analysis_result import SessionAnalysisResult


class CoursePreviewBuilder:
    """
    Transforms session analysis results into a visual-friendly
    course tree representation.
    """

    def build_tree(self, result: SessionAnalysisResult) -> Dict[str, Any]:
        """
        Builds a nested tree dict:
        {
            "subject_code": "BAI401",
            "subject_name": "Artificial Intelligence",
            "modules": [
                {
                    "module_number": 1,
                    "title": "Module 1",
                    "sources": ["textbook1.pdf"],
                    "chapters": [
                        {
                            "chapter_title": "Chapter 1: Intro to AI",
                            "source_file": "textbook1.pdf",
                            "confidence": 0.95
                        }
                    ],
                    "sample_concepts": ["Search Agents", "DFS"]
                }
            ]
        }
        """
        modules_tree = []

        # Group chapter mappings by module number across all analyzed files
        chapters_by_module: Dict[int, List[Dict[str, Any]]] = {}
        for fa in result.file_analyses:
            for mapping in fa.module_mappings:
                mod_num = mapping.get("assigned_module", 0)
                chapters_by_module.setdefault(mod_num, []).append({
                    "chapter_title": mapping.get("chapter_title", ""),
                    "source_file": fa.filename,
                    "confidence": mapping.get("confidence", 0.0),
                    "matching_topics": mapping.get("matching_topics", [])
                })

        # Process each module preview
        for mp in sorted(result.module_previews, key=lambda x: x.module_number):
            mod_num = mp.module_number
            chapters = chapters_by_module.get(mod_num, [])

            modules_tree.append({
                "module_number": mod_num,
                "title": mp.title,
                "concept_count": mp.concept_count,
                "confidence": mp.confidence,
                "sources": mp.sources,
                "sample_concepts": mp.sample_concepts,
                "chapters": chapters
            })

        return {
            "subject_code": result.subject_code or "UNKNOWN",
            "subject_name": result.subject_name or "Unknown Subject",
            "department": result.department,
            "semester": result.semester,
            "total_files": result.total_files,
            "modules": modules_tree
        }
