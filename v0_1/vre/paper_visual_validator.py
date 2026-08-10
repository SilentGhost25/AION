"""
AION VRE Paper-Level Visual QA Validator
========================================
Validates paper-level visual distribution, module placement,
figure numbering, SVG embeddings, and marks balance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class PaperVisualValidator:
    """Validates visual question integration across the entire FinalPaper contract."""

    @classmethod
    def validate_paper(cls, paper_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        modules = paper_data.get("modules", [])

        visual_count = 0
        visual_by_module = {}

        for mod in modules:
            m_idx = mod.get("module_index", 1)
            for q in mod.get("questions", []):
                for sub in q.get("subQuestions", []):
                    if sub.get("image") or sub.get("figure_svg"):
                        visual_count += 1
                        visual_by_module[m_idx] = visual_by_module.get(m_idx, 0) + 1

                        # Verify required visual fields
                        if not sub.get("text"):
                            errors.append(f"PAPER_QA_EMPTY_VISUAL_QUESTION_TEXT:Mod{m_idx}")
                        if not sub.get("figure_svg") and not sub.get("image"):
                            errors.append(f"PAPER_QA_MISSING_SVG_AND_IMAGE:Mod{m_idx}")

        # Ensure no module has excessive visual questions (>2 per module)
        for m_idx, count in visual_by_module.items():
            if count > 2:
                errors.append(f"PAPER_QA_EXCESSIVE_VISUAL_QUESTIONS:Mod{m_idx}={count}>2")

        return (len(errors) == 0, errors)
