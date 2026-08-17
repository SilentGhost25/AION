# core/contracts/canonical_question.py

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from core.generation.output_schema import MathBlock
from core.contracts.question import GeneratedQuestion


class QuestionSegment(BaseModel):
    """One content segment. Math is always in math_block, never inferred from text."""
    segment_type : str
    text_value   : str | None = None
    math_block   : MathBlock | None = None
    figure_id    : str | None = None
    alt_text     : str | None = None


class CanonicalQuestion(BaseModel):
    """
    Final structured question for frontend.
    H4 — rendered_math removed. Frontend renders from math_block.latex.
    """
    slot_id         : str
    question_no     : int
    sub_label       : str
    marks           : int
    bloom_level     : str
    co              : str
    module_id       : int
    question_type   : str
    segments        : List[QuestionSegment] = Field(default_factory=list)
    diagram_request : Any = None
    integrity       : Dict[str, Any] = Field(default_factory=dict)
    provenance      : Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_generated(cls, question: GeneratedQuestion) -> CanonicalQuestion:
        """
        H3 — Every [MATH:id] must resolve.
        H4 — No rendered_math parameter. Frontend uses math_block.latex.
        """
        import re

        text = question.question_text
        parts = re.split(r"(\[MATH:[^\]]+\])", text)

        # Obtain math blocks list
        math_blocks_list = []
        if hasattr(question, "output") and question.output is not None:
            math_blocks_list = question.output.math_blocks
        elif hasattr(question, "math_blocks"):
            math_blocks_list = question.math_blocks
            
        declared = {b.block_id: b for b in math_blocks_list}
        segments = []

        for part in parts:
            math_match = re.match(r"\[MATH:([^\]]+)\]", part)
            if math_match:
                block_id = math_match.group(1)
                block    = declared.get(block_id)

                if block is None:
                    raise ValueError(
                        f"[MATH:{block_id}] in question text has no "
                        f"corresponding MathBlock. "
                        f"Declared blocks: {list(declared.keys())}"
                    )

                segments.append(QuestionSegment(
                    segment_type = "math",
                    math_block   = block,
                    alt_text     = block.unicode_fallback,
                ))

            elif part.strip():
                segments.append(QuestionSegment(
                    segment_type = "text",
                    text_value   = part,
                ))

        # Belt-and-suspenders: every declared block must appear
        referenced = set(re.findall(r"\[MATH:([^\]]+)\]", text))
        unused     = set(declared.keys()) - referenced
        if unused:
            raise ValueError(
                f"MathBlocks declared but never referenced: {unused}. "
                f"Every declared block must appear in question_text."
            )

        evidence_ids = list(getattr(question, "evidence_ids", ()))

        return cls(
            slot_id         = question.slot_id,
            question_no     = question.question_no,
            sub_label       = question.sub_label,
            marks           = question.marks,
            bloom_level     = question.bloom_level,
            co              = question.co,
            module_id       = question.module_id,
            question_type   = question.question_type,
            segments        = segments,
            diagram_request = getattr(question, "diagram_request", None),
            integrity       = {
                "math_validated"  : True,
                "bloom_validated" : True,
                "demand_validated": True,
            },
            provenance      = {
                "chunk_ids" : evidence_ids,
                "slot_id"   : question.slot_id,
            },
        )
