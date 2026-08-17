# core/generation/output_schema.py

from typing import List, Optional
from pydantic import BaseModel, field_validator, model_validator, Field


class MathSource(BaseModel):
    chunk_id        : str
    page            : Optional[int] = None


class MathBlock(BaseModel):
    """Canonical math transport. LaTeX is authoritative. Always render from latex."""
    block_id        : str
    latex           : str
    display_mode    : bool = False
    unicode_fallback: str | None = None
    source          : MathSource | str | None = None

    @field_validator("latex")
    @classmethod
    def latex_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("LaTeX must not be empty")
        return v

    @field_validator("latex")
    @classmethod
    def latex_no_corruption(cls, v: str) -> str:
        if "\ufffd" in v or "\x00" in v:
            raise ValueError("LaTeX contains corruption characters")
        return v


class DiagramRequest(BaseModel):
    """Represents a request for a diagram or illustration."""
    diagram_type    : str
    description     : str


class QuestionOutput(BaseModel):
    """
    The ONLY object Qwen returns.
    Contains NO CO, NO bloom_level, NO marks, NO module_id.
    H8 — instruction field enables reliable task clause extraction.
    """
    instruction     : str           # H8 — the action clause ("Analyze X and Y")
    question_text   : str           # full question including context/data
    math_blocks     : List[MathBlock] = Field(default_factory=list)
    diagram_request : Optional[DiagramRequest] = None

    @field_validator("instruction")
    @classmethod
    def instruction_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("Instruction clause must not be empty")
        return v

    @field_validator("question_text")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 8:
            raise ValueError("Question text too short")
        return v

    @field_validator("question_text")
    @classmethod
    def no_answer_in_text(cls, v: str) -> str:
        PATTERNS = [
            "answer:", "solution:", "the answer is", "correct answer",
            "model answer", "expected answer",
        ]
        v_lower = v.lower()
        for p in PATTERNS:
            if p in v_lower:
                raise ValueError(f"Answer leakage: '{p}'")
        return v

    @field_validator("question_text")
    @classmethod
    def no_meta_language(cls, v: str) -> str:
        PATTERNS = [
            "from the source", "from the notes", "provided notes",
            "uploaded document", "source material",
            "based on the provided", "according to the notes",
            "/fontfile", "flatdecode", "endobj",
        ]
        v_lower = v.lower()
        for p in PATTERNS:
            if p in v_lower:
                raise ValueError(f"Meta-language: '{p}'")
        return v

    @model_validator(mode="after")
    def math_references_consistent(self) -> "QuestionOutput":
        """
        H3 — Every [MATH:id] in question_text must have a declared MathBlock.
        Every declared MathBlock must be referenced.
        Silent drops are not allowed.
        """
        import re
        referenced = set(re.findall(r"\[MATH:([^\]]+)\]", self.question_text))
        declared   = {b.block_id for b in self.math_blocks}

        orphan = referenced - declared
        if orphan:
            raise ValueError(
                f"[MATH:...] placeholders reference undeclared blocks: {orphan}. "
                f"Declared blocks: {declared}"
            )

        unused = declared - referenced
        if unused:
            raise ValueError(
                f"MathBlocks declared but never referenced: {unused}. "
                f"Every declared block must appear in question_text."
            )

        return self
