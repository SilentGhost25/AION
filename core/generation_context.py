"""
AION Generation Context
=======================
Replaces raw path passing throughout the pipeline.
Everything the generator needs in one object.
No file paths. Only IDs and pre-loaded data.
"""

from dataclasses import dataclass, field
from typing import Optional


from core.config.production_model import get_production_model

@dataclass
class GenerationContext:
    """
    Everything needed to generate a question paper in a single object.
    Passed between pipeline stages. Side-effect free.
    """

    # Identity
    job_id:      str
    doc_id:      str
    filename:    str

    # Configuration
    exam_type:   str                   # IA or SEE
    difficulty:  str                   # Easy / Medium / Hard / Mixed
    selected_modules: list[str]        # module IDs to generate from
    marks_per_question: int            # 10 for IA, 20 for SEE
    model:       str = field(default_factory=get_production_model)

    # Pre-loaded content (no path access after creation)
    modules:     list[dict] = field(default_factory=list)
    chunks:      list[dict] = field(default_factory=list)
    figures:     list[dict] = field(default_factory=list)

    # Generation state
    questions:   list[dict] = field(default_factory=list)
    paper:       Optional[dict] = None
    error:       str = ""

    @classmethod
    def from_document(
        cls,
        doc,              # Document object
        job_id:      str,
        exam_type:   str   = "IA",
        difficulty:  str   = "Mixed",
        selected_modules: list[str] = None,
        model:       Optional[str] = None,
    ) -> "GenerationContext":
        """Create a GenerationContext from a ready Document."""
        marks = 10 if exam_type == "IA" else 20
        return cls(
            job_id            = job_id,
            doc_id            = doc.id,
            filename          = doc.filename,
            exam_type         = exam_type,
            difficulty        = difficulty,
            selected_modules  = selected_modules or [],
            marks_per_question = marks,
            model             = model or get_production_model(),
            modules           = doc.modules,
            chunks            = doc.chunks,
            figures           = doc.figures,
        )

    def get_chunks_for_module(self, module_id: str) -> list[dict]:
        """Return only chunks belonging to the specified module."""
        return [
            c for c in self.chunks
            if c.get("module_id") == module_id
            or c.get("module")   == module_id
        ]

    def get_module(self, module_id: str) -> Optional[dict]:
        for m in self.modules:
            if m.get("id") == module_id or m.get("module_id") == module_id:
                return m
        return None
