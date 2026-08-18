# core/contracts/text_chunk.py

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TextChunk:
    """
    P0.2 — Canonical TextChunk contract.
    has_formula is a required field, not an optional attribute.
    Every extractor/adapter must populate it explicitly.
    """
    chunk_id             : str
    document_id          : str
    module_id            : int
    page_start           : int
    page_end             : int
    text                 : str

    # -- CONTENT FLAGS (required — never optional) -----------------------------
    has_formula          : bool      # True if chunk contains equations
    has_table            : bool      # True if chunk contains tables
    has_figure_reference : bool      # True if chunk references a figure

    # -- LINKED ARTIFACT IDS ---------------------------------------------------
    equation_ids         : List[str] = field(default_factory=list)
    table_ids            : List[str] = field(default_factory=list)
    figure_ids           : List[str] = field(default_factory=list)

    # -- QUALITY ---------------------------------------------------------------
    extraction_confidence: float = 1.0
    topic                : str   = ""
    concept_tags         : List[str] = field(default_factory=list)
    status               : str   = "VALID"

    def __post_init__(self):
        # Populate flags from linked IDs if not explicitly set
        if self.equation_ids and not self.has_formula:
            object.__setattr__(self, "has_formula", True)
        if self.table_ids and not self.has_table:
            object.__setattr__(self, "has_table", True)
        if self.figure_ids and not self.has_figure_reference:
            object.__setattr__(self, "has_figure_reference", True)

    @classmethod
    def from_raw(
        cls,
        chunk_id     : str,
        document_id  : str,
        module_id    : int,
        page_start   : int,
        page_end     : int,
        text         : str,
        equation_ids : Optional[List[str]] = None,
        table_ids    : Optional[List[str]] = None,
        figure_ids   : Optional[List[str]] = None,
        **kwargs,
    ) -> "TextChunk":
        """
        Factory method used by all extractors/adapters.
        Derives has_formula/has_table/has_figure_reference from linked IDs.
        """
        eq  = equation_ids or []
        tbl = table_ids    or []
        fig = figure_ids   or []
        return cls(
            chunk_id              = chunk_id,
            document_id           = document_id,
            module_id             = module_id,
            page_start            = page_start,
            page_end              = page_end,
            text                  = text,
            has_formula           = bool(eq),
            has_table             = bool(tbl),
            has_figure_reference  = bool(fig),
            equation_ids          = eq,
            table_ids             = tbl,
            figure_ids            = fig,
            **{k: v for k, v in kwargs.items()
               if k in cls.__dataclass_fields__},
        )
