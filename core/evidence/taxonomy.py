"""
AION Core Evidence — Taxonomy
==============================
Defines the canonical EvidenceType taxonomy and eligibility sets.
Every extracted chunk/block is classified before entering retrieval.
"""

from __future__ import annotations

from enum import Enum
from typing import Set


class EvidenceType(str, Enum):
    """
    Classification taxonomy for extracted objects and chunks.
    Determines retrieval eligibility and processing pipeline.
    """

    # -- ACADEMIC EVIDENCE (retrieval eligible) --------------------------------
    TEXT_PROSE        = "TEXT_PROSE"        # normal academic paragraph
    TEXT_DEFINITION   = "TEXT_DEFINITION"   # definition / terminology
    TEXT_THEOREM      = "TEXT_THEOREM"      # theorem / lemma / proof
    TEXT_EXAMPLE      = "TEXT_EXAMPLE"      # worked example
    TEXT_CODE         = "TEXT_CODE"         # pseudocode / algorithm / code snippet
    EQUATION          = "EQUATION"          # mathematical expression
    TABLE_DATA        = "TABLE_DATA"        # data table
    FIGURE_DIAGRAM    = "FIGURE_DIAGRAM"    # diagram / circuit / graph
    FIGURE_CHART      = "FIGURE_CHART"      # chart / plot
    LIST_ENUMERATION  = "LIST_ENUMERATION"  # ordered/unordered list

    # -- NON-ACADEMIC (quarantined — never retrieval eligible) -----------------
    PDF_METADATA      = "PDF_METADATA"      # /FontFile2, /ToUnicode, /Contents, etc.
    PDF_FONT_DATA     = "PDF_FONT_DATA"     # font stream internals
    PDF_IMAGE_DATA    = "PDF_IMAGE_DATA"    # raw image stream metadata
    PDF_XREF          = "PDF_XREF"          # cross-reference table
    BINARY_STREAM     = "BINARY_STREAM"     # compressed / encoded stream
    UNICODE_CORRUPT   = "UNICODE_CORRUPT"   # replacement char (\ufffd) contamination

    # -- EXCLUDED (not retrieval eligible, but not errors) ---------------------
    REFERENCE_LIST    = "REFERENCE_LIST"    # bibliography / references
    INDEX_PAGE        = "INDEX_PAGE"        # book index
    HEADER_FOOTER     = "HEADER_FOOTER"     # page header / footer
    TOC               = "TOC"               # table of contents
    BLANK             = "BLANK"             # empty / whitespace only


RETRIEVAL_ELIGIBLE_TYPES: Set[EvidenceType] = {
    EvidenceType.TEXT_PROSE,
    EvidenceType.TEXT_DEFINITION,
    EvidenceType.TEXT_THEOREM,
    EvidenceType.TEXT_EXAMPLE,
    EvidenceType.TEXT_CODE,
    EvidenceType.EQUATION,
    EvidenceType.TABLE_DATA,
    EvidenceType.FIGURE_DIAGRAM,
    EvidenceType.FIGURE_CHART,
    EvidenceType.LIST_ENUMERATION,
}

QUARANTINE_TYPES: Set[EvidenceType] = {
    EvidenceType.PDF_METADATA,
    EvidenceType.PDF_FONT_DATA,
    EvidenceType.PDF_IMAGE_DATA,
    EvidenceType.PDF_XREF,
    EvidenceType.BINARY_STREAM,
    EvidenceType.UNICODE_CORRUPT,
}

EXCLUDED_TYPES: Set[EvidenceType] = {
    EvidenceType.REFERENCE_LIST,
    EvidenceType.INDEX_PAGE,
    EvidenceType.HEADER_FOOTER,
    EvidenceType.TOC,
    EvidenceType.BLANK,
}
