"""
AION Math Integrity Architecture — Core Data Structures & Invariants
====================================================================
Implements non-negotiable invariants M1-M5 and core math data structures.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class MathIntegrityViolation(Exception):
    """Raised when any foundational math integrity invariant (M1-M5) is broken."""

    def __init__(
        self,
        code: str,
        message: str = "",
        math_id: str = "",
        field: str = "",
        position: int = -1,
        context: str = "",
        char: str = "",
        detail: str = "",
    ):
        self.code = code
        self.message = message or f"Math integrity violation: {code}"
        self.math_id = math_id
        self.field = field
        self.position = position
        self.context = context
        self.char = char
        self.detail = detail
        super().__init__(self.message)


class MathRepresentation(str, Enum):
    LATEX          = "LATEX"          # canonical (M2)
    UNICODE        = "UNICODE"        # derived display
    MATHML         = "MATHML"         # derived markup
    SVG            = "SVG"            # derived rendering
    OMML           = "OMML"           # derived DOCX
    ASCII          = "ASCII"          # derived plaintext fallback
    EXPRESSION_AST = "AST"            # machine representation


class MathSourceType(str, Enum):
    NATIVE_LATEX   = "NATIVE_LATEX"   # LaTeX found natively in document
    OCR_IMAGE      = "OCR_IMAGE"      # extracted from equation image
    UNICODE_TEXT   = "UNICODE_TEXT"   # converted from Unicode
    DOCLING_STRUCT = "DOCLING_STRUCT" # Docling structural extraction
    INLINE_TEXT    = "INLINE_TEXT"    # embedded in prose


class MathValidationStatus(str, Enum):
    VALID       = "VALID"
    REPAIRABLE  = "REPAIRABLE"
    CORRUPT     = "CORRUPT"      # M3 violation — block generation
    TRUNCATED   = "TRUNCATED"    # delimiter imbalance
    UNPARSEABLE = "UNPARSEABLE"


class EquationType(str, Enum):
    INLINE     = "INLINE"       # within text flow: $...$
    DISPLAY    = "DISPLAY"      # standalone: \[...\]
    NUMBERED   = "NUMBERED"     # \begin{equation}...\end{equation}
    DEFINITION = "DEFINITION"   # variable definition
    FORMULA    = "FORMULA"      # named formula
    NUMERICAL  = "NUMERICAL"    # pure arithmetic


class HealerAction(str, Enum):
    SYMBOL_REPLACE    = "SYMBOL_REPLACE"    # π -> \pi
    ENCODING_REPAIR   = "ENCODING_REPAIR"   # UTF-8 re-decode
    DELIMITER_BALANCE = "DELIMITER_BALANCE" # close open \[
    OCR_RERUN         = "OCR_RERUN"         # re-OCR equation image
    BLOCKED           = "BLOCKED"           # cannot heal


@dataclass(frozen=True)
class MathSymbol:
    """Entry in the Math Symbol Registry translation table."""
    name        : str
    unicode     : str
    latex       : str
    mathml      : str
    html_entity : str
    ascii_approx: Optional[str]
    category    : str
    domains     : List[str]

    def __post_init__(self):
        assert self.unicode.encode("utf-8"), "Unicode character must be valid UTF-8"
        assert (
            self.latex.startswith("\\") or self.latex.startswith("^") or len(self.latex) == 1
        ), f"Invalid LaTeX symbol format: {self.latex}"


@dataclass
class MathNode:
    """Node in a mathematical expression Abstract Syntax Tree (AST)."""
    node_type : str                             # "fraction" | "symbol" | "operator" | "sqrt" ...
    value     : Optional[str] = None            # leaf value
    children  : List[MathNode] = field(default_factory=list)
    metadata  : Dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def symbol_set(self) -> Set[str]:
        symbols: Set[str] = set()
        if self.is_leaf() and self.value:
            symbols.add(self.value)
        for child in self.children:
            symbols |= child.symbol_set()
        return symbols


@dataclass
class ExpressionAST:
    root       : MathNode
    variables  : List[str] = field(default_factory=list)
    constants  : List[str] = field(default_factory=list)
    operators  : List[str] = field(default_factory=list)
    complexity : int = 1


@dataclass
class MathArtifact:
    """
    Canonical representation of any mathematical expression in AION.
    M1 — Math is an object, not a string.
    M2 — LaTeX is the canonical representation.
    """
    math_id             : str
    latex               : str
    normalized_latex    : str
    ast                 : Optional[ExpressionAST] = None

    unicode_text        : Optional[str] = None
    mathml              : Optional[str] = None
    svg                 : Optional[str] = None
    omml                : Optional[str] = None

    source_type         : MathSourceType = MathSourceType.INLINE_TEXT
    source_text         : Optional[str] = None
    source_image        : Optional[bytes] = None
    equation_type       : EquationType = EquationType.INLINE

    document_id         : str = "doc_001"
    page                : int = 1
    bbox                : Optional[Tuple[int, int, int, int]] = None

    validation_status   : MathValidationStatus = MathValidationStatus.VALID
    parse_confidence    : float = 1.0
    render_confidence   : float = 1.0
    round_trip_verified : bool = False

    source_hash         : str = ""
    canonical_hash      : str = ""
    placeholder         : str = ""

    def __post_init__(self):
        if not self.placeholder:
            self.placeholder = f"[MATH:{self.math_id}]"

        # M3 invariant check — no replacement characters allowed
        self._assert_no_replacement_char()

        assert self.placeholder.startswith("[MATH:") and self.placeholder.endswith("]"), (
            f"Invalid placeholder format: {self.placeholder}"
        )

        if not self.canonical_hash:
            target = self.normalized_latex or self.latex or ""
            self.canonical_hash = hashlib.sha256(target.encode("utf-8")).hexdigest()

        if not self.source_hash and self.source_text:
            self.source_hash = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()

    def _assert_no_replacement_char(self):
        for field_val in (self.latex, self.normalized_latex, self.unicode_text, self.source_text):
            if field_val and "\ufffd" in field_val:
                raise MathIntegrityViolation(
                    code="M3_REPLACEMENT_CHAR",
                    math_id=self.math_id,
                    message="Unicode replacement character detected in MathArtifact",
                    field=field_val,
                )

    def best_for_llm(self) -> str:
        """M4 — Qwen references equations via placeholders, never serializes them."""
        return self.placeholder

    def best_for_display(self) -> str:
        """Returns LaTeX for rendering."""
        return self.normalized_latex or self.latex

    def verify_canonical_hash(self) -> bool:
        target = self.normalized_latex or self.latex or ""
        computed = hashlib.sha256(target.encode("utf-8")).hexdigest()
        return computed == self.canonical_hash


@dataclass
class ProtectedTextEnvelope:
    """Text envelope with all math extracted and replaced by placeholders."""
    text        : str
    artifacts   : Dict[str, MathArtifact] = field(default_factory=dict)
    original    : str = ""
    document_id : str = "doc_001"

    def restore(self) -> str:
        """Re-inserts canonical LaTeX into text for rendering."""
        result = self.text
        for placeholder, artifact in self.artifacts.items():
            result = result.replace(placeholder, artifact.best_for_display())
        return result

    def for_llm(self) -> str:
        """Returns text with placeholders intact for Qwen."""
        return self.text

    def math_ids(self) -> List[str]:
        return [a.math_id for a in self.artifacts.values()]

    def has_math(self) -> bool:
        return len(self.artifacts) > 0
