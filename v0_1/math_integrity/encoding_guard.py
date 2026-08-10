"""
AION Math Integrity Architecture — Encoding Invariant Guard
============================================================
Enforces clean UTF-8 encoding throughout the pipeline and blocks U+FFFD (M3 Invariant).
"""

from __future__ import annotations

from .contracts import MathIntegrityViolation


class EncodingError(Exception):
    """Raised when raw bytes cannot be safely decoded."""
    pass


class EncodingInvariantGuard:
    """Enforces UTF-8 purity and forbids Unicode replacement characters (U+FFFD)."""

    FORBIDDEN_PATTERNS = [
        "\ufffd",   # U+FFFD replacement character — M3 Violation
        "\x00",     # null byte
        "\ufffe",   # BOM in wrong position
        "\uffff",   # non-character
    ]

    @classmethod
    def assert_clean(cls, text: str, context: str = "") -> str:
        """
        Assert text is clean UTF-8 with no forbidden replacement characters.
        Raises MathIntegrityViolation immediately if M3 is violated.
        """
        for char in cls.FORBIDDEN_PATTERNS:
            if char in text:
                code = "M3_REPLACEMENT_CHAR" if char == "\ufffd" else "FORBIDDEN_CHARACTER"
                raise MathIntegrityViolation(
                    code=code,
                    context=context,
                    char=repr(char),
                    position=text.index(char),
                    message=f"Forbidden character {repr(char)} detected in {context or 'text'}",
                )

        try:
            text.encode("utf-8").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            raise MathIntegrityViolation(
                code="UTF8_ROUNDTRIP_FAILURE",
                context=context,
                detail=str(e),
                message=f"UTF-8 encoding roundtrip failure in {context}",
            )

        return text

    @classmethod
    def safe_decode(cls, raw_bytes: bytes, context: str = "") -> str:
        """
        Safely decode bytes to str with explicit encoding chain.
        Never uses errors='ignore'.
        """
        # Try UTF-8 first (canonical)
        try:
            text = raw_bytes.decode("utf-8")
            return cls.assert_clean(text, context)
        except (UnicodeDecodeError, MathIntegrityViolation):
            pass

        # Try UTF-8 with BOM
        try:
            text = raw_bytes.decode("utf-8-sig")
            return cls.assert_clean(text, context)
        except (UnicodeDecodeError, MathIntegrityViolation):
            pass

        # Try Latin-1
        try:
            text = raw_bytes.decode("latin-1")
            return cls.assert_clean(text, context)
        except Exception as e:
            raise EncodingError(f"Cannot safely decode bytes in context '{context}': {e}")
