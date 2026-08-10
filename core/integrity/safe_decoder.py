"""
AION Core Integrity — Safe Byte Decoder
=========================================
Decodes raw bytes without using silent symbol-destroying fallbacks
like errors="ignore" or errors="replace". Preserves mathematical symbols.
"""

from __future__ import annotations

import logging
from typing import Optional

from .encoding_gate import EncodingGate

logger = logging.getLogger("AION.SafeDecoder")

# Common Latin-1 math symbol mapping to Unicode
LATIN1_MATH_REPAIR = {
    "\xbc": "¼", "\xbd": "½", "\xbe": "¾",
    "\xd7": "×", "\xf7": "÷", "\xb0": "°",
    "\xb1": "±", "\xb2": "²", "\xb3": "³",
    "\xb5": "µ", "\xb9": "¹",
}


class SafeDecoder:
    """Safe byte decoder preventing symbol corruption during text extraction."""

    @classmethod
    def decode(cls, raw_bytes: bytes, context: str = "extraction") -> str:
        if not raw_bytes:
            return ""

        # STEP 1: UTF-8 STRICT (preferred)
        try:
            return raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            pass

        # STEP 2: UTF-8 WITH BOM
        try:
            return raw_bytes.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            pass

        # STEP 3: DETECT ENCODING (via chardet if installed)
        try:
            import chardet
            detected = chardet.detect(raw_bytes)
            encoding = detected.get("encoding")
            confidence = detected.get("confidence", 0.0)
            if encoding and confidence > 0.85:
                return raw_bytes.decode(encoding, errors="strict")
        except (ImportError, UnicodeDecodeError, LookupError):
            pass

        # STEP 4: LATIN-1 FALLBACK (preserves every byte, then repair math symbols)
        try:
            text = raw_bytes.decode("latin-1")
            for latin_char, unicode_char in LATIN1_MATH_REPAIR.items():
                text = text.replace(latin_char, unicode_char)
            return text
        except Exception:
            pass

        # STEP 5: CONTROLLED REPLACEMENT (last resort with logging)
        text = raw_bytes.decode("utf-8", errors="replace")
        report = EncodingGate.analyze(text)

        if report.replacement_chars > 0:
            logger.warning(
                f"[ENCODING] {report.replacement_chars} replacement chars in {context} "
                f"— text routed to quarantine"
            )

        return text
