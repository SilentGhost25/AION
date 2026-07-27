"""
AION Content Validator.
Validates whether a text chunk contains real academic prose vs code/file paths/variable noise.
Runs before concept extraction and question generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ChunkQuality:
    is_valid: bool
    score: float          # 0.0 to 1.0
    reason: str
    chunk_type: str       # prose | code | mixed | noise


# ─────────────────────────────────────────────
# Code / noise signal patterns
# ─────────────────────────────────────────────

CODE_PATTERNS = [
    # Python imports
    r"^\s*(import|from)\s+\w+",
    # Function/class definitions
    r"^\s*(def |class )\w+",
    # Variable assignments
    r"^\s*\w+\s*=\s*[\[\{\"\'\w]",
    # Django/web framework specific
    r"(urlpatterns|INSTALLED_APPS|DATABASES|settings\.py|urls\.py|views\.py|models\.py)",
    # HTML tags
    r"<[a-zA-Z][^>]{0,80}>",
    # File paths
    r"[a-z_]+/[a-z_]+/[a-z_]+",
    # Curly brace templates
    r"\{\{.*?\}\}|\{%.*?%\}",
    # Shell/terminal commands
    r"^\s*\$\s+\w+",
    r"^\s*(pip|python|npm|git|cd|ls|mkdir)\s+",
    # JSON/dict like
    r"^\s*[\"\'][\w_]+[\"\']\s*:\s*",
    # Camel case variable names (not prose)
    r"\b[a-z]+[A-Z][a-zA-Z]+\b",
    # ALL_CAPS constants
    r"\b[A-Z_]{4,}\b",
    # Code brackets
    r"[\[\]\{\}]{2,}",
]

PROSE_SIGNALS = [
    # Complete sentences ending with period
    r"[a-zA-Z]{4,}\s+[a-zA-Z]{4,}\s+[a-zA-Z]{4,}.*\.",
    # Academic verbs
    r"\b(is|are|was|were|can|will|shall|may|should|must|refers|defined|used|known|called|means|describes|represents|denotes)\b",
    # Connective academic language
    r"\b(therefore|however|furthermore|moreover|consequently|additionally|thus|hence|because|since|although|whereas)\b",
    # Academic domain words
    r"\b(algorithm|equation|theorem|definition|principle|concept|method|approach|technique|system|model|process|function|structure|analysis|example|figure|table|note|formula)\b",
]

NOISE_LINE_PATTERNS = [
    r"^\s*#{1,6}\s",                       # Markdown headers with code
    r"^\s*```",                             # Code fences
    r"^\s*\|.+\|.+\|",                    # Markdown tables from code docs
    r"^\s*[-*]{3,}\s*$",                   # Dividers
    r"^\s*[A-Z_]{3,}\s*=",               # CONSTANT = value
    r"^\s*(True|False|None|null|undefined)\s*$",
    r"^\s*\d+\.\d+\.\d+",                 # Version numbers like 3.2.1
    r"^\s*(Copyright|License|MIT|GPL|Apache)\b",
    r"http[s]?://\S+",                     # URLs
    r"^\s*[<>]{2,}",                       # Git conflict markers
    r"^\s*\.\.\.",                          # Ellipsis only lines
]

MIN_PROSE_WORDS        = 50    # was 15 — too low, let noise through
MIN_AVG_WORD_LENGTH    = 3.8   # unchanged
MAX_CODE_LINE_RATIO    = 0.35  # was 0.40 — tighten
MIN_PROSE_SIGNAL_HITS  = 2     # was 1 — need more evidence


def _is_code_line(line: str) -> bool:
    return any(re.search(p, line, flags=re.M) for p in CODE_PATTERNS)


def _is_noise_line(line: str) -> bool:
    return any(re.search(p, line, flags=re.M) for p in NOISE_LINE_PATTERNS)


def _avg_word_length(text: str) -> float:
    words = re.findall(r"[a-zA-Z]+", text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _prose_signal_hits(text: str) -> int:
    return sum(
        1 for p in PROSE_SIGNALS
        if re.search(p, text, flags=re.I)
    )


def validate_chunk(chunk: str) -> ChunkQuality:
    """
    Validate whether a chunk is suitable for question generation.
    Returns ChunkQuality with is_valid flag and reason.
    """
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    words = chunk.split()

    if len(words) < MIN_PROSE_WORDS:
        # Exception: short chunks with math definitions are still valid
        has_definition = bool(re.search(
            r"\b(definition|theorem|lemma|corollary|proof|example)\b",
            chunk, re.I
        ))
        if has_definition and len(words) >= 20:
            pass  # allow short definition blocks through
        else:
            return ChunkQuality(
                is_valid=False, score=0.1,
                reason=f"Too short ({len(words)} words, minimum {MIN_PROSE_WORDS})",
                chunk_type="noise"
            )

    # ── Reject chunks that are mostly URLs or web references ──
    url_count = len(re.findall(r"https?://\S+", chunk))
    if url_count >= 3:
        return ChunkQuality(
            is_valid=False, score=0.05,
            reason=f"Contains {url_count} URLs — web/HTML reference, not academic prose",
            chunk_type="noise"
        )
    if url_count >= 1 and len(words) < 100:
        return ChunkQuality(
            is_valid=False, score=0.1,
            reason="Contains URLs in short chunk — likely HTML tutorial content",
            chunk_type="noise"
        )

    # ── Reject chunks that are mostly TOC-like entries ──
    toc_like = sum(
        1 for ln in lines
        if re.search(r"\.{3,}\s*\d+\s*$", ln)
        or re.search(r"^\s*(chapter|unit|module)\s+\d+\s*$", ln, re.I)
    )
    if toc_like >= max(3, len(lines) // 3):
        return ChunkQuality(
            is_valid=False, score=0.1,
            reason=f"Looks like table of contents ({toc_like} entries)",
            chunk_type="noise"
        )

    if lines:
        code_lines  = sum(1 for ln in lines if _is_code_line(ln))
        noise_lines = sum(1 for ln in lines if _is_noise_line(ln))
        bad_ratio   = (code_lines + noise_lines) / len(lines)
    else:
        bad_ratio = 1.0

    avg_wl = _avg_word_length(chunk)
    prose_hits = _prose_signal_hits(chunk)

    score = 1.0
    score -= bad_ratio * 0.5
    score += min(prose_hits, 6) * 0.06
    score -= max(0, MIN_AVG_WORD_LENGTH - avg_wl) * 0.15
    score = max(0.0, min(1.0, score))

    if bad_ratio > MAX_CODE_LINE_RATIO:
        return ChunkQuality(
            is_valid=False,
            score=score,
            reason=f"Too many code/noise lines ({bad_ratio:.0%})",
            chunk_type="code" if code_lines > noise_lines else "mixed"
        )

    # ── Allow math-heavy / complexity / theorem content even if word length is short ──
    has_math = bool(re.search(
        r"[=<>≤≥±∑∫√\^]|O\(|Θ\(|Ω\(|\b(log|exp|sin|cos|theta|alpha|beta|lambda|sigma)\b"
        r"|\b(algorithm|complexity|theorem|proof|lemma|definition|analysis)\b",
        chunk, re.I
    ))

    if avg_wl < MIN_AVG_WORD_LENGTH and has_math:
        # Override word length check for math / complexity chunks with short tokens
        pass
    elif avg_wl < MIN_AVG_WORD_LENGTH:
        return ChunkQuality(
            is_valid=False,
            score=score,
            reason=f"Average word length too short ({avg_wl:.1f} chars) — likely code tokens",
            chunk_type="code"
        )

    if prose_hits < MIN_PROSE_SIGNAL_HITS:
        return ChunkQuality(
            is_valid=False,
            score=score,
            reason=f"Too few academic prose signals ({prose_hits})",
            chunk_type="noise"
        )

    chunk_type = "mixed" if bad_ratio > 0.15 else "prose"
    return ChunkQuality(
        is_valid=True,
        score=score,
        reason="OK",
        chunk_type=chunk_type
    )


def filter_chunks(chunks: List[str], min_score: float = 0.45) -> List[str]:
    """
    Filter a list of chunks, keeping only valid academic prose.
    """
    scored = []
    for chunk in chunks:
        quality = validate_chunk(chunk)
        if quality.is_valid and quality.score >= min_score:
            scored.append((quality.score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored]


def clean_chunk(chunk: str) -> str:
    """
    Remove code lines and noise lines from a chunk while preserving prose.
    """
    lines = chunk.splitlines()
    cleaned = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if _is_noise_line(s):
            continue
        if _is_code_line(s) and len(s) < 80:
            continue
        cleaned.append(ln)

    text = "\n".join(cleaned)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
