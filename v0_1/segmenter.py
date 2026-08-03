"""
AION Module: Segmenter
Maturity:    v0.1 — Document Segmenter
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class ModuleSegment:
    title:      str
    content:    str
    word_count: int


@dataclass
class SegmentResult:
    segments: List[ModuleSegment]


def _clean_pdf_artifacts(text: str) -> str:
    """Remove PDF extraction artifacts from text."""
    # Remove [PDF p.N] markers
    text = re.sub(r'\[PDF\s+p\.?\s*\d+\]', '', text)
    # Remove page number lines
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def segment_document(text: str, file_path: str = "") -> SegmentResult:
    # Step 1: Clean PDF artifacts first
    text = _clean_pdf_artifacts(text)
    text = text.strip()

    if not text:
        return SegmentResult(segments=[])

    print(f"[SEGMENTER] Total text length: {len(text)} chars, {len(text.split())} words")

    # Step 2: Try module/chapter markers
    pattern = re.compile(
        r"(?:^|\n)\s*"
        r"(MODULE\s+\d+|MODULE\s+[IVXLCDM]+"
        r"|CHAPTER\s+\d+|CHAPTER\s+[IVXLCDM]+"
        r"|Module\s+\d+|Module\s+[IVXLCDM]+"
        r"|Chapter\s+\d+|Chapter\s+[IVXLCDM]+)"
        r"([^\n]*)",
        re.I
    )

    matches = list(pattern.finditer(text))
    print(f"[SEGMENTER] Found {len(matches)} module/chapter markers")

    segments = []

    if matches:
        for i, match in enumerate(matches):
            content_start = match.end()
            content_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            base_title  = match.group(1).strip()
            rest_line   = match.group(2).strip().lstrip(":-– ")
            title = f"{base_title}: {rest_line}" if rest_line else base_title

            content    = text[content_start:content_end].strip()
            word_count = len(content.split())

            print(f"[SEGMENTER] Segment '{title}': {word_count} words")

            if word_count < 10:
                print(f"[SEGMENTER] [WARNING] Skipping '{title}' — too short ({word_count} words)")
                continue

            segments.append(ModuleSegment(
                title      = title,
                content    = content,
                word_count = word_count
            ))

    # Step 3: Fallback — split into equal parts
    if not segments:
        print("[SEGMENTER] No markers found — splitting into equal parts")
        words      = text.split()
        total      = len(words)
        chunk_size = max(1, total // 5)

        for i in range(min(5, max(1, total))):
            start   = i * chunk_size
            end     = (i + 1) * chunk_size if i < 4 else total
            content = " ".join(words[start:end])
            wc      = len(content.split())

            if wc == 0:
                continue

            segments.append(ModuleSegment(
                title      = f"Module {i + 1}",
                content    = content,
                word_count = wc
            ))
            print(f"[SEGMENTER] Fallback segment 'Module {i+1}': {wc} words")

    print(f"[SEGMENTER] Final: {len(segments)} valid segments")
    return SegmentResult(segments=segments)
