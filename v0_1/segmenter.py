"""
AION Module: Segmenter
Enhanced: RobustSegmenter with multi-strategy segmentation pipeline.
Strategies:
1. Explicit module/chapter markers
2. Heading detection (ALL CAPS, 1.1 Heading, Title Case)
3. Paragraph density analysis
4. Sentence boundary split
5. Sentence-aware equal split (never splits mid-sentence)
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ModuleSegment:
    title:      str
    content:    str
    word_count: int

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class SegmentResult:
    segments: List[ModuleSegment]


def _clean_pdf_artifacts(text: str) -> str:
    """Remove PDF extraction artifacts from text."""
    text = re.sub(r'\[PDF\s+p\.?\s*\d+\]', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


class RobustSegmenter:
    """
    Multi-strategy segmenter.
    Tries 4 strategies before falling back to sentence-aware equal splits.
    """

    EXPLICIT_PATTERNS = [
        r'(?i)^module\s*[-–:]?\s*\d+',
        r'(?i)^chapter\s*\d+',
        r'(?i)^unit\s*\d+',
        r'(?i)^\d+\.\s+[A-Z][a-z]',
        r'(?i)^part\s+[IVX]+',
        r'(?i)^section\s+\d+',
    ]

    HEADING_PATTERNS = [
        r'^[A-Z][A-Z\s]{8,}$',
        r'^\d+\.\d*\s+[A-Z]',
        r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,4}$',
    ]

    def segment(self, text: str, target_n: int = 5, min_words: int = 50) -> List[ModuleSegment]:
        text = _clean_pdf_artifacts(text)
        if not text:
            return []

        # Strategy 1: Explicit markers
        segments = self._by_explicit_markers(text, target_n)
        if self._is_valid(segments, min_words):
            print(f"[SEGMENTER] Strategy: explicit markers -> {len(segments)} segments")
            return segments

        # Strategy 2: Heading detection
        segments = self._by_headings(text, target_n)
        if self._is_valid(segments, min_words):
            print(f"[SEGMENTER] Strategy: heading detection -> {len(segments)} segments")
            return segments

        # Strategy 3: Paragraph density analysis
        segments = self._by_paragraph_density(text, target_n)
        if self._is_valid(segments, min_words):
            print(f"[SEGMENTER] Strategy: paragraph density -> {len(segments)} segments")
            return segments

        # Strategy 4: Sentence boundary split
        segments = self._by_sentence_boundaries(text, target_n)
        if self._is_valid(segments, min_words):
            print(f"[SEGMENTER] Strategy: sentence boundary -> {len(segments)} segments")
            return segments

        # Final fallback: sentence-aware equal split
        print(f"[SEGMENTER] Strategy: sentence-aware equal split -> {target_n} segments")
        return self._by_sentence_aware_equal(text, target_n)

    def _by_explicit_markers(self, text: str, target_n: int) -> List[ModuleSegment]:
        lines = text.split('\n')
        splits = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            for pat in self.EXPLICIT_PATTERNS:
                if re.match(pat, stripped):
                    splits.append({"line_idx": i, "title": stripped})
                    break

        return self._build_segments(lines, splits, "Module")

    def _by_headings(self, text: str, target_n: int) -> List[ModuleSegment]:
        lines = text.split('\n')
        splits = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) < 5:
                continue
            for pat in self.HEADING_PATTERNS:
                if re.match(pat, stripped):
                    splits.append({"line_idx": i, "title": stripped[:60]})
                    break

        return self._build_segments(lines, splits, "Section")

    def _by_paragraph_density(self, text: str, target_n: int) -> List[ModuleSegment]:
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]

        if len(paragraphs) < target_n:
            return []

        group_size = len(paragraphs) // target_n
        segments = []

        for i in range(target_n):
            start = i * group_size
            end = (i + 1) * group_size if i < target_n - 1 else len(paragraphs)
            content = "\n\n".join(paragraphs[start:end])
            wc = len(content.split())
            if wc > 0:
                segments.append(ModuleSegment(
                    title=f"Module {i+1}",
                    content=content,
                    word_count=wc
                ))

        return segments

    def _by_sentence_boundaries(self, text: str, target_n: int) -> List[ModuleSegment]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        total_words = len(text.split())
        target_words = max(1, total_words // target_n)

        segments = []
        current_sentences = []
        current_words = 0

        for sent in sentences:
            current_sentences.append(sent)
            current_words += len(sent.split())

            if current_words >= target_words and len(segments) < target_n - 1:
                content = " ".join(current_sentences)
                segments.append(ModuleSegment(
                    title=f"Module {len(segments)+1}",
                    content=content,
                    word_count=len(content.split())
                ))
                current_sentences = []
                current_words = 0

        if current_sentences:
            content = " ".join(current_sentences)
            segments.append(ModuleSegment(
                title=f"Module {len(segments)+1}",
                content=content,
                word_count=len(content.split())
            ))

        return segments

    def _by_sentence_aware_equal(self, text: str, target_n: int) -> List[ModuleSegment]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        total = len(sentences)
        group = max(1, total // target_n)
        segments = []

        for i in range(target_n):
            start = i * group
            end = (i + 1) * group if i < target_n - 1 else total
            content = " ".join(sentences[start:end])
            wc = len(content.split())
            if wc > 0:
                segments.append(ModuleSegment(
                    title=f"Module {i+1}",
                    content=content,
                    word_count=wc
                ))

        return segments

    def _build_segments(self, lines: List[str], splits: List[Dict[str, Any]], prefix: str) -> List[ModuleSegment]:
        if not splits:
            return []

        segments = []
        for i, split in enumerate(splits):
            start_line = split["line_idx"]
            end_line = splits[i+1]["line_idx"] if i + 1 < len(splits) else len(lines)
            content = "\n".join(lines[start_line:end_line]).strip()
            wc = len(content.split())
            if wc > 10:
                segments.append(ModuleSegment(
                    title=f"{prefix} {i+1}: {split['title'][:40]}",
                    content=content,
                    word_count=wc
                ))

        return segments

    def _is_valid(self, segments: List[ModuleSegment], min_words: int) -> bool:
        if not segments or len(segments) < 2:
            return False
        return all(s.word_count >= min_words for s in segments)


def segment_document(text: str, file_path: str = "") -> SegmentResult:
    segmenter = RobustSegmenter()
    segments = segmenter.segment(text, target_n=5)
    return SegmentResult(segments=segments)


# Compatibility alias
Segmenter = RobustSegmenter
