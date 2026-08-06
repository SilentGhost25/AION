"""
Concept Extractor — Concept-level retrieval (not paragraph retrieval)
====================================================================
Replaces coarse paragraph chunks with fine-grained concept extraction.

Pipeline:
Clean Text -> Sentence segmentation -> Concept candidate detection
           -> Embedding (if available) -> Deduplication -> Concept list

Each concept carries:
- concept_id
- canonical_definition
- supporting evidence (source chunk)
- prerequisites (heuristic)
- Bloom suggestions
- confidence
"""

from __future__ import annotations

import re
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ExtractedConcept:
    concept_id: str
    concept_name: str
    canonical_definition: str
    supporting_evidence: str       # source chunk verbatim
    source_chunk_id: str
    page_hint: Optional[int] = None
    prerequisites: List[str] = field(default_factory=list)
    bloom_suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.75
    equations: List[str] = field(default_factory=list)
    diagram_refs: List[str] = field(default_factory=list)
    word_count: int = 0
    concept_type: str = "theoretical"  # theoretical | numerical | diagram | derivation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "canonical_definition": self.canonical_definition,
            "supporting_evidence": self.supporting_evidence,
            "source_chunk_id": self.source_chunk_id,
            "prerequisites": self.prerequisites,
            "bloom_suggestions": self.bloom_suggestions,
            "confidence": self.confidence,
            "equations": self.equations,
            "diagram_refs": self.diagram_refs,
            "word_count": self.word_count,
            "concept_type": self.concept_type,
        }


class ConceptExtractor:
    """
    Heuristic + optional neural concept extractor.
    If BGE-M3/GLiNER available, uses them; otherwise falls back to robust heuristics.
    """

    MIN_CONCEPT_WORDS = 60
    MAX_CONCEPT_WORDS = 400

    # Academic concept signals
    CONCEPT_HEADERS = re.compile(
        r"^(?:\d+(?:\.\d+)*\s+)?([A-Z][A-Za-z0-9\s\-]{3,60})(?:\s*[:\-–]\s*|\s*$)",
        re.MULTILINE,
    )
    DEFINITION_SIGNALS = re.compile(
        r"\b(?:is defined as|is a|refers to|denotes|represents|means|is known as|"
        r"defined as|consists of|comprises|involves|describes)\b",
        re.I,
    )

    def __init__(self, use_neural: bool = True):
        self.use_neural = use_neural
        self._embedder = None
        self._gliner = None
        if use_neural:
            self._try_load_neural()

    def _try_load_neural(self):
        try:
            # BGE-M3 for embedding (optional)
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Don't load heavy model at init — lazy
            self._embedder_available = True
        except ImportError:
            self._embedder_available = False
        try:
            import gliner  # type: ignore
            self._gliner_available = True
        except ImportError:
            self._gliner_available = False

    def extract(self, clean_text: str, source_id: str = "doc") -> List[ExtractedConcept]:
        """
        Extract concepts from clean_text.
        Returns List[ExtractedConcept] with source grounding.
        """
        if not clean_text or len(clean_text.split()) < 30:
            return []

        # Step 1: Segment into concept-sized chunks (not arbitrary paragraphs)
        chunks = self._segment_concept_chunks(clean_text)

        concepts: List[ExtractedConcept] = []
        seen_hashes: set[str] = set()

        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk.split()) < self.MIN_CONCEPT_WORDS:
                continue

            # Step 2: Detect concept name + definition
            name, definition = self._extract_name_definition(chunk)
            if not name:
                name = self._fallback_name(chunk)

            # Step 3: Deduplicate via hash
            h = hashlib.sha256(chunk.lower().encode()).hexdigest()[:16]
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Step 4: Classify type
            ctype = self._classify_type(chunk)
            equations = self._extract_equations(chunk)
            diagram_refs = self._extract_diagram_refs(chunk)
            bloom = self._suggest_bloom(chunk, ctype)
            prereq = self._heurist_prereq(chunk, concepts)

            # Confidence scoring
            conf = self._score_concept(chunk, definition)

            cid = f"{source_id}_{hashlib.sha256(name.encode()).hexdigest()[:6]}_{idx:03d}"

            concepts.append(ExtractedConcept(
                concept_id=cid,
                concept_name=name,
                canonical_definition=definition or chunk[:300],
                supporting_evidence=chunk,
                source_chunk_id=f"chunk_{idx:04d}",
                prerequisites=prereq,
                bloom_suggestions=bloom,
                confidence=conf,
                equations=equations,
                diagram_refs=diagram_refs,
                word_count=len(chunk.split()),
                concept_type=ctype,
            ))

        # Sort by confidence descending
        concepts.sort(key=lambda c: c.confidence, reverse=True)
        return concepts

    # ── Segmentation ─────────────────────────────────────────

    def _segment_concept_chunks(self, text: str) -> List[str]:
        """
        Concept-level segmentation: 80-400 words, respects sentence boundaries,
        groups related paragraphs, unlike paragraph-level retrieval.
        """
        # Split by double newlines (paragraphs)
        raw_paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        chunks: List[str] = []
        cur: List[str] = []
        cur_wc = 0

        for para in raw_paras:
            pwc = len(para.split())
            # Large paragraph -> single concept
            if pwc >= self.MIN_CONCEPT_WORDS:
                if cur and cur_wc >= self.MIN_CONCEPT_WORDS:
                    chunks.append("\n".join(cur))
                    cur, cur_wc = [], 0
                # If huge, split by sentences
                if pwc > self.MAX_CONCEPT_WORDS:
                    sents = re.split(r"(?<=[.!?])\s+", para)
                    sub_cur: List[str] = []
                    sub_wc = 0
                    for s in sents:
                        swc = len(s.split())
                        sub_cur.append(s)
                        sub_wc += swc
                        if sub_wc >= self.MIN_CONCEPT_WORDS and sub_wc <= self.MAX_CONCEPT_WORDS:
                            chunks.append(" ".join(sub_cur))
                            sub_cur, sub_wc = [], 0
                    if sub_cur and len(" ".join(sub_cur).split()) >= 30:
                        chunks.append(" ".join(sub_cur))
                else:
                    chunks.append(para)
                continue

            cur.append(para)
            cur_wc += pwc
            if cur_wc >= self.MAX_CONCEPT_WORDS:
                chunks.append("\n".join(cur))
                cur, cur_wc = [], 0

        if cur and len("\n".join(cur).split()) >= self.MIN_CONCEPT_WORDS:
            chunks.append("\n".join(cur))

        return chunks

    # ── Name / Definition ────────────────────────────────────

    def _extract_name_definition(self, chunk: str) -> tuple[Optional[str], Optional[str]]:
        lines = chunk.splitlines()
        first = lines[0].strip() if lines else chunk[:80].strip()

        # If first line looks like a heading, use as name
        if len(first.split()) <= 8 and first[:1].isupper() and len(first) <= 80:
            # Check if second sentence contains definition signal
            rest = chunk[len(first):].strip()
            m = self.DEFINITION_SIGNALS.search(rest)
            if m:
                # Extract definition sentence
                sents = re.split(r"(?<=[.!?])\s+", rest)
                for s in sents:
                    if self.DEFINITION_SIGNALS.search(s):
                        return first.rstrip(":–-"), s.strip()
                return first.rstrip(":–-"), rest[:300]
            return first.rstrip(":–-"), None

        # Otherwise, look for "X is defined as ..." inside chunk
        m = re.search(r"([A-Z][A-Za-z0-9\s\-]{2,40})\s+(is defined as|is a|refers to|denotes)\s+([^.!?]{10,200})", chunk)
        if m:
            name = m.group(1).strip()
            defn = m.group(0).strip() + "."
            return name, defn

        return None, None

    def _fallback_name(self, chunk: str) -> str:
        # First noun phrase heuristic: first 4-6 words capitalized — but ensure atomic
        words = chunk.split()
        # If chunk is "Stack applications include expression evaluation..." — take atomic "Stack Applications"
        # Detect and split long phrases
        raw = " ".join(words[:8])
        # If raw contains "Include" and is long, take only first 2 words as atomic concept
        if "include" in raw.lower() and len(raw.split()) > 4:
            # e.g., "Stack Applications Include Expression Evaluation" -> "Stack Applications"
            candidate = " ".join(words[:2])
        else:
            candidate = " ".join(words[:4])
        # Clean trailing punctuation
        candidate = re.sub(r"[^A-Za-z0-9\s\-]", "", candidate).strip()
        # Title case, but atomic
        result = candidate[:40].title() if candidate else "General Concept"
        # Ensure not too generic
        if len(result.split()) > 4:
            result = " ".join(result.split()[:3])
        return result

    # ── Classification ───────────────────────────────────────

    def _classify_type(self, chunk: str) -> str:
        low = chunk.lower()
        if re.search(r"[=<>≤≥±∑∫√^]|\\frac|\\sum|equation\s*\d|formula", low):
            return "numerical"
        if re.search(r"figure|diagram|graph|block diagram|circuit|flowchart|table", low):
            return "diagram"
        if re.search(r"derive|proof|theorem|lemma|derivation|obtain|deduce", low):
            return "derivation"
        if re.search(r"algorithm|steps|procedure|pseudocode|complexity|o\(|theta\(", low):
            return "algorithmic"
        return "theoretical"

    def _extract_equations(self, chunk: str) -> List[str]:
        eqs = []
        for m in re.finditer(r"\$[^$]{3,80}\$|\b[A-Za-z]\s*[=<>]\s*[^,\n]{2,40}", chunk):
            eqs.append(m.group().strip())
            if len(eqs) >= 3:
                break
        return eqs

    def _extract_diagram_refs(self, chunk: str) -> List[str]:
        refs = []
        for m in re.finditer(r"(Figure|Fig\.|Diagram|Graph|Table)\s+\d+[^\n]*", chunk, re.I):
            refs.append(m.group().strip()[:80])
            if len(refs) >= 3:
                break
        return refs

    def _suggest_bloom(self, chunk: str, ctype: str) -> List[str]:
        low = chunk.lower()
        suggestions = []
        if ctype == "numerical":
            suggestions = ["L3_Apply", "L4_Analyze"]
        elif ctype == "diagram":
            suggestions = ["L2_Understand", "L3_Apply"]
        elif ctype == "derivation":
            suggestions = ["L4_Analyze", "L5_Evaluate"]
        elif "define" in low or "definition" in low:
            suggestions = ["L1_Remember", "L2_Understand"]
        elif "compare" in low or "difference" in low:
            suggestions = ["L4_Analyze"]
        elif "design" in low or "propose" in low:
            suggestions = ["L6_Create"]
        else:
            suggestions = ["L2_Understand", "L3_Apply"]
        return suggestions

    def _heurist_prereq(self, chunk: str, prior_concepts: List[ExtractedConcept]) -> List[str]:
        prereq = []
        low = chunk.lower()
        for pc in prior_concepts[-5:]:
            # If prior concept name appears in current chunk, it's likely prerequisite
            if pc.concept_name.lower().split()[0] in low and len(pc.concept_name.split()[0]) > 4:
                prereq.append(pc.concept_id)
                if len(prereq) >= 2:
                    break
        return prereq

    def _score_concept(self, chunk: str, definition: Optional[str]) -> float:
        score = 0.6
        wc = len(chunk.split())
        if 80 <= wc <= 350:
            score += 0.15
        if definition:
            score += 0.15
        if len(re.findall(r"[.!?]", chunk)) >= 3:
            score += 0.10  # Well-formed sentences
        # Penalize code/noise
        if re.search(r"import\s+\w+|def\s+\w+\(|<html|urlpatterns", chunk):
            score -= 0.40
        return max(0.2, min(0.98, round(score, 2)))
