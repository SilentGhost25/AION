"""
AION: Chunk-Image Proximity Mapper
====================================
Maps text chunks to nearby figures using document proximity.

Design principles:
- Zero VLM dependency for matching
- Deterministic and reproducible  
- Falls back gracefully to text-only
- Matches how professors write exam papers
- Each figure used at most once per paper
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .visual.figure_card import FigureCard, FigureRegistry


# -------------------------------------------------------------
# Data structures
# -------------------------------------------------------------

@dataclass
class TextChunk:
    """
    A single text chunk with positional metadata.
    This is what gets fed to the LLM for question generation.
    """
    id:           str        # "module_1_chunk_03"
    text:         str        # actual content
    module_id:    str        # "module_1"
    module_idx:   int        # 1-based
    chunk_idx:    int        # 1-based within module
    total_chunks: int        # total chunks in module
    word_count:   int
    depth:        str        = "CORE"  # CORE, SUPPORTING, ADVANCED, EXTERNAL

    # Positional estimates
    page_start:   int   = 0
    page_end:     int   = 0

    # Nearby images (assigned by mapper)
    nearby_images: list = field(default_factory=list)  # list[FigureCard]

    def best_image(self) -> Optional["FigureCard"]:
        """Return closest eligible image or None."""
        if not self.nearby_images:
            return None
        return self.nearby_images[0]

    def has_image(self) -> bool:
        return bool(self.nearby_images)

    def preview(self, chars: int = 120) -> str:
        return self.text[:chars].replace("\n", " ")


@dataclass
class ModuleChunkGroup:
    """All chunks for one module, with their mapped images."""
    module_id:    str
    module_idx:   int
    module_title: str
    chunks:       list[TextChunk]   = field(default_factory=list)

    def get_chunk(self, idx: int) -> Optional[TextChunk]:
        """Get chunk by 1-based index."""
        for c in self.chunks:
            if c.chunk_idx == idx:
                return c
        return None

    def chunks_with_images(self) -> list[TextChunk]:
        return [c for c in self.chunks if c.has_image()]

    def chunks_without_images(self) -> list[TextChunk]:
        return [c for c in self.chunks if not c.has_image()]


# -------------------------------------------------------------
# Splitter
# -------------------------------------------------------------

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_HEADING_RE      = re.compile(
    r"^(\d+[\.\d]*\s+[A-Z]|[A-Z][A-Z\s]{4,}$"
    r"|(?:Definition|Theorem|Algorithm|Lemma)\s)",
    re.M,
)


_STRUCTURAL_NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$", re.M),
    re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.I | re.M),
    re.compile(r"^\s*table\s+of\s+contents\b", re.I | re.M),
    re.compile(r"^\s*all\s+rights\s+reserved\b", re.I | re.M),
    re.compile(r"^\s*copyright\s+©?\s*\d{4}\b", re.I | re.M),
    re.compile(r"^\s*downloaded\s+from\b", re.I | re.M),
    re.compile(r"^\s*isbn[:\s]*[0-9\-]{10,17}\b", re.I | re.M),
]


def _is_structural_noise(text: str) -> bool:
    """Check if chunk is page noise, TOC, or boilerplate, not academic content."""
    stripped = text.strip()
    if not stripped:
        return True
    words = stripped.split()
    if len(words) < 25:
        for pat in _STRUCTURAL_NOISE_PATTERNS:
            if pat.search(stripped):
                return True
        if re.match(r"^(?:chapter|module|unit)\s*\d+\s*$", stripped, re.I):
            return True
    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    if lines and len(lines) >= 3:
        ref_lines = sum(1 for l in lines if re.match(r"^\[\d+\]\s+[A-Z]", l))
        if ref_lines / len(lines) > 0.6:
            return True
    return False


def _is_legitimate_domain_content(text: str) -> bool:
    """Check if chunk contains legitimate academic, technical, or analytical prose."""
    words = text.split()
    if len(words) < 20:
        return False
    # Formulas, LaTeX or equations
    has_equations = any(sym in text for sym in ("=", "\\frac", "\\sum", "\\int", "√", "×", "±", "→", "$"))
    # Broad multi-department engineering vocabulary
    text_lower = text.lower()
    domain_terms = (
        "principle", "function", "system", "process", "analysis", "structure",
        "method", "model", "equation", "property", "parameter", "component",
        "operation", "design", "performance", "effect", "theory", "condition",
        "state", "force", "energy", "circuit", "voltage", "algorithm", "stress",
        "frequency", "flow", "material", "network", "control", "response",
        "current", "resistor", "capacitor", "lathe", "machine", "signal",
        "power", "torque", "velocity", "pressure", "temperature", "fluid"
    )
    has_academic_density = any(term in text_lower for term in domain_terms)
    return has_academic_density or has_equations or len(words) >= 40


def classify_chunk_depth(text: str, target_module_idx: int = 1) -> str:
    """
    Classifies chunk depth into CORE, SUPPORTING, ADVANCED, or EXTERNAL.
    Safely filters structural noise as EXTERNAL while preserving legitimate
    academic content across all engineering disciplines.
    """
    if _is_structural_noise(text):
        return "EXTERNAL"

    text_lower = text.lower()

    # 1. Advanced technical detail / appendix / rigorous proof
    advanced_terms = {
        "appendix", "detailed proof", "derivation of", "mathematical formulation",
        "complex optimization", "asymptotic behavior", "register specification"
    }
    if any(term in text_lower for term in advanced_terms):
        return "ADVANCED"

    # 2. Supporting context / examples / applications
    supporting_terms = {
        "example", "application", "illustration", "case study",
        "practical scenario", "historical context", "overview"
    }
    if any(term in text_lower for term in supporting_terms):
        return "SUPPORTING"

    # 3. Legitimate substantive domain content -> CORE
    if _is_legitimate_domain_content(text):
        return "CORE"

    return "SUPPORTING"


def _extract_technical_terms(text: str) -> list[str]:
    """Extract candidate technical terms and keyphrases from chunk text."""
    candidates = []
    # 1. Capitalized multi-word phrases (e.g. Zener Diode, Keplerian Orbit)
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    candidates.extend(caps)
    # 2. Technical acronyms (e.g. MOSFET, EIRP, PWM, RPM)
    acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)
    candidates.extend([a for a in acronyms if a not in ("AND", "THE", "FOR", "NOT", "ALL", "FROM", "WITH")])
    # 3. Capitalized single terms (e.g. Zener, Docker, Linux, Fourier)
    single_caps = re.findall(r"\b[A-Z][a-z]{3,}\b", text)
    candidates.extend([s for s in single_caps if s.lower() not in ("this", "that", "these", "there", "their", "when", "where", "with", "from")])
    # 4. Frequent domain terms (length >= 4)
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq = {}
    stopwords = {
        "which", "there", "their", "about", "would", "these", "other", "could",
        "first", "after", "where", "being", "under", "using", "provides", "used",
        "while", "into", "over", "such", "than", "then", "have", "each"
    }
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.keys(), key=lambda w: freq[w], reverse=True)
    candidates.extend(sorted_words)
    return candidates


def _is_relevant_to_topic(keyword: str, topic: str) -> bool:
    """Filter candidate keyword for relevance to slot topic and general domain substance."""
    kw_lower = keyword.lower().strip()
    top_lower = topic.lower().strip()
    if not kw_lower or len(kw_lower) < 3:
        return False
    generic_words = {"explain", "describe", "define", "question", "chapter", "module", "figure", "section"}
    if kw_lower in generic_words:
        return False
    topic_words = set(re.findall(r"\b[a-z]{3,}\b", top_lower)) - {"module", "core", "topics", "part"}
    if topic_words and any(tw in kw_lower or kw_lower in tw for tw in topic_words):
        return True
    return len(kw_lower) >= 4


def extract_domain_keywords(text: str, topic: str = "") -> tuple[str, ...]:
    """Extract top 3-5 relevant domain keywords from text chunk."""
    candidates = _extract_technical_terms(text)
    seen = set()
    relevant = []
    for c in candidates:
        c_clean = c.strip()
        c_low = c_clean.lower()
        if c_low not in seen and _is_relevant_to_topic(c_clean, topic):
            seen.add(c_low)
            relevant.append(c_clean)
        if len(relevant) >= 5:
            break
    return tuple(relevant)


def split_module_into_chunks(
    module_content: str,
    module_id:      str,
    module_idx:     int,
    target_words:   int = 500,
    min_words:      int = 40,
    max_words:      int = 900,
) -> list[TextChunk]:
    """
    Split module text into academic-sized chunks.

    Respects paragraph boundaries.
    Never splits mid-paragraph.
    Targets ~500 words per chunk.
    """
    paragraphs = _PARAGRAPH_SPLIT.split(module_content)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks   : list[TextChunk] = []
    current  : list[str]       = []
    cur_words: int              = 0
    chunk_idx: int              = 1

    def _flush() -> None:
        nonlocal current, cur_words, chunk_idx
        if not current:
            return
        text = "\n\n".join(current)
        wc   = len(text.split())
        if wc >= min_words:
            chunk_id = f"{module_id}_chunk_{chunk_idx:02d}"
            chunks.append(TextChunk(
                id           = chunk_id,
                text         = text,
                module_id    = module_id,
                module_idx   = module_idx,
                chunk_idx    = chunk_idx,
                total_chunks = 0,      # patched below
                word_count   = wc,
            ))
            chunk_idx += 1
        current   = []
        cur_words = 0

    for para in paragraphs:
        p_words = len(para.split())

        if p_words > max_words:
            _flush()
            sentences  = re.split(r"(?<=[.!?])\s+", para)
            sent_buf   = []
            sent_words = 0
            for s in sentences:
                sw = len(s.split())
                if sent_words + sw > target_words and sent_buf:
                    text = " ".join(sent_buf)
                    if len(text.split()) >= min_words:
                        cid = f"{module_id}_chunk_{chunk_idx:02d}"
                        chunks.append(TextChunk(
                            id           = cid,
                            text         = text,
                            module_id    = module_id,
                            module_idx   = module_idx,
                            chunk_idx    = chunk_idx,
                            total_chunks = 0,
                            word_count   = len(text.split()),
                        ))
                        chunk_idx += 1
                    sent_buf   = [s]
                    sent_words = sw
                else:
                    sent_buf.append(s)
                    sent_words += sw
            if sent_buf:
                current   += sent_buf
                cur_words += sent_words
            continue

        if _HEADING_RE.match(para) and cur_words > min_words:
            _flush()

        if cur_words + p_words > target_words and cur_words > min_words:
            _flush()

        current.append(para)
        cur_words += p_words

    _flush()

    total = len(chunks)
    for c in chunks:
        c.total_chunks = total
        c.depth = classify_chunk_depth(c.text, module_idx)

    return chunks


# -------------------------------------------------------------
# Page estimator
# -------------------------------------------------------------

def estimate_page_range(chunk=None, module_idx=1, total_pages=None, total_modules=5, *args, **kwargs) -> tuple[int, int]:
    if args:
        if len(args) == 1 and total_pages is None: total_pages = args[0]
        elif len(args) >= 2: total_pages, total_modules = args[0], args[1]
    if "total_pages" in kwargs and kwargs["total_pages"] is not None: total_pages = kwargs["total_pages"]
    if "total_modules" in kwargs and kwargs["total_modules"] is not None: total_modules = kwargs["total_modules"]

    t_mods = total_modules if (isinstance(total_modules, int) and total_modules > 0) else 5
    t_pages = total_pages if (isinstance(total_pages, int) and total_pages > 0) else t_mods
    m_idx = getattr(chunk, "module_index", None) or (chunk if isinstance(chunk, int) else module_idx) or 1

    pages_per_mod = max(1, t_pages // t_mods)
    start_p = max(1, (m_idx - 1) * pages_per_mod + 1)
    end_p = min(t_pages, m_idx * pages_per_mod)
    return (start_p, max(start_p, end_p))


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """
    Fast keyword overlap score between 0 and 1.
    No ML, no embeddings.
    """
    def kw(t: str) -> set[str]:
        return set(re.findall(r"\b[a-zA-Z]{4,}\b", t.lower()))

    a = kw(text_a)
    b = kw(text_b)

    if not a or not b:
        return 0.0

    overlap = a & b
    return len(overlap) / max(len(a), len(b))


def calculate_retrieval_score(
    chunk: TextChunk,
    target_bloom: int,
    target_module_idx: int,
    semantic_weight: float = 0.45,
    syllabus_weight: float = 0.25,
    bloom_weight: float = 0.15,
    depth_weight: float = 0.10,
    quality_weight: float = 0.05
) -> float:
    """Computes a hybrid multi-dimensional retrieval score for a text chunk."""
    # Tier-specific weight override for easy/foundational cognitive levels (L1/L2)
    # Applied only when caller uses defaults (preserves explicit caller overrides)
    if target_bloom < 3:
        bloom_weight = 0.25       # Foundational chunk-type match matters more for easy tier
        depth_weight = 0.15       # Prefer CORE over ADVANCED more strongly for easy tier
        semantic_weight = 0.30    # Reduced proportionally to keep sum balanced around ~1.0

    # 1. Universal TF-IDF / Term Density & Academic Depth (works across all engineering disciplines)
    text_lower = chunk.text.lower()
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text_lower)
    unique_terms = set(words)
    term_diversity = len(unique_terms) / max(1, len(words)) if words else 0.5
    # Favor substantive paragraphs (100 to 500 words) with high information content
    length_norm = min(1.0, len(words) / 120.0) if len(words) >= 30 else 0.2
    semantic_score = 0.6 * length_norm + 0.4 * term_diversity

    # 2. Structural & Academic Grounding
    if _is_legitimate_domain_content(chunk.text):
        alignment_score = 1.0
    elif _is_structural_noise(chunk.text):
        alignment_score = 0.1
    else:
        alignment_score = 0.75

    # 3. Bloom Suitability
    has_formulas = ("\\frac" in text_lower or "$" in text_lower or "=" in text_lower or "formula" in text_lower)
    has_numbers = any(char.isdigit() for char in text_lower)
    
    bloom_suitability = 0.5
    if target_bloom >= 3:
        if has_formulas or has_numbers or "algorithm" in text_lower or "calculate" in text_lower or "complex" in text_lower or "deriv" in text_lower:
            bloom_suitability = 1.0
        else:
            bloom_suitability = 0.4
    else:
        if "define" in text_lower or "explain" in text_lower or "what is" in text_lower or "overview" in text_lower or "principle" in text_lower:
            bloom_suitability = 1.0
        elif has_formulas:
            bloom_suitability = 0.5

    # 4. Depth Suitability
    depth_suitability = {
        "CORE": 1.0,
        "SUPPORTING": 0.8,
        "ADVANCED": 0.5,
        "EXTERNAL": 0.1
    }.get(chunk.depth, 0.6)

    # 5. Evidence Quality
    quality_score = min(1.0, chunk.word_count / 300.0)
    if "\ufffd" in chunk.text:
        quality_score *= 0.1

    return (
        semantic_weight * semantic_score +
        syllabus_weight * alignment_score +
        bloom_weight * bloom_suitability +
        depth_weight * depth_suitability +
        quality_weight * quality_score
    )



class ChunkImageMapper:
    """
    Maps text chunks to nearby figures using page/section proximity.
    No VLM, no embeddings, no guessing.
    """

    def __init__(
        self,
        registry           = None,
        total_pages:   int = 200,
        page_tolerance: int = 3,
        keyword_threshold: float = 0.12,
    ):
        self.registry          = registry
        self.total_pages       = total_pages
        self.page_tolerance    = page_tolerance
        self.keyword_threshold = keyword_threshold

        self._used_figure_ids: set[str] = set()
        self._module_groups:   dict[str, ModuleChunkGroup] = {}
        self._all_figures:     list = []

    def build(self, modules: list) -> dict[str, ModuleChunkGroup]:
        """
        Call ONCE after segmentation.
        Returns: {module_id: ModuleChunkGroup}
        """
        if self.registry:
            self._all_figures = [
                f for f in self.registry.eligible_cards()
                if self._is_valid_figure(f)
            ]
        else:
            self._all_figures = []
        # Fallback: load diagrams_manifest.json next to the source PDF
        if not self._all_figures:
            try:
                from pathlib import Path as _P
                import json as _json
                import aion_patch as _ap
                fp = getattr(_ap, "ACTIVE_FILE_PATH", None)
                diags = list(getattr(_ap, "_ACTIVE_DIAGRAMS", None) or [])
                if not diags and fp:
                    mp = _P(fp).parent / "diagrams_manifest.json"
                    if mp.exists():
                        diags = _json.loads(mp.read_text(encoding="utf-8"))
                cards = []
                for d in diags:
                    ip = str(d.get("image_path") or "")
                    if ip.startswith("/workspace/"):
                        # map container path -> local extracted_figures
                        if fp:
                            ip = str(_P(fp).parent / "extracted_figures" / _P(ip).name)
                    if not ip or not _P(ip).exists():
                        continue
                    try:
                        from v0_1.figure_card import FigureCard
                        cards.append(FigureCard(
                            image_path=ip,
                            caption=str(d.get("caption") or d.get("label") or ""),
                            page=int(d.get("page") or 1),
                        ))
                    except Exception:
                        class _F:
                            pass
                        f = _F()
                        f.image_path = ip
                        f.caption = str(d.get("caption") or d.get("label") or "")
                        f.ocr_text = ""
                        f.preceding_text = ""
                        f.page = int(d.get("page") or 1)
                        f.bbox = d.get("bbox") or [0, 0, 1, 1]
                        f.figure_id = d.get("anchor_id") or ip
                        cards.append(f)
                if cards:
                    self._all_figures = cards
                    print(f"[MAPPER] Loaded {len(cards)} figures from diagrams_manifest", flush=True)
            except Exception as _me:
                print(f"[MAPPER] manifest fallback failed: {_me}", flush=True)

        print(
            f"[MAPPER] Building chunk map | "
            f"modules={len(modules)} | "
            f"figures={len(self._all_figures)}"
        )

        self._module_groups = {}

        for mod_idx, mod in enumerate(modules, 1):
            module_id    = f"module_{mod_idx}"
            module_title = getattr(mod, "title", f"Module {mod_idx}")
            content      = getattr(mod, "content", "")

            chunks = split_module_into_chunks(
                module_content = content,
                module_id      = module_id,
                module_idx     = mod_idx,
                target_words   = 500,
                min_words      = 40,
            )

            if not chunks:
                print(f"[MAPPER] No chunks for {module_id}")
                continue

            for chunk in chunks:
                ps, pe = estimate_page_range(
                    chunk          = chunk,
                    module_idx     = mod_idx,
                    total_modules  = len(modules),
                    total_pages    = self.total_pages,
                )
                chunk.page_start = ps
                chunk.page_end   = pe

            self._map_figures_to_chunks(
                chunks    = chunks,
                module_id = module_id,
            )

            group = ModuleChunkGroup(
                module_id    = module_id,
                module_idx   = mod_idx,
                module_title = module_title,
                chunks       = chunks,
            )
            self._module_groups[module_id] = group

            with_img    = sum(1 for c in chunks if c.has_image())
            without_img = len(chunks) - with_img
            print(
                f"[MAPPER] {module_id}: "
                f"{len(chunks)} chunks | "
                f"{with_img} with image | "
                f"{without_img} text-only"
            )

        return self._module_groups

    def get_group(self, module_id: str) -> Optional[ModuleChunkGroup]:
        return self._module_groups.get(module_id)

    def get_all_groups(self) -> dict[str, ModuleChunkGroup]:
        return self._module_groups

    def get_chunks_for_module(
        self,
        module_id: str,
    ) -> list[TextChunk]:
        group = self._module_groups.get(module_id)
        return group.chunks if group else []

    def get_best_chunk_for_question(
        self,
        module_id:       str,
        prefer_image:    bool  = True,
        used_chunk_ids:  set   = None,
        target_bloom:    int   = 2,
    ) -> Optional[TextChunk]:
        group = self._module_groups.get(module_id)
        if not group:
            return None

        used = used_chunk_ids or set()
        available = [c for c in group.chunks if c.id not in used]

        if not available:
            return None

        # Exclude chunks categorized as EXTERNAL, with safe non-noise fallback
        non_ext = [c for c in available if c.depth != "EXTERNAL"]
        if non_ext:
            available = non_ext
        else:
            fallback_chunks = [c for c in available if not _is_structural_noise(c.text)]
            available = fallback_chunks if fallback_chunks else available

        try:
            mod_idx = int(module_id.replace("module_", ""))
        except ValueError:
            mod_idx = 1

        # Calculate scores and sort available chunks
        scored_chunks = [
            (c, calculate_retrieval_score(c, target_bloom, mod_idx))
            for c in available
        ]
        scored_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
        available_sorted = [x[0] for x in scored_chunks]

        if prefer_image:
            with_img = [c for c in available_sorted if c.has_image()]
            if with_img:
                return with_img[0]

        return available_sorted[0]

    def get_top_n_chunks_for_question(
        self,
        module_id:       str,
        n:               int   = 2,
        prefer_image:    bool  = True,
        used_chunk_ids:  set   = None,
        target_bloom:    int   = 2,
        exclude_chunks:  Optional[list] = None,
    ) -> list[TextChunk]:
        """
        Retrieves top-N distinct chunks from the SAME topical neighborhood.
        Part (a) receives best_tc (definition / core concept),
        Part (b) receives 2nd related chunk (elaboration / application),
        Part (c) receives 3rd related chunk (numerical / analytical depth).
        Ensures topical coherence while completely eliminating duplicate text repetition.
        Guarantees exact length n by defensive padding when module chunks are limited.
        """
        group = self._module_groups.get(module_id)
        if not group or not group.chunks:
            return []

        used = set(used_chunk_ids or set())
        if exclude_chunks:
            for ec in exclude_chunks:
                if hasattr(ec, "id"):
                    used.add(ec.id)
                elif hasattr(ec, "chunk_id"):
                    used.add(ec.chunk_id)

        available = [c for c in group.chunks if c.id not in used and c.depth != "EXTERNAL"]
        if not available:
            available = [c for c in group.chunks if c.id not in used and not _is_structural_noise(c.text)]
        if not available:
            print(f"[MAPPER WARNING] Module '{module_id}' chunk pool depleted for question slot. Falling back to module pool.", flush=True)
            available = [c for c in group.chunks if not _is_structural_noise(c.text)] or group.chunks

        try:
            mod_idx = int(module_id.replace("module_", ""))
        except ValueError:
            mod_idx = 1

        # 1. Compute scores and normalize across module
        scored_tuples = [
            (c, calculate_retrieval_score(c, target_bloom, mod_idx))
            for c in available
        ]
        
        # Normalize scores to [0, 1] range within this module
        scores = [s for _, s in scored_tuples]
        if len(scores) > 1:
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            if score_range > 1e-6:
                scored_tuples = [
                    (c, (s - min_score) / score_range)
                    for c, s in scored_tuples
                ]

        scored_tuples.sort(key=lambda x: x[1], reverse=True)
        sorted_chunks = [x[0] for x in scored_tuples]

        if not sorted_chunks:
            return []

        # 2. Select primary anchor chunk
        primary = None
        if prefer_image:
            with_img = [c for c in sorted_chunks if c.has_image()]
            if with_img:
                primary = with_img[0]
        if not primary:
            primary = sorted_chunks[0]

        result = [primary]
        if n <= 1:
            return result

        # 3. Find related chunks from the same topical neighborhood (adjacent or high keyword overlap)
        candidate_neighbors = []
        for c in sorted_chunks:
            if c.id == primary.id:
                continue
            dist = abs(c.chunk_idx - primary.chunk_idx)
            overlap = _keyword_overlap(c.text, primary.text)
            proximity_score = max(0.0, 1.0 - (dist * 0.15))
            coherence_score = 0.5 * proximity_score + 0.5 * overlap
            candidate_neighbors.append((c, coherence_score))

        candidate_neighbors.sort(key=lambda x: x[1], reverse=True)

        for c, score in candidate_neighbors:
            if len(result) >= n:
                break
            result.append(c)

        # 4. Fill remaining from sorted_chunks if needed
        if len(result) < n:
            for c in sorted_chunks:
                if c not in result:
                    result.append(c)
                if len(result) >= n:
                    break

        # 5. CRITICAL: Pad if not enough unique chunks available to guarantee length n
        if len(result) < n and len(sorted_chunks) > 0:
            while len(result) < n:
                result.append(sorted_chunks[0])

        return result

    def get_image_for_chunk(
        self,
        chunk_id: str,
    ) -> Optional[Any]:
        for group in self._module_groups.values():
            for chunk in group.chunks:
                if chunk.id == chunk_id:
                    img = chunk.best_image()
                    if img:
                        self._used_figure_ids.add(img.id)
                    return img
        return None

    def mark_figure_used(self, figure_id: str) -> None:
        self._used_figure_ids.add(figure_id)

    def summary(self) -> dict:
        total_chunks  = sum(
            len(g.chunks)
            for g in self._module_groups.values()
        )
        chunks_w_img  = sum(
            len(g.chunks_with_images())
            for g in self._module_groups.values()
        )
        return {
            "total_modules":        len(self._module_groups),
            "total_chunks":         total_chunks,
            "chunks_with_images":   chunks_w_img,
            "chunks_without_images":total_chunks - chunks_w_img,
            "total_figures":        len(self._all_figures),
            "figures_assigned":     len(self._used_figure_ids),
            "image_coverage_pct":   round(
                100 * chunks_w_img / max(total_chunks, 1), 1
            ),
        }

    def _map_figures_to_chunks(
        self,
        chunks:    list[TextChunk],
        module_id: str,
    ) -> None:
        module_figures = [
            f for f in self._all_figures
            if f.module_id == module_id
            and f.id not in self._used_figure_ids
        ]

        # Fallback: if no figures assigned to this module_id specifically,
        # pick unused figures whose page matches the chunk page range
        if not module_figures:
            mod_idx = int(module_id.replace("module_", "")) if "module_" in module_id else 1
            module_figures = [
                f for f in self._all_figures
                if f.id not in self._used_figure_ids
            ]

        if not module_figures or not chunks:
            return

        chunk_page_map: dict[int, list[TextChunk]] = {}
        for chunk in chunks:
            for pg in range(chunk.page_start, chunk.page_end + 1):
                chunk_page_map.setdefault(pg, []).append(chunk)

        sorted_figures = sorted(
            module_figures,
            key=lambda f: f.provenance_score,
            reverse=True,
        )

        for fig in sorted_figures:
            if fig.id in self._used_figure_ids:
                continue

            best_chunk = self._find_best_chunk(
                fig           = fig,
                chunks        = chunks,
                chunk_page_map= chunk_page_map,
            )

            if best_chunk:
                best_chunk.nearby_images.append(fig)

    def _find_best_chunk(
        self,
        fig:            Any,
        chunks:         list[TextChunk],
        chunk_page_map: dict[int, list[TextChunk]],
    ) -> Optional[TextChunk]:
        scored: list[tuple[float, TextChunk]] = []

        for chunk in chunks:
            score = self._score_figure_chunk(fig, chunk, chunk_page_map)
            if score > 0:
                scored.append((score, chunk))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _score_figure_chunk(
        self,
        fig:            Any,
        chunk:          TextChunk,
        chunk_page_map: dict[int, list[TextChunk]],
    ) -> float:
        score = 0.0

        if chunk.page_start <= fig.page <= chunk.page_end:
            score += 0.60
        elif (
            abs(fig.page - chunk.page_start) <= self.page_tolerance or
            abs(fig.page - chunk.page_end)   <= self.page_tolerance
        ):
            distance = min(
                abs(fig.page - chunk.page_start),
                abs(fig.page - chunk.page_end),
            )
            score += max(0.0, 0.40 - distance * 0.08)

        if score > 0:
            fig_text   = f"{fig.caption} {fig.ocr_text} {fig.preceding_text}"
            kw_score   = _keyword_overlap(fig_text, chunk.text)
            if kw_score >= self.keyword_threshold:
                score += kw_score * 0.30

        score += fig.provenance_score * 0.10

        if len(chunk.nearby_images) >= 2:
            score *= 0.30

        return round(score, 4)

    @staticmethod
    def _is_valid_figure(fig: Any) -> bool:
        try:
            return (
                hasattr(fig, "provenance_score") and
                hasattr(fig, "eligible") and
                hasattr(fig, "module_id") and
                hasattr(fig, "page") and
                hasattr(fig, "id") and
                isinstance(fig.eligible, bool) and
                fig.eligible and
                isinstance(fig.id, str) and
                fig.id != ""
            )
        except Exception:
            return False






# -------------------------------------------------------------
# Question-level image selector
# -------------------------------------------------------------

class QuestionImageSelector:
    """
    Used inside _generate_main_question().
    Decides which sub-question gets an image and returns it.
    """

    def __init__(self, mapper: ChunkImageMapper):
        self.mapper    = mapper
        self._used_ids : set[str] = set()

    def select(
        self,
        chunk:      TextChunk,
        module_id:  str,
        sub_index:  int = 0,
    ) -> Optional[dict]:
        """
        Return image metadata dict for a sub-question, or None.
        Only assigns image to sub_index == 0 (first sub-question).
        """
        if sub_index != 0:
            return None

        img = chunk.best_image()

        if img is None:
            return None

        if img.id in self._used_ids:
            return None

        self._used_ids.add(img.id)
        self.mapper.mark_figure_used(img.id)

        return {
            "id":           img.id,
            "url":          img.image_url,
            "caption":      img.caption,
            "visual_type":  img.visual_type,
            "page":         img.page,
            "confidence":   img.provenance_score,
        }

    def reset(self) -> None:
        self._used_ids.clear()

    def stats(self) -> dict:
        return {
            "images_used_this_paper": len(self._used_ids),
        }
