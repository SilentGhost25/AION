"""
AION Module: Pipeline Orchestrator
Maturity:    v0.1 — PURE SEQUENTIAL PIPELINE
Upgrades to: Async EventBus Orchestrator with Fail-Safe Context Expansion Loops
Contract:    Run pure pipeline matching AION Architecture Diagram.
"""

import re
import json
import time
import random
import hashlib
import pickle
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .uploader  import upload
from .extractor import extract
from .cleaner   import clean
from .memory    import ConceptMemoryStore
from .learner   import Learner
from .generator import generate, generate_turbo, _is_valid_cs_question
from .critic    import review
from .schemas   import GeneratedQuestion


# ─────────────────────────────────────────────────────────────
# Turbo helpers (defined here so main.py is self-contained)
# ─────────────────────────────────────────────────────────────

_PREAMBLE = re.compile(
    r"^(here('s| is)|sure|certainly|below is|absolutely)[^\n]*[:\n]+",
    re.I,
)
_TRAILER = re.compile(
    r"\n\s*(\*\*)?Note:?.*$|\n\s*\(Note:.*$",
    re.S | re.I,
)



def _wrap(text: str, width: int = 100, indent: str = "          ") -> str:
    """Format question text cleanly without slicing or truncation."""
    words   = text.split()
    lines   = []
    current = []
    length  = 0
    for word in words:
        if length + len(word) + 1 > width and current:
            lines.append(" ".join(current))
            current = [word]
            length  = len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return ("\n" + indent).join(lines)


def _clean_turbo(raw: str) -> str:
    """Clean turbo output without dropping multi-sentence or multi-part questions."""
    t = raw.strip()
    t = _PREAMBLE.sub("", t)
    t = _TRAILER.sub("", t)
    t = re.sub(r"^\**Question\**\s*:?\s*", "", t, flags=re.I)
    t = re.sub(r"^\**Descriptive Exam Question\**\s*:?\s*", "", t, flags=re.I)
    t = re.sub(r"^\**VTU Exam Question\**\s*:?\s*", "", t, flags=re.I)
    t = re.sub(r"^Q\d*[.)]\s*", "", t)
    t = re.sub(r"\*{1,2}", "", t)          # strip bold/italic markers
    t = re.sub(r"\s*\(\d+\s*marks?\)", "", t, flags=re.I)  # strip "(5 marks)"
    return t.strip()


# ── Updated VTU verb pattern ──────────────────────────────────
# Matches single verb at start — not a comma-separated list
_VTU_VERB = re.compile(
    r"^(\(?\d+\)?[.\s]*)?("
    r"explain|compare|derive|analyse|analyze|illustrate|describe|"
    r"define|discuss|evaluate|justify|design|examine|interpret|"
    r"construct|apply|assess|investigate|demonstrate|identify|"
    r"outline|summarize|classify|differentiate|formulate|"
    r"elaborate|state|show|prove|calculate|determine|solve|"
    r"implement|develop|create|build|propose|suggest|recommend|"
    r"critique|review|reflect|predict|estimate|measure|test|"
    r"verify|validate|simulate|model|represent|map|trace|"
    r"what|how|why|when|where|which|who"  # question word starts
    r")\b",
    re.I,
)

# Leaked verb list pattern — these are prompt artifacts, not real questions
_LEAKED_VERB_LIST = re.compile(
    r"^(Analyse|Evaluate|Examine|Justify|Assess|Explain|Compare|Derive)"
    r"\s*,\s*(Evaluate|Compare|Derive|Analyse|Illustrate|Discuss|"
    r"Critically examine|Justify|Assess|Explain)",
    re.I,
)


def _cheap_validate(question: str) -> bool:
    """
    Fast validation — no LLM needed.
    Only rejects questions that are clearly broken.
    """
    q     = question.strip()
    words = q.split()

    # ── Length check ──────────────────────────────────────────
    if len(words) < 8:
        return False
    if len(words) > 200:     # raised from 120 — multi-part questions can be long
        return False

    # ── Reject leaked verb lists ──────────────────────────────
    # e.g. "Analyse, Evaluate, Critically examine, Justify, or Assess..."
    if _LEAKED_VERB_LIST.match(q):
        return False

    # ── Must start with a VTU verb or question word ───────────
    if not _VTU_VERB.match(q):
        return False

    # ── Must not still contain source references ──────────────
    _SOURCE_REF = re.compile(
        r"\b(as per the (source|material|text|notes|document)|"
        r"from the (source|material|text|notes|document)|"
        r"as described in the|as outlined in the|"
        r"as mentioned in the|as stated in the|"
        r"provided in the (source|material)|"
        r"in the source material|"
        r"s material\b)",   # catches the broken "decisionss material" artifact
        re.I,
    )
    if _SOURCE_REF.search(q):
        return False

    return True


def _grounded(question: str, chunk: str, min_overlap: int = 1) -> bool:
    """Question must share key terms with source chunk."""
    q_terms = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", question)}
    c_terms = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", chunk)}
    return len(q_terms & c_terms) >= min_overlap


# ─────────────────────────────────────────────────────────────
# Concept Caching
# ─────────────────────────────────────────────────────────────

CACHE_DIR = Path("extracted_output") / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _concept_cache_key(file_path: str) -> str:
    """SHA256 of file path + file size + modified time (or contained files if directory)."""
    p    = Path(file_path)
    try:
        if p.is_dir():
            suffixes = {".pdf", ".txt", ".md"}
            files = sorted(
                [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in suffixes and not f.name.startswith(".")],
                key=lambda f: f.name
            )
            raw_parts = [file_path]
            for f in files:
                stat = f.stat()
                raw_parts.append(f"{f.name}:{stat.st_size}:{stat.st_mtime}")
            raw = "|".join(raw_parts)
        else:
            stat = p.stat()
            raw  = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
    except Exception:
        raw  = f"{file_path}:0:0"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_cached_concepts(file_path: str):
    try:
        key   = _concept_cache_key(file_path)
        cache = CACHE_DIR / f"{key}.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                data = pickle.load(f)
            print(f"[CACHE] Loaded {len(data['concepts'])} concepts from cache (skipping extraction)")
            return data["concepts"], data["memory_stats"]
    except Exception:
        pass
    return None, None


def _save_cached_concepts(file_path: str, concepts, memory_stats):
    try:
        key   = _concept_cache_key(file_path)
        cache = CACHE_DIR / f"{key}.pkl"
        with open(cache, "wb") as f:
            pickle.dump({"concepts": concepts, "memory_stats": memory_stats}, f)
        print(f"[CACHE] Saved {len(concepts)} concepts to cache")
    except Exception as e:
        print(f"[CACHE] Could not save cache: {e}")


def run_pipeline(
    file_path:    str,
    max_concepts: int = 10,
    mode:         str = "balanced",
) -> Tuple[List[GeneratedQuestion], List[Tuple[GeneratedQuestion, str]]]:
    """
    Executes the end-to-end AION v0.1 ingestion & generation pipeline.

    mode:
      "turbo"    — question ONLY, no answer, no critic, ~3-5 s/question
      "balanced" — question + answer, LLM critic, normal speed
      "deep"     — question + answer + marking scheme, strict critic
    """
    print("=" * 60)
    print(f"[START] AION v0.1 Ingestion & Generation Pipeline starting (Mode: {mode.upper()})...")
    print("=" * 60 + "\n")

    # ── 1 & 2: Try cache first ───────────────────────────────
    cached_concepts, cached_stats = _load_cached_concepts(file_path)

    if cached_concepts is not None:
        concepts     = cached_concepts
        memory_store = ConceptMemoryStore()
        print(f"[GENOME] Concepts loaded from cache: {len(concepts)}")
        print(f"   - Total Concepts in Memory Graph: {cached_stats.get('total_concepts', '?')}\n")
    else:
        # ── 1. Ingestion & Preprocessing ────────────────────────
        validated_path = upload(file_path)
        p = Path(validated_path)
        if p.is_dir():
            suffixes = {".pdf", ".txt", ".md"}
            files = sorted(
                [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in suffixes and not f.name.startswith(".")],
                key=lambda f: f.name
            )
            if not files:
                raise FileNotFoundError(f"No PDF, TXT, or MD files found in directory: {file_path}")

            print(f"[PIPELINE] Processing directory: {file_path} ({len(files)} files found)")

            memory_store = ConceptMemoryStore()
            learner      = Learner(memory_store=memory_store)
            concepts     = []

            for file_item in files:
                print(f"\n[PIPELINE] Processing file: {file_item.name} ...")
                try:
                    val_file_path = upload(str(file_item))
                    document      = extract(val_file_path)
                    cleaned       = clean(document)

                    report_path = Path("extracted_output") / "last_report.json"
                    if report_path.exists():
                        try:
                            rpt = json.loads(report_path.read_text(encoding="utf-8"))
                            print(f"  [DIAG] Method used: {rpt.get('method')}")
                            kept    = rpt.get("kept_pages", [])
                            print(f"  [DIAG] Kept pages ({len(kept)} total)")
                        except Exception:
                            pass

                    print(f"  [DOCUMENT] Document Processed: {document.source_path}")
                    print(f"     - Cleaned lines:      {cleaned.original_line_count - cleaned.removed_line_count}")

                    file_concepts = learner.learn(cleaned)
                    concepts.extend(file_concepts)
                    print(f"  [GENOME] Concepts Extracted from {file_item.name}: {len(file_concepts)}")
                except Exception as e:
                    print(f"  [ERROR] Failed to process {file_item.name}: {e}")

            print(f"\n[GENOME] All concepts Extracted & Synced to Memory. Combined concepts count: {len(concepts)}")
            stats = memory_store.stats()
            print(f"   - Total Concepts in Memory Graph: {stats['total_concepts']}\n")

            # Save to cache for next run
            _save_cached_concepts(file_path, concepts, stats)
        else:
            document       = extract(validated_path)
            cleaned        = clean(document)

            # Diagnostic report
            report_path = Path("extracted_output") / "last_report.json"
            if report_path.exists():
                try:
                    rpt = json.loads(report_path.read_text(encoding="utf-8"))
                    print(f"[DIAG] Method used: {rpt.get('method')}")
                    print(f"[DIAG] TOC found: {rpt.get('toc_found')}")
                    print(f"[DIAG] TOC entries: {len(rpt.get('toc_entries', []))}")
                    kept    = rpt.get("kept_pages", [])
                    dropped = rpt.get("dropped_pages", {})
                    print(f"[DIAG] Kept pages ({len(kept)} total): {kept[:10]}...")
                    print(f"[DIAG] Dropped breakdown: { {k: len(v) for k, v in dropped.items()} }\n")
                except Exception as err:
                    print(f"[DIAG] Could not read report: {err}")

            print(f"[DOCUMENT] Document Processed: {document.source_path}")
            print(f"   - Raw lines:          {cleaned.original_line_count}")
            print(f"   - Noise lines removed:{cleaned.removed_line_count}")
            print(f"   - Cleaned lines:      {cleaned.original_line_count - cleaned.removed_line_count}\n")

            # ── 2. Concept Learning & Knowledge Graph Sync ──────────
            memory_store = ConceptMemoryStore()
            learner      = Learner(memory_store=memory_store)
            concepts     = learner.learn(cleaned)

            print(f"[GENOME] Concepts Extracted & Synced to Memory: {len(concepts)}")
            stats = memory_store.stats()
            print(f"   - Total Concepts in Memory Graph: {stats['total_concepts']}\n")

            # Save to cache for next run
            _save_cached_concepts(file_path, concepts, stats)

    accepted: List[GeneratedQuestion]             = []
    rejected: List[Tuple[GeneratedQuestion, str]] = []

    # ── 3a. TURBO — Question Only, No Answer, No LLM Critic ─
    if mode == "turbo":
        print("[TURBO] Question-Only Generation (No Ideal Answer · No Critic)...")
        print(f"[TURBO] Generating {max_concepts} questions in parallel...\n")

        # Shuffle concepts so each run generates from different parts of the material
        # Use time-based seed so it is different every run
        random.seed(int(time.time()))
        target_concepts = random.sample(
            concepts[:max_concepts * 2],          # pool: 2x the required count
            min(max_concepts, len(concepts))      # pick exactly max_concepts
        )

        def _generate_one(args):
            idx, concept = args
            try:
                gq = generate_turbo(concept, marks=5)
                return idx, gq, None
            except Exception as e:
                return idx, None, str(e)

        results = {}
        # Use min(max_concepts, 4) workers (safe default for Ollama)
        num_workers = min(len(target_concepts), 4) if len(target_concepts) > 0 else 1

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_generate_one, (idx, concept)): idx
                for idx, concept in enumerate(target_concepts, 1)
            }
            for future in as_completed(futures):
                idx, gq, error = future.result()
                results[idx] = (gq, error)

        # Process results in order
        for idx in sorted(results.keys()):
            gq, error = results[idx]
            concept = target_concepts[idx-1]
            chunk = getattr(concept, "content", "") or getattr(concept, "text", "") or ""

            if error or gq is None:
                print(f"  [Q{idx:02d}] [ERROR] {error}")
                continue

            question_text = gq.question_text
            q = _clean_turbo(question_text)
            gq.question_text = q

            if _cheap_validate(q) and _grounded(q, chunk):
                accepted.append(gq)
                wrapped = _wrap(q)
                print(f"  [Q{idx:02d}] [OK] {wrapped}\n")
            else:
                rejected.append((gq, "cheap_validate_failed"))
                wrapped = _wrap(q)
                print(f"  [Q{idx:02d}] [FAIL] Failed validation -\n          {wrapped}\n")

    # ── 3b. BALANCED / DEEP — Full RAG² + LLM Critic ────────
    else:
        print(f"[RAG2] Running RAG^2 Answer-First Generation & Self-Critic Gate ({mode.upper()} Mode)...")
        for concept in concepts[:max_concepts]:
            question     = generate(concept, mode=mode)
            ok, reason   = review(question)

            if ok:
                accepted.append(question)
            else:
                rejected.append((question, reason))

    # ── 4. Audit Report ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[AUDIT] PIPELINE AUDIT REPORT")
    print(f"{'='*60}")
    print(f"[OK]       Accepted Questions : {len(accepted)}")
    print(f"[REJECTED] Rejected Questions : {len(rejected)}\n")

    if accepted:
        print("-" * 60)
        print("ACCEPTED QUESTIONS SAMPLE:")
        print("-" * 60)
        for i, q in enumerate(accepted[:3], 1):
            ans_preview = (
                "[Not generated in Turbo Mode]"
                if q.ideal_answer is None
                else (q.ideal_answer[:120] + "...")
            )
            print(f"Q{i} [{q.marks} Marks | Bloom Level: {q.bloom_level}]:")
            print(f"   {q.question_text}")
            print(f"   Ideal Answer: {ans_preview}\n")

    if rejected:
        print("-" * 60)
        print("REJECTED QUESTIONS REASON CODES:")
        print("-" * 60)
        for q, reason in rejected[:3]:
            print(f"[FAIL] [{reason}]")
            print(f"   Question: {q.question_text}\n")

    return accepted, rejected


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _marks_for_concept(concept) -> int:
    """Derive marks from concept metadata, defaulting to 5."""
    return getattr(concept, "marks", None) or 5


def _bloom_for_concept(concept) -> int:
    """Derive Bloom level from concept metadata, defaulting to 2."""
    return getattr(concept, "bloom_dna", None) or getattr(concept, "bloom_level", None) or 2


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sample_file = sys.argv[1] if len(sys.argv) > 1 else "v0_1/sample_lecture.txt"
    sample_path = Path(sample_file)

    if not sample_path.exists():
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(
            "Artificial Intelligence is defined as the simulation of human intelligence "
            "processes by machines, especially computer systems.\n"
            "Machine Learning is a subset of artificial intelligence that provides systems "
            "the ability to automatically learn and improve from experience.\n"
            "Deep Learning is a specialized subfield of machine learning based on artificial "
            "neural networks with representation learning.\n",
            encoding="utf-8",
        )
        print(f"Created sample input file at: {sample_file}\n")

    run_pipeline(str(sample_path), mode="turbo")
