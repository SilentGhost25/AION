"""
AION Module: Pipeline Orchestrator
Maturity:    v0.1 — MODULE-BY-MODULE PARALLEL EXAM ORCHESTRATOR
"""

import re
import json
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .uploader import upload
from .extractor import extract
from .cleaner import clean
from .memory import ConceptMemoryStore
from .learner import Learner
from .schemas import GeneratedQuestion
from .segmenter import segment_document, ModuleSegment
from .generator import (
    get_vtu_vibe_question, 
    _is_valid_vtu_question, 
    IA_PARTITIONS, 
    SEE_PARTITIONS, 
    get_bloom_level_name
)

# ─────────────────────────────────────────────────────────────
# Modules Caching
# ─────────────────────────────────────────────────────────────
CACHE_DIR = Path("extracted_output") / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _modules_cache_key(file_path: str) -> str:
    import hashlib
    p = Path(file_path)
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

def _load_cached_modules(file_path: str):
    import pickle
    try:
        key   = _modules_cache_key(file_path)
        cache = CACHE_DIR / f"modules_{key}.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                data = pickle.load(f)
            print(f"[CACHE] Loaded {len(data['modules'])} segmented modules from cache (skipping ingestion)")
            return data["modules"]
    except Exception:
        pass
    return None

def _save_cached_modules(file_path: str, modules):
    import pickle
    try:
        key   = _modules_cache_key(file_path)
        cache = CACHE_DIR / f"modules_{key}.pkl"
        with open(cache, "wb") as f:
            pickle.dump({"modules": modules}, f)
        print(f"[CACHE] Saved {len(modules)} segmented modules to cache")
    except Exception as e:
        print(f"[CACHE] Could not save modules cache: {e}")

# ─────────────────────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    file_path: str, 
    max_concepts: int = 10, 
    mode: str = "turbo", 
    exam_type: str = "see"
) -> Tuple[List[dict], List[dict]]:
    """
    Saves and generates an aligned VTU Question Paper grouped strictly by Module.
    
    Generates exactly 4 main questions per module:
      - MQ1 & MQ2: Paired choice at same Bloom Level
      - MQ3 & MQ4: Paired choice at same higher Bloom Level
    """
    print("=" * 60)
    print(f"[START] AION Exam Generation Pipeline ({exam_type.upper()} Exam Mode)...")
    print("=" * 60 + "\n")

    # Try cache first
    cached_modules = _load_cached_modules(file_path)
    if cached_modules is not None:
        modules = cached_modules
    else:
        # Ingestion & Segmentation
        validated_path = upload(file_path)
        p = Path(validated_path)
        
        modules = []
        if p.is_dir():
            suffixes = {".pdf", ".txt", ".md"}
            files = sorted(
                [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in suffixes and not f.name.startswith(".")],
                key=lambda f: f.name
            )
            if not files:
                raise FileNotFoundError(f"No PDF, TXT, or MD files found in directory: {file_path}")
            
            print(f"[PIPELINE] Processing directory: {file_path} ({len(files)} files found)")
            for file_item in files:
                print(f"[PIPELINE] Ingesting file: {file_item.name} ...")
                try:
                    val_file_path = upload(str(file_item))
                    doc = extract(val_file_path)
                    content = doc.raw_text.strip()
                    words = len(content.split())
                    modules.append(ModuleSegment(title=file_item.stem, content=content, word_count=words))
                except Exception as e:
                    print(f"  [ERROR] Failed to process {file_item.name}: {e}")
        else:
            raw_document = extract(validated_path)
            # Segment document into Chapters/Modules
            seg_result = segment_document(raw_document.raw_text, file_path=validated_path)
            modules = seg_result.segments

        # Save to cache
        _save_cached_modules(file_path, modules)

    print(f"[SEGMENTER] Identified {len(modules)} Modules/Chapters in source material.")

    target_partitions = SEE_PARTITIONS if exam_type.lower() == "see" else IA_PARTITIONS
    target_marks = 20 if exam_type.lower() == "see" else 10

    output_paper = []
    
    # Thread pool for ultra-fast parallel generation
    executor = ThreadPoolExecutor(max_workers=6)

    for mod_idx, mod in enumerate(modules, 1):
        print(f"\n[MODULE {mod_idx}] Processing: '{mod.title}' ({mod.word_count} words)")

        if mod.word_count < 10:
            print(f"[MODULE {mod_idx}] [WARNING] Skipping — insufficient content ({mod.word_count} words)")
            continue

        preview = mod.content[:200].replace('\n', ' ')
        print(f"[MODULE {mod_idx}] Content preview: {preview}...")

        # Extract small chunks/concepts from this module
        mod_chunks = [c.strip() for c in re.split(r"\n\n+", mod.content) if len(c.split()) > 30]
        if len(mod_chunks) < 4:
            # Fallback split
            mod_chunks = [mod.content[i:i+1500] for i in range(0, len(mod.content), 1200)]

        # Determine Bloom pairs for the 4 questions
        # Pair 1: Bloom Level 2 (Understand) or 3 (Apply)
        # Pair 2: Bloom Level 4 (Analyze) or 5 (Evaluate)
        pair1_bloom = random.choice([2, 3])
        pair2_bloom = random.choice([4, 5])
        bloom_levels = [pair1_bloom, pair1_bloom, pair2_bloom, pair2_bloom]

        module_questions = []

        # Threaded task submission
        futures = []
        for mq_idx in range(1, 5):
            bloom = bloom_levels[mq_idx - 1]
            partition = random.choice(target_partitions)
            
            # Select unique chunks for subquestions to avoid repetition
            selected_chunks = random.sample(mod_chunks, min(len(partition), len(mod_chunks)))
            if len(selected_chunks) < len(partition):
                selected_chunks += [random.choice(mod_chunks) for _ in range(len(partition) - len(selected_chunks))]

            futures.append(
                executor.submit(
                    _generate_main_question, 
                    mq_idx, partition, bloom, selected_chunks, target_marks
                )
            )

        for fut in as_completed(futures):
            res = fut.result()
            module_questions.append(res)

        # Sort questions MQ1 to MQ4
        module_questions.sort(key=lambda x: x["mq_index"])
        output_paper.append({
            "module_index": mod_idx,
            "module_title": mod.title,
            "questions": module_questions
        })

    executor.shutdown()

    # Print Beautiful Academic Output
    _print_exam_paper(output_paper, exam_type.upper())

    return output_paper, []

def _generate_main_question(
    mq_idx:       int, 
    partition:    List[int], 
    bloom:        int, 
    chunks:       List[str], 
    total_marks:  int
) -> dict:
    """Worker function to build a main question with diverse subquestions."""
    sub_questions = []
    sub_letters   = ["a", "b", "c", "d", "e"]
    used_verbs    = set()

    for idx, marks in enumerate(partition):
        chunk  = chunks[idx % len(chunks)]
        q_text = get_vtu_vibe_question(
            chunk, marks, bloom, _used_verbs=used_verbs
        )
        
        first_word = q_text.split()[0] if q_text else ""
        used_verbs.add(first_word)

        # Validation retry
        retry = 0
        while not _is_valid_vtu_question(q_text) and retry < 2:
            alt_chunk = random.choice(chunks)
            q_text    = get_vtu_vibe_question(
                alt_chunk, marks, bloom, _used_verbs=used_verbs
            )
            retry += 1

        sub_questions.append({
            "letter": sub_letters[idx] if len(partition) > 1 else None,
            "text":   q_text,
            "marks":  marks
        })

    return {
        "mq_index":    mq_idx,
        "bloom_level": bloom,
        "bloom_name":  get_bloom_level_name(bloom),
        "total_marks": total_marks,
        "sub_questions": sub_questions
    }

def _print_exam_paper(paper: List[dict], exam_type: str):
    """Prints the final generated VTU paper structured perfectly."""
    print("\n" + "="*80)
    print(f"                     VTU {exam_type} QUESTION PAPER")
    print("="*80)

    for mod in paper:
        print(f"\nMODULE {mod['module_index']}: {mod['module_title'].upper()}")
        print("-" * 80)

        # Group into MQ1 vs MQ2 (OR) and MQ3 vs MQ4 (OR)
        # MQ1 & MQ2 are Choice Pair 1
        # MQ3 & MQ4 are Choice Pair 2
        
        # Choice 1
        _print_mq(mod["questions"][0])
        print(f"{' '*36}[OR]")
        _print_mq(mod["questions"][1])
        
        print("\n" + "· " * 40 + "\n")
        
        # Choice 2
        _print_mq(mod["questions"][2])
        print(f"{' '*36}[OR]")
        _print_mq(mod["questions"][3])
        print()

def _print_mq(mq: dict):
    prefix = f"Q{mq['mq_index']} "
    bloom_tag = f" [Bloom Level {mq['bloom_level']}: {mq['bloom_name']}]"
    
    if len(mq["sub_questions"]) == 1:
        sq = mq["sub_questions"][0]
        print(f"{prefix}{sq['text']} ({sq['marks']} Marks){bloom_tag}")
    else:
        print(f"{prefix}Answer the following subquestions:{bloom_tag}")
        for sq in mq["sub_questions"]:
            print(f"   ({sq['letter']}) {sq['text']} ({sq['marks']} Marks)")
