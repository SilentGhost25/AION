"""
AION Module: Pipeline Orchestrator
Maturity:    v0.1 — MODULE-BY-MODULE PARALLEL EXAM ORCHESTRATOR
Integrated:  Custom Model, Difficulty System, Formula Extractor, Visual RAG
"""

import re
import json
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .uploader import upload
from .extractor import extract
from .cleaner import clean
from .memory import ConceptMemoryStore
from .learner import Learner
from .schemas import GeneratedQuestion
from .segmenter import segment_document, ModuleSegment
from .content_validator import validate_content, validate_chunks
from .paper_validator import enforce_or_parity
from .generator import (
    get_vtu_vibe_question,
    _is_valid,
    IA_PARTITIONS,
    SEE_PARTITIONS,
    get_bloom_level_name
)

from .difficulty import DifficultyManager, DifficultyLevel

# Universal Academic Pipeline (new, grounded) — import is optional for backward compat
try:
    from core.pipeline.aion_pipeline import AionUniversalPipeline
    HAS_UNIFIED = True
except ImportError:
    HAS_UNIFIED = False
from .visual import (
    safe_build_planner,
    VisualQuestionGenerator,
    ModuleVisualPlanner,
    FigureRegistry,
    _build_module_map
)
from .chunk_image_mapper import (
    ChunkImageMapper,
    QuestionImageSelector,
    split_module_into_chunks,
    TextChunk
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

def run_unified_pipeline(
    file_path:      str,
    exam_type:      str  = "see",
    difficulty:     str  = "mixed",
    num_questions:  int  = 8,
    use_llm:        bool = True,
    allow_external: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """
    Universal Academic Pipeline (AION Development Context):
      Upload → Extract → Understand → Build Concept Graph → Ground
           → Reason → Plan → Compose → Audit → Output
    Every question grounded: Concept ID | Source chunk | Confidence | Expected answer | Bloom | Question
    Stateless, pluggable, hallucination-resistant.

    Returns (accepted, rejected) as dicts compatible with legacy run_pipeline.
    """
    if not HAS_UNIFIED:
        raise ImportError("AionUniversalPipeline not available — check core/pipeline/aion_pipeline.py")

    pipeline = AionUniversalPipeline(
        use_llm=use_llm,
        allow_external=allow_external,
        exam_type=exam_type,
        difficulty=difficulty,
    )
    result = pipeline.run(
        source_path=file_path,
        num_questions=num_questions,
    )
    # Convert to legacy dict format for Flask / CLI
    accepted = []
    for q in result.accepted:
        accepted.append({
            "question_text": q.question_text,
            "concept_id": q.concept_id,
            "source_hash": q.source_hash,
            "marks": q.marks,
            "bloom_level": q.bloom_level,
            "bloom_label": q.bloom_label,
            "expected_answer": q.expected_answer,
            "question_type": q.question_type,
            "grounding": q.grounding,
            "confidence": q.confidence,
            "plan_id": q.plan_id,
        })
    rejected = []
    for q, rep in result.rejected:
        rejected.append({
            "question_text": q.question_text,
            "concept_id": q.concept_id,
            "reason_codes": rep.reason_codes,
            "overall_score": rep.overall_score,
            "gates": [g.__dict__ for g in rep.gates],
        })
    # Also expose full result via side-file for debugging
    import json
    from pathlib import Path as _P
    out = _P("extracted_output") / "unified_pipeline_report.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
    return accepted, rejected


def run_pipeline(
    file_path:          str,
    max_concepts:       int  = 10,
    mode:               str  = "turbo",
    exam_type:          str  = "see",
    difficulty:         str  = "mixed",
    include_visual:     bool = True,
    use_unified:        bool = False,
    request_contract:   Optional[Any] = None,
    pipeline_trace:     Optional[Any] = None,
    sub_question_count: Optional[int] = None,  # 1, 2, or 3 — user-specified
) -> Tuple[List[dict], List[dict]]:
    """
    Saves and generates an aligned VTU Question Paper grouped strictly by Module.
    Generates exactly 4 main questions per module.
    Sub-questions per main question are strictly capped to max 3.
    """
    from core.validators.academic_validator import validate_academic_quality

    t_start = time.time()
    if pipeline_trace:
        pipeline_trace.stage("PipelineStart", status="PASS", metrics={"file": Path(file_path).name, "exam": exam_type})

    # ── Unified pipeline delegate (grounded) ─────────────────
    if use_unified and HAS_UNIFIED:
        print("[PIPELINE] Delegating to Universal Academic Pipeline (grounded, hallucination-resistant)")
        return run_unified_pipeline(
            file_path=file_path,
            exam_type=exam_type,
            difficulty=difficulty,
            num_questions=max(4, max_concepts),
        )

    print("=" * 60)
    print(f"[START] AION Exam Generation Pipeline ({exam_type.upper()} Exam Mode)...")
    print(f"[CONFIG] Difficulty: {difficulty.upper()} | Visual RAG: {include_visual}")
    print("=" * 60 + "\n")

    diff_manager = DifficultyManager.from_string(difficulty)

    # 1. Ingestion & Segmentation
    t0 = time.time()
    cached_modules = _load_cached_modules(file_path)
    if cached_modules is not None:
        modules = cached_modules
        if pipeline_trace:
            pipeline_trace.stage("Extraction", status="PASS", duration_ms=(time.time()-t0)*1000, metrics={"cached": True, "modules": len(modules)})
    else:
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

                    # Modular Academic Validation Gate
                    acad_res = validate_academic_quality(content)
                    if pipeline_trace:
                        pipeline_trace.stage(
                            f"AcademicValidator:{file_item.stem}",
                            status = "PASS" if acad_res.valid else "WARN",
                            metrics = {"score": acad_res.academic_score, "noise": acad_res.noise_score},
                            message = acad_res.rejection_reason if not acad_res.valid else "Clean academic text"
                        )

                    words = len(content.split())
                    modules.append(ModuleSegment(title=file_item.stem, content=content, word_count=words))
                except Exception as e:
                    print(f"  [ERROR] Failed to process {file_item.name}: {e}")
        else:
            raw_document = extract(validated_path)
            content = raw_document.raw_text

            # Modular Academic Validation Gate
            acad_res = validate_academic_quality(content)
            if pipeline_trace:
                pipeline_trace.stage(
                    "AcademicValidator",
                    status = "PASS" if acad_res.valid else "WARN",
                    metrics = {"score": acad_res.academic_score, "noise": acad_res.noise_score},
                    message = acad_res.rejection_reason if not acad_res.valid else "Clean academic text"
                )

            seg_result = segment_document(content, file_path=validated_path)
            modules = seg_result.segments

        _save_cached_modules(file_path, modules)
        if pipeline_trace:
            pipeline_trace.stage("Extraction", status="PASS", duration_ms=(time.time()-t0)*1000, metrics={"cached": False, "modules": len(modules)})

    print(f"[SEGMENTER] Identified {len(modules)} Modules/Chapters in source material.")

    # 2. Extract Visual Figures & Build Proximity Chunk Map
    mapper   = None
    selector = None

    if include_visual:
        try:
            doc_id   = FigureRegistry.make_document_id(file_path)
            print("[VISUAL] Extracting figures (fast proximity mode)...")

            # Extract figures separately (no VLM blocking)
            from .visual.figure_extractor import extract_figures
            figures = extract_figures(
                file_path, 
                doc_id=doc_id,
                module_map=_build_module_map(modules),
            )

            # Mark all figures eligible by default (no VLM filtering)
            for fig in figures:
                fig.eligible = True

            class MockRegistry:
                def __init__(self, figs):
                    self.figs = figs
                def eligible_cards(self):
                    return self.figs

            # Build chunk-image map with mock registry
            mapper = ChunkImageMapper(
                registry        = MockRegistry(figures),
                total_pages     = 200,
                page_tolerance  = 3,
            )
            mapper.build(modules)
            selector = QuestionImageSelector(mapper)

            s = mapper.summary()
            print(
                f"[MAPPER] Coverage: "
                f"{s['chunks_with_images']}/{s['total_chunks']} chunks "
                f"({s['image_coverage_pct']}%)"
            )

        except Exception as e:
            import traceback
            print(f"[MAPPER] Setup failed: {e}")
            traceback.print_exc()
            mapper   = None
            selector = None

    # Validate partitions — filter by user-specified sub-question count if provided
    target_marks   = 20 if exam_type.lower() == "see" else 10
    raw_partitions = SEE_PARTITIONS if exam_type.lower() == "see" else IA_PARTITIONS
    base_partitions = [p for p in raw_partitions if len(p) <= 3 and sum(p) == target_marks]

    if sub_question_count and sub_question_count in (1, 2, 3):
        filtered = [p for p in base_partitions if len(p) == sub_question_count]
        target_partitions = filtered if filtered else base_partitions
        print(f"[PIPELINE] Sub-question count locked to {sub_question_count} "
              f"({len(target_partitions)} matching partitions available)")
    else:
        target_partitions = base_partitions

    if not target_partitions:
        target_partitions = [[10, 10]] if target_marks == 20 else [[5, 5]]

    output_paper = []
    executor = ThreadPoolExecutor(max_workers=1)

    for mod_idx, mod in enumerate(modules, 1):
        module_id = f"module_{mod_idx}"
        print(f"\n[MODULE {mod_idx}] Processing: '{mod.title}' ({mod.word_count} words)")

        if mod.word_count < 10:
            print(f"[MODULE {mod_idx}] Skipping — too short")
            continue

        if mapper:
            module_chunks = mapper.get_chunks_for_module(module_id)
        else:
            module_chunks = []

        if not module_chunks:
            raw_splits = [
                c.strip()
                for c in re.split(r"\n\n+", mod.content)
                if len(c.split()) > 20
            ]
            module_chunks_text = raw_splits or [mod.content]
        else:
            module_chunks_text = [c.text for c in module_chunks]

        # Content Validation Gate — Filter corrupted/noisy chunks
        valid_chunks, rejected_chunks, avg_conf = validate_chunks(module_chunks_text)
        if valid_chunks:
            module_chunks_text = valid_chunks
        else:
            print(f"[VALIDATOR] Warning: All chunks rejected in module '{mod.title}'. Using sanitized text.")
            module_chunks_text = [validate_content(c).clean_text or c for c in module_chunks_text if c.strip()]

        pair1_bloom = random.choice([2, 3])
        pair2_bloom = random.choice([4, 5])
        bloom_levels = [pair1_bloom, pair1_bloom, pair2_bloom, pair2_bloom]

        used_chunk_ids: set[str] = set()
        module_questions         = []
        futures = []

        for mq_idx in range(1, 5):
            bloom     = bloom_levels[mq_idx - 1]
            partition = random.choice(target_partitions)

            if mapper and module_chunks:
                prefer_img = (mq_idx == 1)
                best_tc    = mapper.get_best_chunk_for_question(
                    module_id      = module_id,
                    prefer_image   = prefer_img,
                    used_chunk_ids = used_chunk_ids,
                )
                if best_tc:
                    used_chunk_ids.add(best_tc.id)
                    selected_chunks = [best_tc.text] * len(partition)
                    best_chunk_obj  = best_tc
                else:
                    selected_chunks = [random.choice(module_chunks_text)] * len(partition)
                    best_chunk_obj  = None
            else:
                selected_chunks = random.sample(
                    module_chunks_text,
                    min(len(partition), len(module_chunks_text))
                )
                while len(selected_chunks) < len(partition):
                    selected_chunks.append(random.choice(module_chunks_text))
                best_chunk_obj = None

            futures.append(
                executor.submit(
                    _generate_main_question,
                    mq_idx, partition, bloom, selected_chunks, target_marks,
                    diff_manager, best_chunk_obj, selector, module_id
                )
            )

        for fut in as_completed(futures):
            res = fut.result()
            module_questions.append(res)

        module_questions.sort(key=lambda x: x["mq_index"])

        # Enforce OR pair parity (Q1 vs Q2, Q3 vs Q4)
        if len(module_questions) >= 2:
            module_questions[0], module_questions[1] = enforce_or_parity(
                module_questions[0], module_questions[1], target_marks
            )
        if len(module_questions) >= 4:
            module_questions[2], module_questions[3] = enforce_or_parity(
                module_questions[2], module_questions[3], target_marks
            )

        output_paper.append({
            "module_index": mod_idx,
            "module_title": mod.title,
            "questions": module_questions
        })

    executor.shutdown()

    if mapper:
        print(f"\n[MAPPER] Final: {mapper.summary()}")

    _print_exam_paper(output_paper, exam_type.upper())

    qa_report = {}
    try:
        from .qa_engine import QPGeneratorWithQA
        qa_manager = QPGeneratorWithQA()
        qa_report  = qa_manager.run_full_paper_qa(output_paper)
        print(f"\n[QA ENGINE] Completed Paper QA Check | Quality Score: {qa_report['quality_score']}/100 | Total Issues: {qa_report['total_issues_found']}")
    except Exception as e:
        print(f"[QA ENGINE] Warning running paper QA check: {e}")

    return output_paper, qa_report


def _generate_main_question(
    mq_idx:         int,
    partition:      List[int],
    bloom:          int,
    chunks:         List[str],
    total_marks:    int,
    diff_manager:   DifficultyManager = None,
    chunk_obj:      TextChunk = None,
    selector:       QuestionImageSelector = None,
    module_id:      str  = "",
) -> dict:
    """Worker function to build a main question with max 3 subquestions."""
    dm = diff_manager or DifficultyManager.from_string("mixed")

    # HARD CLAMP: never more than 3 sub-questions
    if len(partition) > 3:
        partition = sorted(partition, reverse=True)
        while len(partition) > 3:
            smallest      = partition.pop()
            partition[-1] += smallest
        partition = sorted(partition, reverse=True)

    sub_questions = []
    sub_letters   = ["a", "b", "c"]

    for idx, marks in enumerate(partition):
        chunk      = chunks[idx % len(chunks)]
        difficulty = dm.assign_difficulty(idx, len(partition), marks)
        sub_bloom  = dm.get_bloom_for_difficulty(difficulty)

        image_data = None
        if selector and chunk_obj and idx == 0:
            try:
                image_data = selector.select(
                    chunk     = chunk_obj,
                    module_id = module_id,
                    sub_index = idx,
                )
            except Exception as e:
                print(f"[SELECTOR] Error: {e}")
                image_data = None

        q_text = get_vtu_vibe_question(
            chunk        = chunk,
            marks        = marks,
            bloom        = sub_bloom,
            difficulty   = difficulty,
            diff_manager = dm,
        )
        retry = 0
        while not _is_valid(q_text) and retry < 2:
            q_text = get_vtu_vibe_question(
                chunk        = random.choice(chunks),
                marks        = marks,
                bloom        = sub_bloom,
                difficulty   = difficulty,
                diff_manager = dm,
            )
            retry += 1

        if image_data:
            if not re.search(
                r"\b(figure|diagram|given|shown|refer|image|chart)\b",
                q_text, re.I
            ):
                q_text = (
                    "With reference to the given figure, "
                    + q_text[0].lower()
                    + q_text[1:]
                )

        sub_questions.append({
            "letter":     sub_letters[idx] if len(partition) > 1 else None,
            "text":       q_text,
            "marks":      marks,
            "difficulty": difficulty,
            "bloom":      sub_bloom,
            "image":      image_data,
        })

    actual_total = sum(sq["marks"] for sq in sub_questions)

    # For a single sub-question, report the bloom of that sub-question
    # (the one that drove the verb selection), not the module-level placeholder.
    reported_bloom = sub_questions[0]["bloom"] if len(sub_questions) == 1 else bloom

    return {
        "mq_index":      mq_idx,
        "bloom_level":   reported_bloom,
        "bloom_name":    get_bloom_level_name(reported_bloom),
        "total_marks":   actual_total,
        "sub_questions": sub_questions,
    }


def _print_exam_paper(paper: List[dict], exam_type: str):
    """Prints the final generated VTU paper structured perfectly."""
    print("\n" + "="*80)
    print(f"                     VTU {exam_type} QUESTION PAPER")
    print("="*80)

    for mod in paper:
        print(f"\nMODULE {mod['module_index']}: {mod['module_title'].upper()}")
        print("-" * 80)

        _print_mq(mod["questions"][0])
        print(f"{' '*36}[OR]")
        _print_mq(mod["questions"][1])

        print("\n" + "· " * 40 + "\n")

        _print_mq(mod["questions"][2])
        print(f"{' '*36}[OR]")
        _print_mq(mod["questions"][3])
        print()


def _print_mq(mq: dict):
    prefix = f"Q{mq['mq_index']} "
    bloom_tag = f" [Bloom Level {mq['bloom_level']}: {mq['bloom_name']}]"

    if len(mq["sub_questions"]) == 1:
        sq = mq["sub_questions"][0]
        diff_tag = f"[{sq.get('difficulty', '?').upper()}]"
        img_tag  = " [IMG]" if sq.get("image") else ""
        print(f"{prefix}{sq['text']} ({sq['marks']} Marks) {diff_tag}{img_tag}{bloom_tag}")
    else:
        print(f"{prefix}Answer the following:{bloom_tag}")
        for sq in mq["sub_questions"]:
            diff_tag = f"[{sq.get('difficulty', '?').upper()}]"
            img_tag  = " [IMG]" if sq.get("image") else ""
            print(
                f"   ({sq['letter']}) {sq['text']} "
                f"({sq['marks']} Marks) {diff_tag}{img_tag}"
            )
