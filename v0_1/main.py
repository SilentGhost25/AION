"""
AION Module: Pipeline Orchestrator
Maturity:    v0.1 — PURE SEQUENTIAL PIPELINE
Upgrades to: Async EventBus Orchestrator with Fail-Safe Context Expansion Loops
Contract:    Run pure pipeline matching AION Architecture Diagram.
"""

from pathlib import Path
from typing import List, Tuple
from .uploader import upload
from .extractor import extract
from .cleaner import clean
from .memory import ConceptMemoryStore
from .learner import Learner
from .generator import generate
from .critic import review
from .schemas import GeneratedQuestion


def run_pipeline(file_path: str, max_concepts: int = 10, mode: str = "balanced") -> Tuple[List[GeneratedQuestion], List[Tuple[GeneratedQuestion, str]]]:
    """
    Executes the end-to-end AION v0.1 ingestion & generation pipeline.
    Pure pipeline flow matching the architecture diagram.

    mode: "turbo" (5M, 150w), "balanced" (10M, 250-300w), or "deep" (20M, 400-500w)
    """
    print(f"============================================================")
    print(f"[START] AION v0.1 Ingestion & Generation Pipeline starting (Mode: {mode.upper()})...")
    print(f"============================================================\n")

    # 1. Ingestion & Preprocessing
    validated_path = upload(file_path)
    document = extract(validated_path)               # Upload -> Extract
    cleaned = clean(document)                         # Clean

    # ── DIAGNOSTIC: show which pages were kept/dropped ──
    import json
    report_path = Path("extracted_output") / "last_report.json"
    if report_path.exists():
        try:
            rpt = json.loads(report_path.read_text(encoding="utf-8"))
            print(f"[DIAG] Method used: {rpt.get('method')}")
            print(f"[DIAG] TOC found: {rpt.get('toc_found')}")
            print(f"[DIAG] TOC entries: {len(rpt.get('toc_entries', []))}")
            if rpt.get('toc_entries'):
                print(f"[DIAG] First 5 TOC entries:")
                for e in rpt['toc_entries'][:5]:
                    print(f"         {e}")
            print(f"[DIAG] Kept pages ({len(rpt.get('kept_pages', []))} total): {rpt.get('kept_pages', [])[:10]}...")
            print(f"[DIAG] Dropped breakdown: { {k: len(v) for k,v in rpt.get('dropped_pages',{}).items()} }\n")
        except Exception as err:
            print(f"[DIAG] Could not read report: {err}")

    print(f"[DOCUMENT] Document Processed: {document.source_path}")
    print(f"   - Raw lines: {cleaned.original_line_count}")
    print(f"   - Noise lines removed: {cleaned.removed_line_count}")
    print(f"   - Cleaned lines: {cleaned.original_line_count - cleaned.removed_line_count}\n")

    # 2. Concept Learning & Knowledge Evolution Graph Sync
    memory_store = ConceptMemoryStore()
    learner = Learner(memory_store=memory_store)
    concepts = learner.learn(cleaned)                 # Concept Extraction & Memory Upsert

    print(f"[GENOME] Concepts Extracted & Synced to Memory: {len(concepts)}")
    print(f"   - Total Concepts in Memory Graph: {memory_store.stats()['total_concepts']}\n")

    accepted: List[GeneratedQuestion] = []
    rejected: List[Tuple[GeneratedQuestion, str]] = []

    # 3. RAG^2 Generation & Self-Critic Review
    print(f"[RAG2] Running RAG^2 Answer-First Generation & Self-Critic Gate ({mode.upper()} Mode)...")
    for concept in concepts[:max_concepts]:
        question = generate(concept, mode=mode)        # Ideal Answer -> Exam Question (RAG^2)
        ok, reason = review(question)                 # Self-Critic Gate

        if ok:
            accepted.append(question)
        else:
            rejected.append((question, reason))

    # 4. Reporting & Summary Audit
    print(f"\n============================================================")
    print(f"[AUDIT] PIPELINE AUDIT REPORT")
    print(f"============================================================")
    print(f"[OK] Accepted Questions: {len(accepted)}")
    print(f"[REJECTED] Rejected Questions: {len(rejected)}\n")

    if accepted:
        print("-" * 60)
        print("ACCEPTED QUESTIONS SAMPLE:")
        print("-" * 60)
        for i, q in enumerate(accepted[:3], 1):
            print(f"Q{i} [{q.marks} Marks | Bloom Level: {q.bloom_level}]: {q.question_text}")
            print(f"   Ideal Answer: {q.ideal_answer[:120]}...\n")

    if rejected:
        print("-" * 60)
        print("REJECTED QUESTIONS REASON CODES:")
        print("-" * 60)
        for q, reason in rejected[:3]:
            print(f"[FAIL] [{reason}] -> Question Draft: {q.question_text[:60]}...")

    return accepted, rejected


if __name__ == "__main__":
    import sys
    # Default test file or user provided argument
    sample_file = sys.argv[1] if len(sys.argv) > 1 else "v0_1/sample_lecture.txt"

    # Create sample lecture note if missing
    sample_path = Path(sample_file)
    if not sample_path.exists():
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(
            "Artificial Intelligence is defined as the simulation of human intelligence processes by machines, especially computer systems.\n"
            "Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.\n"
            "Deep Learning is a specialized subfield of machine learning based on artificial neural networks with representation learning.\n"
            "Supervised Learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs.\n"
            "Reinforcement Learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.\n",
            encoding="utf-8"
        )
        print(f"Created sample input file at: {sample_file}\n")

    run_pipeline(str(sample_path))
