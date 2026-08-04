"""
AION Minimal Pipeline — Emergency Mode
======================================
Stripped-down pipeline.
Sequential processing. One question at a time.
Memory-safe sentence chunking.
"""

import json
import time
import re
from pathlib import Path
from .single_request_llm import llm
from .emergency_config import EMERGENCY_CONFIG


def emergency_chunk(text: str, chunk_size: int = 300) -> list[str]:
    """
    Split text into small chunks at sentence boundaries.
    Prevents massive context windows.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        words = len(sent.split())

        if current_words + words > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0

        current.append(sent)
        current_words += words

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_minimal_prompt(chunk: str, bloom: int = 2, marks: int = 5) -> str:
    """
    Ultra-short prompt template under 500 words.
    """
    BLOOM_VERBS = {
        1: "Define",
        2: "Explain",
        3: "Apply",
        4: "Analyze",
        5: "Evaluate",
        6: "Design",
    }

    verb = BLOOM_VERBS.get(bloom, "Explain")
    chunk_truncated = " ".join(chunk.split()[:200])

    prompt = f"""{verb} one exam question based on:

{chunk_truncated}

Marks: {marks}
Output ONLY the question text."""

    return prompt


def generate_one_question(chunk: str, config: dict) -> dict:
    """
    Generate exactly one question.
    No retries. No fallback.
    Returns dict or None.
    """
    bloom = config.get("bloom_level", 2)
    marks = config.get("marks", 5)

    prompt = build_minimal_prompt(chunk, bloom, marks)

    response = llm.generate(
        prompt=prompt,
        max_tokens=EMERGENCY_CONFIG["max_output_tokens"]
    )

    if not response:
        print("[GEN] ✗ Failed to generate question", flush=True)
        return None

    question = response.strip()

    if len(question) < 15:
        print(f"[GEN] ✗ Question too short: {question}", flush=True)
        return None

    if len(question.split()) > marks * 15:
        print(f"[GEN] ⚠ Question too long, truncating", flush=True)
        question = " ".join(question.split()[:marks * 15])

    return {
        "question": question,
        "marks":    marks,
        "bloom":    bloom,
        "co":       f"CO{config.get('module', 1)}",
        "rbtl":     f"L{bloom}",
    }


def emergency_pipeline(pdf_path: str, n_questions: int = 5) -> dict:
    """
    Emergency mode pipeline for local execution.
    """
    print("\n" + "="*60, flush=True)
    print("EMERGENCY MODE — Local Only", flush=True)
    print("="*60, flush=True)

    if not llm.verify_model_loaded():
        raise RuntimeError(f"Model {llm.model} not available in Ollama")

    print(f"[EXTRACT] {Path(pdf_path).name}", flush=True)

    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    words = full_text.split()
    print(f"[EXTRACT] {len(words)} words, {len(doc)} pages", flush=True)

    chunks = emergency_chunk(
        full_text,
        chunk_size=EMERGENCY_CONFIG["chunk_size_words"]
    )

    print(f"[CHUNK] {len(chunks)} chunks created", flush=True)

    questions = []
    target = min(n_questions, len(chunks))

    print(f"[GEN] Generating {target} questions...", flush=True)

    for i in range(target):
        chunk = chunks[i]
        print(f"\n[Q{i+1}/{target}] ", end="", flush=True)

        config = {
            "bloom_level": (i % 3) + 2,  # Rotate L2-L4
            "marks":       5 if i % 2 == 0 else 10,
            "module":      (i % 5) + 1,
        }

        q = generate_one_question(chunk, config)

        if q:
            questions.append(q)
            print(f"✓ {q['question'][:60]}...", flush=True)
        else:
            print("✗ Failed", flush=True)

        time.sleep(2)

    print(f"\n[DONE] {len(questions)}/{target} questions generated", flush=True)

    return {
        "questions": questions,
        "metadata": {
            "source":      Path(pdf_path).name,
            "total_words": len(words),
            "chunks":      len(chunks),
            "generated":   len(questions),
            "mode":        "emergency",
            "model":       llm.model,
        }
    }
