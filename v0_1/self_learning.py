"""
AION Self-Learning Bridge
=========================
Connects the document ingestion pipeline to the ConceptMemoryStore.
Called after every successful document upload AND after every paper generation.

This makes AION genuinely improve with each upload:
- New concepts are extracted and stored
- Existing concepts gain confidence on re-encounter
- Covered topics are tracked to diversify future questions
- Generated questions are fed back as learned patterns
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


# -- Persistent learning store paths -------------------------------------------
LEARNING_DIR = Path("memory")
LEARNING_DIR.mkdir(parents=True, exist_ok=True)

CONCEPTS_PATH      = LEARNING_DIR / "concepts.json"
COVERAGE_PATH      = LEARNING_DIR / "topic_coverage.json"
QUESTIONS_PATH     = LEARNING_DIR / "generated_questions.json"
SUBJECT_STATS_PATH = LEARNING_DIR / "subject_stats.json"


# -- Coverage tracker -----------------------------------------------------------
def _load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _save_json(path: Path, data):
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[SELF-LEARNING] Could not save {path.name}: {e}")


# -- Main API -------------------------------------------------------------------

def learn_from_document(file_path: str, subject: str = "general", doc_id: str = "") -> dict:
    """
    Called after every document upload.
    Extracts concepts from the document and stores them in persistent memory.
    Returns a summary of what was learned.
    """
    try:
        from v0_1.learner import Learner
        from v0_1.schemas import CleanedDocument
        from v0_1.memory import ConceptMemoryStore

        # Read the document text
        p = Path(file_path)
        if not p.exists():
            return {"status": "skipped", "reason": "file not found"}

        # Extract text
        text = ""
        if p.suffix.lower() == ".pdf":
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
            doc = fitz.open(str(p))
            text = " ".join(page.get_text() for page in doc)
            doc.close()
        elif p.suffix.lower() in (".txt", ".md"):
            text = p.read_text(encoding="utf-8", errors="ignore")
        else:
            return {"status": "skipped", "reason": f"unsupported file type {p.suffix}"}

        if len(text.strip()) < 100:
            return {"status": "skipped", "reason": "insufficient text"}

        # Learn from the document
        doc_id = doc_id or hashlib.sha256(text[:1000].encode()).hexdigest()[:16]
        cleaned = CleanedDocument(
            doc_id=doc_id,
            clean_text=text,
            subject=subject,
        )

        store = ConceptMemoryStore(storage_path=str(CONCEPTS_PATH))
        learner = Learner(memory_store=store)
        concepts = learner.learn(cleaned)

        # Update subject stats
        stats = _load_json(SUBJECT_STATS_PATH, {})
        if subject not in stats:
            stats[subject] = {"uploads": 0, "concepts": 0, "questions_generated": 0}
        stats[subject]["uploads"] += 1
        stats[subject]["concepts"] += len(concepts)
        stats[subject]["last_upload"] = datetime.now().isoformat()
        _save_json(SUBJECT_STATS_PATH, stats)

        print(f"[SELF-LEARNING] Learned {len(concepts)} concepts from {p.name} (subject={subject})")
        return {
            "status": "ok",
            "concepts_learned": len(concepts),
            "total_stored": len(store.get_all()),
            "doc_id": doc_id,
        }

    except Exception as e:
        print(f"[SELF-LEARNING] learn_from_document failed: {e}")
        return {"status": "error", "reason": str(e)}


def learn_from_generated_paper(modules: list, subject: str = "general", exam_type: str = "IAT"):
    """
    Called after every successful paper generation.
    Records which topics were covered so future generations can diversify.
    Also stores generated question texts as learned patterns.
    """
    try:
        coverage = _load_json(COVERAGE_PATH, {})
        questions_db = _load_json(QUESTIONS_PATH, [])

        if subject not in coverage:
            coverage[subject] = {}

        new_questions = 0
        for mod in modules:
            mod_num = mod.get("module_index") or mod.get("moduleIndex") or mod.get("module_num", 1)
            mod_key = f"module_{mod_num}"
            if mod_key not in coverage[subject]:
                coverage[subject][mod_key] = {"times_covered": 0, "topics": []}

            coverage[subject][mod_key]["times_covered"] += 1
            coverage[subject][mod_key]["last_covered"] = datetime.now().isoformat()

            for q in mod.get("questions", []):
                sub_qs = q.get("subQuestions") or q.get("sub_questions") or []
                texts = [sq.get("text", "") for sq in sub_qs if sq.get("text")]
                if not texts:
                    texts = [q.get("text", "") or q.get("question_text", "")]

                for text in texts:
                    if text and len(text) > 20:
                        # Store as learned question pattern
                        q_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
                        questions_db.append({
                            "hash": q_hash,
                            "subject": subject,
                            "module": mod_num,
                            "text": text[:300],
                            "bloom": q.get("bloom") or q.get("bloomLevel", "L2"),
                            "co": q.get("co") or q.get("coMapping", "CO1"),
                            "marks": q.get("marks", 5),
                            "generated_at": datetime.now().isoformat(),
                        })
                        new_questions += 1

        # Keep last 500 questions to avoid unbounded growth
        questions_db = questions_db[-500:]

        _save_json(COVERAGE_PATH, coverage)
        _save_json(QUESTIONS_PATH, questions_db)

        print(f"[SELF-LEARNING] Recorded {new_questions} generated questions for subject={subject}")

    except Exception as e:
        print(f"[SELF-LEARNING] learn_from_generated_paper failed: {e}")


def get_covered_topics(subject: str, module_num: int) -> List[str]:
    """
    Returns list of topics already covered for this subject/module.
    Used by the question planner to avoid repeating topics.
    """
    try:
        coverage = _load_json(COVERAGE_PATH, {})
        mod_key = f"module_{module_num}"
        return coverage.get(subject, {}).get(mod_key, {}).get("topics", [])
    except Exception:
        return []


def get_learning_stats(subject: str = None) -> dict:
    """Returns learning statistics for display."""
    stats = _load_json(SUBJECT_STATS_PATH, {})
    concepts = _load_json(CONCEPTS_PATH, [])
    questions = _load_json(QUESTIONS_PATH, [])
    coverage = _load_json(COVERAGE_PATH, {})

    if subject:
        return {
            "subject": subject,
            "stats": stats.get(subject, {}),
            "total_concepts": len(concepts),
            "questions_generated": len([q for q in questions if q.get("subject") == subject]),
            "modules_covered": list(coverage.get(subject, {}).keys()),
        }
    return {
        "total_subjects": len(stats),
        "total_concepts": len(concepts),
        "total_questions_generated": len(questions),
        "subjects": stats,
    }
