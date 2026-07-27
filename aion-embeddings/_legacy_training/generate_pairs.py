import sqlite3
import json
import random
import yaml
import re
from pathlib import Path
from typing import List, Tuple, Dict

def load_config():
    with open("config/aion_config.yaml") as f:
        return yaml.safe_load(f)

def get_chunks_by_subject(db_conn: sqlite3.Connection, subject: str) -> List[Dict]:
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT id, text, subject FROM chunks WHERE subject = ? AND length(text) > 100",
        (subject,)
    )
    rows = cursor.fetchall()
    return [{"id": r[0], "text": r[1], "subject": r[2]} for r in rows]

def generate_pairs_from_chunks(
    chunks: List[Dict],
    subject: str,
    config: dict
) -> List[Dict]:
    """
    Generate training pairs using sliding window positives.
    Adjacent chunks are positives (same context window).
    Chunks from different subjects are easy negatives.
    """
    pairs = []
    
    for i, chunk in enumerate(chunks):
        # Positive: adjacent chunk (same context)
        if i + 1 < len(chunks):
            positive = chunks[i + 1]
            
            # Easy negative: random chunk from same subject but far away
            neg_idx = random.randint(0, len(chunks) - 1)
            while abs(neg_idx - i) < 10:  # Must be far from anchor
                neg_idx = random.randint(0, len(chunks) - 1)
            
            easy_negative = chunks[neg_idx]
            
            pairs.append({
                "anchor": chunk["text"],
                "positive": positive["text"],
                "negative": easy_negative["text"],
                "subject": subject,
                "pair_type": "easy"
            })
    
    return pairs

def load_question_pairs(questions_dir: str = "data/training_pairs") -> List[Dict]:
    """
    Load manually curated Q&A pairs if they exist.
    Format: jsonl with anchor, positive, subject fields.
    """
    pairs = []
    qdir = Path(questions_dir)
    
    if not qdir.exists():
        return pairs
    
    for jsonl_file in qdir.glob("*.jsonl"):
        if jsonl_file.name == "generated_pairs.jsonl":
            continue
        with open(jsonl_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    
    print(f"[PAIRS] Loaded {len(pairs)} curated pairs from {questions_dir}")
    return pairs

def generate_all_pairs(output_path: str = "data/training_pairs/generated_pairs.jsonl", db_path: str = "vector_store/metadata.db"):
    config = load_config()
    
    # Ensure database directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db_conn = sqlite3.connect(db_path)
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            text TEXT,
            subject TEXT
        )
    """)
    db_conn.commit()
    
    subject_weights = config.get("subjects", {}).get("weights", {"AIML": 1.0})
    all_pairs = []
    
    # 1. Try reading from SQLite chunks table
    for subject, weight in subject_weights.items():
        chunks = get_chunks_by_subject(db_conn, subject)
        if chunks:
            pairs = generate_pairs_from_chunks(chunks, subject, config)
            sample_size = int(len(pairs) * weight * 2)
            sampled = random.sample(pairs, min(sample_size, len(pairs)))
            all_pairs.extend(sampled)
            print(f"[PAIRS] {subject} (DB): {len(sampled)} pairs generated")
    
    # 2. Fallback / Augment: Load from Knowledge Evolution Graph (memory/concepts.json) if available
    concepts_file = Path("../memory/concepts.json")
    if not concepts_file.exists():
        concepts_file = Path("memory/concepts.json")
        
    if concepts_file.exists():
        try:
            with open(concepts_file, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
                # Sanitize unescaped control characters in JSON strings (preserving \n, \r, \t)
                raw_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', raw_text)
                concept_data = json.loads(raw_text, strict=False)
            
            concept_chunks = []
            for idx, item in enumerate(concept_data):
                content = item.get("content", "").strip() if isinstance(item, dict) else ""
                if len(content) > 40:
                    concept_chunks.append({"id": f"c_{idx}", "text": content, "subject": "AIML"})
            
            if concept_chunks:
                print(f"[PAIRS] Loaded {len(concept_chunks)} concept genomes from {concepts_file}")
                concept_pairs = generate_pairs_from_chunks(concept_chunks, "AIML", config)
                all_pairs.extend(concept_pairs)
                print(f"[PAIRS] Generated {len(concept_pairs)} training pairs from Knowledge Evolution Graph")
        except Exception as e:
            print(f"[WARN] Could not load memory concepts: {e}")

    # Add curated Q&A pairs
    curated = load_question_pairs()
    all_pairs.extend(curated)
    
    # Shuffle
    random.shuffle(all_pairs)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    print(f"\n[COMPLETE] Total training pairs: {len(all_pairs)}")
    print(f"[SAVED] {output_path}")
    return len(all_pairs)

if __name__ == "__main__":
    generate_all_pairs()
