import os
import json
import hashlib
import sqlite3

from pathlib import Path
from typing import List, Dict
import yaml

def load_config():
    with open("config/aion_config.yaml") as f:
        return yaml.safe_load(f)

from .processors import get_processor

def chunk_text(text: str, chunk_size: int, overlap: int, min_length: int) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        
        if len(chunk) >= min_length:
            chunks.append(chunk)
        
        start += chunk_size - overlap
    
    return chunks

def get_file_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def setup_db(db_path: str = "vector_store/metadata.db"):
    # Ensure vector_store directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            source_file TEXT,
            subject TEXT,
            chunk_index INTEGER,
            text TEXT,
            file_hash TEXT,
            embedded INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            file_hash TEXT PRIMARY KEY,
            filename TEXT,
            subject TEXT,
            chunk_count INTEGER,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def ingest_file(
    filepath: str,
    subject: str,
    config: dict,
    db_conn: sqlite3.Connection
) -> int:
    cfg = config["ingestion"]
    file_hash = get_file_hash(filepath)
    cursor = db_conn.cursor()
    
    # Skip already ingested files
    cursor.execute("SELECT file_hash FROM ingested_files WHERE file_hash = ?", (file_hash,))
    if cursor.fetchone():
        print(f"[SKIP] Already ingested: {filepath}")
        return 0
    
    processor_class = get_processor(filepath)
    if not processor_class:
        print(f"[SKIP] No processor found for: {filepath}")
        return 0
    
    print(f"[INGEST] Processing: {filepath}")
    raw_text = processor_class.extract(filepath)
    if not raw_text.strip():
        print(f"[WARN] No text extracted from: {filepath}")
        return 0
    chunks = chunk_text(
        raw_text,
        cfg["chunk_size"],
        cfg["chunk_overlap"],
        cfg["min_chunk_length"]
    )
    
    chunk_records = []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{file_hash}_{i}".encode()).hexdigest()
        chunk_records.append((
            chunk_id,
            os.path.basename(filepath),
            subject,
            i,
            chunk,
            file_hash
        ))
    
    cursor.executemany(
        "INSERT OR IGNORE INTO chunks (id, source_file, subject, chunk_index, text, file_hash) VALUES (?,?,?,?,?,?)",
        chunk_records
    )
    cursor.execute(
        "INSERT INTO ingested_files (file_hash, filename, subject, chunk_count) VALUES (?,?,?,?)",
        (file_hash, os.path.basename(filepath), subject, len(chunks))
    )
    db_conn.commit()
    
    print(f"[DONE] {len(chunks)} chunks stored from {filepath}")
    return len(chunks)

def ingest_directory(data_dir: str = "data/raw"):
    config = load_config()
    db_conn = setup_db()
    total = 0
    
    for subject_dir in Path(data_dir).iterdir():
        if subject_dir.is_dir():
            subject = subject_dir.name
            if subject not in config["subjects"]["active"]:
                print(f"[WARN] Subject '{subject}' not in active subjects. Skipping.")
                continue
            
            for filepath in subject_dir.rglob("*"):
                if filepath.is_file():
                    processor_class = get_processor(str(filepath))
                    if processor_class:
                        count = ingest_file(str(filepath), subject, config, db_conn)
                        total += count
    
    print(f"\n[COMPLETE] Total chunks ingested: {total}")
    return total

if __name__ == "__main__":
    ingest_directory()
