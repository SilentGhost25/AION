# storage/database.py

import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

DB_PATH = "data/aion.db"

def get_db_path():
    Path("data").mkdir(exist_ok=True)
    return DB_PATH

@contextmanager
def get_connection():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_connection() as conn:
        conn.executescript("""
            -- Uploaded files tracking
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                subject TEXT,
                uploaded_by TEXT DEFAULT 'user',
                file_size INTEGER,
                question_count INTEGER DEFAULT 0,
                processed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Parsed questions
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                subject TEXT,
                question_text TEXT NOT NULL,
                answer_text TEXT,
                question_type TEXT,          -- mcq, short, long, numerical, descriptive
                marks INTEGER,
                bloom_level TEXT,            -- remember, understand, apply, analyze, evaluate, create
                unit TEXT,
                difficulty TEXT,             -- easy, medium, hard
                raw_line TEXT,               -- original line from file
                used_in_training INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES uploaded_files(id)
            );

            -- Training pairs generated from questions
            CREATE TABLE IF NOT EXISTS training_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anchor TEXT NOT NULL,
                positive TEXT NOT NULL,
                negative TEXT,
                pair_type TEXT,              -- question_similarity, question_topic, definition, contextual
                subject TEXT,
                source_question_id TEXT,
                used_in_training INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Question patterns learned
            CREATE TABLE IF NOT EXISTS question_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_template TEXT NOT NULL,  -- e.g., "Explain the concept of {CONCEPT} with {DETAIL}"
                question_type TEXT,
                bloom_level TEXT,
                subject TEXT,
                frequency INTEGER DEFAULT 1,
                example_question TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Model versions
            CREATE TABLE IF NOT EXISTS model_versions (
                version TEXT PRIMARY KEY,
                model_path TEXT NOT NULL,
                parent_version TEXT,
                training_pairs_count INTEGER,
                eval_score REAL,
                is_deployed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata_json TEXT
            );

            -- Training runs log
            CREATE TABLE IF NOT EXISTS training_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT,
                status TEXT,                 -- started, completed, failed, rejected
                pairs_used INTEGER,
                epochs INTEGER,
                loss_final REAL,
                eval_score REAL,
                duration_seconds REAL,
                error_message TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP
            );

            -- System state
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_questions_subject ON questions(subject);
            CREATE INDEX IF NOT EXISTS idx_questions_unused ON questions(used_in_training);
            CREATE INDEX IF NOT EXISTS idx_pairs_unused ON training_pairs(used_in_training);
            CREATE INDEX IF NOT EXISTS idx_files_unprocessed ON uploaded_files(processed);
        """)

        # Initialize system state
        defaults = {
            "total_questions_seen": "0",
            "total_training_runs": "0",
            "last_training_time": "",
            "model_frozen": "false",
            "current_model_version": "v0"
        }
        for key, value in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO system_state (key, value) VALUES (?, ?)",
                (key, value)
            )

def get_state(key: str) -> str:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

def set_state(key: str, value: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, datetime.now().isoformat())
        )

def count_unused_questions() -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM questions WHERE used_in_training = 0"
        ).fetchone()
        return row["cnt"]

def count_unused_pairs() -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM training_pairs WHERE used_in_training = 0"
        ).fetchone()
        return row["cnt"]

def file_already_uploaded(file_hash: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM uploaded_files WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return row is not None

def get_file_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()
