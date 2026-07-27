import json
import sqlite3
from pathlib import Path
from typing import List
from .contracts.learning import TrainingPair
import random
import logging

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """
    Stores ALL training pairs ever generated.
    """
    
    def __init__(self, db_path: str = "data/replay_buffer.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pairs (
                pair_id TEXT PRIMARY KEY,
                anchor TEXT NOT NULL,
                positive TEXT NOT NULL,
                negative TEXT,
                subject TEXT NOT NULL,
                bloom_level TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
    
    def add_pairs(self, pairs: List[TrainingPair]):
        conn = sqlite3.connect(self.db_path)
        for pair in pairs:
            conn.execute("""
                INSERT OR REPLACE INTO pairs 
                (pair_id, anchor, positive, negative, subject, bloom_level, source, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pair.pair_id, pair.anchor, pair.positive, pair.negative,
                pair.subject, pair.bloom_level, pair.source, pair.confidence
            ))
        conn.commit()
        conn.close()
        logger.info(f"Added {len(pairs)} pairs to replay buffer")
    
    def sample(self, count: int, subject: str = None) -> List[TrainingPair]:
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT pair_id, anchor, positive, negative, subject, bloom_level, source, confidence
            FROM pairs
        """
        params = []
        if subject:
            query += " WHERE subject = ?"
            params.append(subject)
        query += " ORDER BY RANDOM() LIMIT ?"
        params.append(count)
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        pairs = [
            TrainingPair(
                pair_id=row[0],
                anchor=row[1],
                positive=row[2],
                negative=row[3],
                subject=row[4],
                bloom_level=row[5],
                source=row[6],
                confidence=row[7]
            )
            for row in rows
        ]
        return pairs
    
    def get_statistics(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
        by_subject = conn.execute("SELECT subject, COUNT(*) as cnt FROM pairs GROUP BY subject").fetchall()
        by_source = conn.execute("SELECT source, COUNT(*) as cnt FROM pairs GROUP BY source").fetchall()
        conn.close()
        return {
            "total_pairs": total,
            "by_subject": {row[0]: row[1] for row in by_subject},
            "by_source": {row[0]: row[1] for row in by_source}
        }
