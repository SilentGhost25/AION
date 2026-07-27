import json
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Maintains a versioned registry of all embedding models.
    """
    
    def __init__(self, registry_dir: str = "data/model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.registry_dir / "registry.db"
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                checkpoint_path TEXT NOT NULL,
                subject TEXT NOT NULL,
                eval_score REAL NOT NULL,
                is_active INTEGER DEFAULT 0,
                parent_model_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject, is_active)
            )
        """)
        conn.commit()
        conn.close()
    
    def register_model(
        self,
        checkpoint_path: str,
        subject: str,
        eval_score: float,
        parent_model_id: Optional[str] = None
    ) -> str:
        checkpoint_path = Path(checkpoint_path)
        model_id = checkpoint_path.name
        
        managed_path = self.registry_dir / subject / model_id
        managed_path.parent.mkdir(parents=True, exist_ok=True)
        
        if checkpoint_path != managed_path:
            shutil.copytree(checkpoint_path, managed_path, dirs_exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        prev_active = conn.execute(
            "SELECT model_id, eval_score FROM models WHERE subject = ? AND is_active = 1",
            (subject,)
        ).fetchone()
        
        should_activate = prev_active is None or eval_score > prev_active[1]
        
        if should_activate and prev_active:
            conn.execute("UPDATE models SET is_active = 0 WHERE model_id = ?", (prev_active[0],))
        
        conn.execute("""
            INSERT INTO models (model_id, checkpoint_path, subject, eval_score, is_active, parent_model_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (model_id, str(managed_path), subject, eval_score, 1 if should_activate else 0, parent_model_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Registered model: {model_id} (subject={subject}, score={eval_score:.4f}, active={should_activate})")
        return model_id
    
    def get_active_model(self, subject: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT checkpoint_path FROM models WHERE subject = ? AND is_active = 1",
            (subject,)
        ).fetchone()
        conn.close()
        
        if row:
            return row[0]
        
        logger.warning(f"No active model found for subject: {subject}")
        return None
    
    def list_models(self, subject: str = None) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        query = "SELECT model_id, subject, eval_score, is_active, created_at FROM models"
        params = []
        
        if subject:
            query += " WHERE subject = ?"
            params.append(subject)
        
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        return [
            {
                "model_id": row[0],
                "subject": row[1],
                "eval_score": row[2],
                "is_active": bool(row[3]),
                "created_at": row[4]
            }
            for row in rows
        ]
    
    def rollback(self, model_id: str):
        conn = sqlite3.connect(self.db_path)
        model = conn.execute("SELECT subject FROM models WHERE model_id = ?", (model_id,)).fetchone()
        
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        
        subject = model[0]
        conn.execute("UPDATE models SET is_active = 0 WHERE subject = ?", (subject,))
        conn.execute("UPDATE models SET is_active = 1 WHERE model_id = ?", (model_id,))
        
        conn.commit()
        conn.close()
        logger.info(f"Rolled back to model: {model_id}")
