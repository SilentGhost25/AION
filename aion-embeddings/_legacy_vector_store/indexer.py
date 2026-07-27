import os
import json
import sqlite3
import numpy as np
import faiss
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from serving.embedder import AIonEmbedder

def load_config():
    with open("config/aion_config.yaml") as f:
        return yaml.safe_load(f)

class AIonIndex:
    """
    FAISS-based vector index with metadata.
    Supports subject filtering, score thresholding,
    and incremental updates.
    """
    
    def __init__(self, index_dir: str = "vector_store/indices"):
        self.config = load_config()
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedder = AIonEmbedder()
        self.dim = self.config["model"]["embedding_dim"]
        
        self.indices: Dict[str, faiss.Index] = {}
        self.id_maps: Dict[str, List[str]] = {}  # subject → list of chunk_ids
        self.chunk_texts: Dict[str, str] = {}    # chunk_id → text
        
        self._load_all_indices()
    
    def _get_index_path(self, subject: str) -> Path:
        return self.index_dir / f"{subject}.faiss"
    
    def _get_idmap_path(self, subject: str) -> Path:
        return self.index_dir / f"{subject}_idmap.json"
    
    def _load_all_indices(self):
        """Load all existing FAISS indices from disk."""
        for subject in self.config["subjects"]["active"]:
            self._load_index(subject)
        self._load_index("base")  # Always load base index
    
    def _load_index(self, subject: str):
        idx_path = self._get_index_path(subject)
        map_path = self._get_idmap_path(subject)
        
        if idx_path.exists() and map_path.exists():
            self.indices[subject] = faiss.read_index(str(idx_path))
            with open(map_path) as f:
                self.id_maps[subject] = json.load(f)
            print(f"[INDEX] Loaded {subject}: {self.indices[subject].ntotal} vectors")
        else:
            # Create new index
            index = faiss.IndexFlatIP(self.dim)  # Inner product (cosine on normalized vectors)
            self.indices[subject] = index
            self.id_maps[subject] = []
    
    def _save_index(self, subject: str):
        faiss.write_index(
            self.indices[subject],
            str(self._get_index_path(subject))
        )
        with open(self._get_idmap_path(subject), "w") as f:
            json.dump(self.id_maps[subject], f)
    
    def _load_chunk_texts(self, db_path: str = "vector_store/metadata.db"):
        """Load all chunk texts into memory for retrieval."""
        if not Path(db_path).exists():
            return
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text FROM chunks")
        rows = cursor.fetchall()
        self.chunk_texts = {row[0]: row[1] for row in rows}
        conn.close()
        print(f"[INDEX] Loaded {len(self.chunk_texts)} chunk texts into memory")
    
    def build_index(
        self,
        subject: Optional[str] = None,
        batch_size: int = 128,
        db_path: str = "vector_store/metadata.db"
    ):
        """
        Build or rebuild the FAISS index for a subject (or all subjects).
        Embeds all chunks from the database.
        """
        self._load_chunk_texts(db_path)
        if not Path(db_path).exists():
            print("[BUILD] No database found. Skipping.")
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        subjects_to_build = [subject] if subject else self.config["subjects"]["active"]
        
        for subj in subjects_to_build:
            print(f"\n[BUILD] Building index for subject: {subj}")
            
            cursor.execute(
                "SELECT id, text FROM chunks WHERE subject = ? AND length(text) > 50",
                (subj,)
            )
            rows = cursor.fetchall()
            
            if not rows:
                print(f"[BUILD] No chunks for {subj}. Skipping.")
                continue
            
            chunk_ids = [r[0] for r in rows]
            chunk_texts = [r[1] for r in rows]
            
            # Batch embed with subject-specific model
            self.embedder.switch_subject(subj)
            
            all_embeddings = []
            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i:i+batch_size]
                embs = self.embedder.embed(batch)
                all_embeddings.append(embs)
                print(f"[BUILD] Embedded {min(i+batch_size, len(chunk_texts))}/{len(chunk_texts)}", end="\r")
            
            embeddings = np.vstack(all_embeddings).astype("float32")
            
            # Reset and rebuild
            new_index = faiss.IndexFlatIP(self.dim)
            new_index.add(embeddings)
            
            self.indices[subj] = new_index
            self.id_maps[subj] = chunk_ids
            
            self._save_index(subj)
            print(f"\n[BUILD] {subj}: {new_index.ntotal} vectors indexed")
        
        # Update embedded status in DB
        placeholders = ",".join(["?" for _ in subjects_to_build])
        cursor.execute(
            f"UPDATE chunks SET embedded = 1 WHERE subject IN ({placeholders})",
            subjects_to_build
        )
        conn.commit()
        conn.close()
    
    def search(
        self,
        query: str,
        subject: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search the index.
        subject=None searches the base (all-subject) index.
        """
        config = load_config()  # Fresh read so admin changes take effect
        top_k = top_k or config["retrieval"]["top_k"]
        score_threshold = score_threshold or config["retrieval"]["score_threshold"]
        
        search_subject = subject or "base"
        
        if search_subject not in self.indices:
            print(f"[SEARCH] Subject '{search_subject}' not in index. Using base.")
            search_subject = "base"
        
        if self.indices[search_subject].ntotal == 0:
            return []
        
        # Embed query using subject-specific model
        self.embedder.switch_subject(search_subject)
        query_emb = self.embedder.embed_single(query).astype("float32").reshape(1, -1)
        
        # Search
        scores, indices = self.indices[search_subject].search(query_emb, top_k * 2)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < score_threshold:
                continue
            
            chunk_id = self.id_maps[search_subject][idx]
            text = self.chunk_texts.get(chunk_id, "")
            
            if text:
                results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": float(score),
                    "subject": search_subject
                })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def add_chunks_incremental(
        self,
        chunk_ids: List[str],
        chunk_texts: List[str],
        subject: str
    ):
        """
        Add new chunks to an existing index without full rebuild.
        """
        if not chunk_ids:
            return
        
        self.embedder.switch_subject(subject)
        embeddings = self.embedder.embed(chunk_texts).astype("float32")
        
        self.indices[subject].add(embeddings)
        self.id_maps[subject].extend(chunk_ids)
        self.chunk_texts.update(dict(zip(chunk_ids, chunk_texts)))
        
        self._save_index(subject)
        print(f"[INCREMENTAL] Added {len(chunk_ids)} chunks to {subject} index")
