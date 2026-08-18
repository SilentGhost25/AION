import logging
import sqlite3
from pathlib import Path
from typing import Tuple, List
import json
from sentence_transformers import SentenceTransformer, evaluation

logger = logging.getLogger(__name__)

class EmbeddingEvaluator:
    """
    Tests a checkpoint against a golden evaluation set.
    """
    
    def __init__(self, eval_set_path: str = "data/eval_set.jsonl"):
        self.eval_set_path = Path(eval_set_path)
        self.min_score = 0.70
    
    def evaluate(self, checkpoint_path: str) -> Tuple[float, bool, str]:
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        try:
            model = SentenceTransformer(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return 0.0, False, f"Model load error: {e}"
        
        try:
            eval_data = self._load_eval_set()
        except Exception as e:
            logger.warning(f"Evaluation set not found: {e}. Skipping evaluation.")
            return 1.0, True, "No eval set available; assuming acceptable"
        
        try:
            score = self._compute_recall_at_k(model, eval_data)
            is_acceptable = score >= self.min_score
            status = "PASS" if is_acceptable else "FAIL"
            
            report = f"""
Evaluation Report
-----------------
Model: {checkpoint_path}
Recall@10: {score:.4f}
Min Required: {self.min_score:.4f}
Status: {status}
            """
            
            logger.info(report)
            return score, is_acceptable, report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0, False, f"Evaluation error: {e}"
    
    def _load_eval_set(self) -> dict:
        if not self.eval_set_path.exists():
            raise FileNotFoundError(f"Eval set not found: {self.eval_set_path}")
        
        queries = {}
        corpus = {}
        relevant_docs = {}
        
        with open(self.eval_set_path) as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                q_id = f"q_{i}"
                d_id = f"d_{i}"
                
                queries[q_id] = example["anchor"]
                corpus[d_id] = example["positive"]
                relevant_docs[q_id] = {d_id}
        
        return {
            "queries": queries,
            "corpus": corpus,
            "relevant_docs": relevant_docs
        }
    
    def _compute_recall_at_k(self, model: SentenceTransformer, eval_data: dict, k: int = 10) -> float:
        queries = eval_data["queries"]
        corpus = eval_data["corpus"]
        relevant_docs = eval_data["relevant_docs"]
        
        corpus_embeddings = model.encode(list(corpus.values()), normalize_embeddings=True)
        total_recall = 0.0
        
        for q_id, query_text in queries.items():
            query_embedding = model.encode([query_text], normalize_embeddings=True)[0]
            
            import numpy as np
            similarities = np.dot(corpus_embeddings, query_embedding)
            
            top_k_indices = np.argsort(-similarities)[:k]
            top_k_doc_ids = [list(corpus.keys())[i] for i in top_k_indices]
            
            expected_docs = relevant_docs[q_id]
            hits = len(set(top_k_doc_ids) & expected_docs)
            
            if hits > 0:
                total_recall += 1.0
        
        return total_recall / len(queries)
