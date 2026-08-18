import json
import sqlite3
import numpy as np
import yaml
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

def load_config():
    with open("config/aion_config.yaml") as f:
        return yaml.safe_load(f)

class HardNegativeMiner:
    def __init__(self, model_path: str = "adapters/base/model_latest"):
        config = load_config()
        
        print(f"[MINER] Loading model from: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.threshold = config["training"]["hard_negative_threshold"]
        self.easy_ratio = config["training"]["easy_negative_ratio"]
        
        print(f"[MINER] Hard negative threshold: {self.threshold}")
        print(f"[MINER] Easy negative ratio: {self.easy_ratio}")
    
    def mine(
        self,
        pairs_path: str = "data/training_pairs/generated_pairs.jsonl",
        output_path: str = "data/training_pairs/hard_negative_pairs.jsonl",
        batch_size: int = 128
    ) -> int:
        """
        Load existing pairs.
        For each anchor, find chunks that are close but wrong.
        Replace easy negatives with hard ones.
        """
        print("\n[MINING] Starting hard negative mining...")
        
        # Load existing pairs
        pairs = []
        anchors = []
        positives = []
        
        with open(pairs_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pair = json.loads(line)
                    pairs.append(pair)
                    anchors.append(pair["anchor"])
                    positives.append(pair["positive"])
        
        print(f"[MINING] Loaded {len(pairs)} pairs")
        
        # Get all unique texts to embed
        all_texts = list(set(anchors + positives))
        print(f"[MINING] Embedding {len(all_texts)} unique texts...")
        
        # Batch encode
        all_embeddings = self.model.encode(
            all_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Build text -> embedding lookup
        text_to_idx = {text: i for i, text in enumerate(all_texts)}
        
        print("[MINING] Finding hard negatives...")
        
        hard_pairs = []
        easy_count = 0
        hard_count = 0
        
        for pair in pairs:
            anchor = pair["anchor"]
            positive = pair["positive"]
            subject = pair.get("subject", "unknown")
            
            anchor_emb = all_embeddings[text_to_idx[anchor]]
            
            # Compute similarity to all texts
            similarities = np.dot(all_embeddings, anchor_emb)
            
            # Sort by similarity descending
            sorted_indices = np.argsort(similarities)[::-1]
            
            found_hard = False
            for idx in sorted_indices:
                candidate_text = all_texts[idx]
                sim_score = similarities[idx]
                
                # Skip if it's the anchor or positive
                if candidate_text == anchor or candidate_text == positive:
                    continue
                
                # Hard negative: similar enough to be confusing but wrong
                if self.threshold > sim_score > 0.5:
                    hard_pairs.append({
                        "anchor": anchor,
                        "positive": positive,
                        "negative": candidate_text,
                        "subject": subject,
                        "pair_type": "hard",
                        "hard_negative_score": float(sim_score)
                    })
                    hard_count += 1
                    found_hard = True
                    break
            
            # If no hard negative found, use original easy negative
            if not found_hard and pair.get("negative"):
                pair["pair_type"] = "easy"
                hard_pairs.append(pair)
                easy_count += 1
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for pair in hard_pairs:
                f.write(json.dumps(pair) + "\n")
        
        print(f"\n[MINING COMPLETE]")
        print(f"  Hard negatives found: {hard_count}")
        print(f"  Easy negatives kept: {easy_count}")
        print(f"  Total pairs: {len(hard_pairs)}")
        print(f"  Saved to: {output_path}")
        
        return len(hard_pairs)
    
    def mine_from_feedback(
        self,
        feedback_path: str = "data/feedback/wrong_results.jsonl",
        output_path: str = "data/training_pairs/feedback_negatives.jsonl"
    ) -> int:
        """
        Convert admin feedback (marked wrong results) into hard negatives.
        This is the most valuable training signal.
        """
        if not Path(feedback_path).exists():
            print("[MINER] No feedback file found. Skipping.")
            return 0
        
        feedback_pairs = []
        with open(feedback_path) as f:
            for line in f:
                item = json.loads(line.strip())
                # Admin marked this result as wrong
                # query = the question asked
                # wrong_result = what the system returned
                # expected = what it should have returned (if provided)
                
                feedback_pairs.append({
                    "anchor": item["query"],
                    "positive": item.get("expected", ""),
                    "negative": item["wrong_result"],
                    "subject": item.get("subject", "unknown"),
                    "pair_type": "feedback_hard",
                    "hard_negative_score": 1.0  # Max difficulty: real failure
                })
        
        valid = [p for p in feedback_pairs if p["positive"]]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for pair in valid:
                f.write(json.dumps(pair) + "\n")
        
        print(f"[FEEDBACK] {len(valid)} feedback-derived hard negatives saved")
        return len(valid)

if __name__ == "__main__":
    miner = HardNegativeMiner()
    miner.mine()
    miner.mine_from_feedback()
