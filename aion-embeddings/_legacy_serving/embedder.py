import os
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer

class AIonEmbedder:
    """
    Core embedding class.
    Loads the correct model/adapter based on config.
    Admin can hot-swap subjects without restarting the API.
    """
    
    _instance = None
    _model = None
    _current_subject = None
    
    def __init__(self):
        self.config = self._load_config()
        self.adapter_base = "adapters"
        self._load_model(self.config["adapters"]["current_subject"])
    
    def _load_config(self) -> dict:
        with open("config/aion_config.yaml") as f:
            return yaml.safe_load(f)
    
    def _reload_config(self):
        """Reload config from disk (admin may have changed it)."""
        self.config = self._load_config()
    
    def _load_model(self, subject: str = "base") -> None:
        """Load subject-specific model or from live registry. Falls back to base if not found."""
        self._reload_config()
        
        registry_path = f"{self.adapter_base}/registry.json"
        subject_path = f"{self.adapter_base}/{subject}/model_latest"
        base_path = f"{self.adapter_base}/base/model_latest"
        fallback = self.config["model"]["base_model"]
        
        model_path = fallback
        
        if Path(registry_path).exists():
            try:
                with open(registry_path, "r") as f:
                    registry = json.load(f)
                    active_model = registry.get("active_model")
                    if active_model and Path(active_model).exists():
                        model_path = active_model
                        print(f"[EMBEDDER] Loaded model from live registry: {model_path}")
            except Exception as e:
                print(f"[EMBEDDER] Failed to read registry.json: {e}")
        elif Path(subject_path).exists():
            model_path = subject_path
            print(f"[EMBEDDER] Loading subject model: {subject_path}")
        elif Path(base_path).exists():
            model_path = base_path
            print(f"[EMBEDDER] Subject model not found, using base: {base_path}")
        else:
            print(f"[EMBEDDER] No fine-tuned model found, using: {fallback}")
        
        self._model = SentenceTransformer(model_path)
        self._current_subject = subject
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(device)
        print(f"[EMBEDDER] Model loaded on: {device}")
    
    def switch_subject(self, subject: str) -> Dict:
        """
        Admin hot-swap. Changes the active model without restart.
        """
        if subject == self._current_subject:
            return {"status": "no_change", "subject": subject}
        
        print(f"[EMBEDDER] Switching subject: {self._current_subject} -> {subject}")
        self._load_model(subject)
        
        return {
            "status": "switched",
            "from": self._current_subject,
            "to": subject
        }
    
    def embed(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
    
    def get_model_info(self) -> Dict:
        self._reload_config()
        return {
            "current_subject": self._current_subject,
            "base_model": self.config["model"]["base_model"],
            "embedding_dim": self.config["model"]["embedding_dim"],
            "max_seq_length": self.config["model"]["max_seq_length"],
            "device": str(next(self._model.parameters()).device)
        }
