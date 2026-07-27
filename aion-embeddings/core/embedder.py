# core/embedder.py

import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer
from storage.database import get_state

def load_config():
    with open("config/aion_config.yaml") as f:
        return yaml.safe_load(f)

class AionEmbedder:
    """
    Lightweight wrapper around SentenceTransformer to handle dynamic loading.
    Checks registry/DB on inference to ensure the current version is active.
    """
    def __init__(self):
        self.config = load_config()
        self.models_dir = Path(self.config["system"]["data_dir"]) / "models"
        self.model = None
        self.current_version = None
        self.load_active_model()

    def get_active_version(self) -> str:
        return get_state("current_model_version") or "v0"

    def load_active_model(self):
        version = self.get_active_version()
        if version == self.current_version and self.model is not None:
            return

        print(f"[Embedder] Loading model version {version}")
        
        if version == "v0":
            model_path = self.config["model"]["base"]
        else:
            candidate = self.models_dir / version / "model"
            if candidate.exists():
                model_path = str(candidate)
            else:
                print(f"[Embedder] WARNING: Model path not found for {version}. Falling back to base.")
                model_path = self.config["model"]["base"]
                
        self.model = SentenceTransformer(model_path)
        self.current_version = version

    def encode(self, sentences: list, **kwargs):
        self.load_active_model() # Check if swap is needed
        return self.model.encode(sentences, **kwargs)
