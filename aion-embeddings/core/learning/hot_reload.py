import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Singleton service that provides embeddings.
    Can hot-swap models without restarting the application.
    """
    
    _instance = None
    _loaded_models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.registry = ModelRegistry()
        self._initialized = True
        logger.info("EmbeddingService initialized")
    
    def embed(self, texts: list[str], subject: str = "general") -> list[list[float]]:
        model_path = self.registry.get_active_model(subject)
        
        if not model_path:
            logger.warning(f"No model for subject {subject}, using fallback")
            model_path = self.registry.get_active_model("general")
        
        if not model_path:
            # Absolute fallback if registry is completely empty
            try:
                import yaml
                with open("config/aion_config.yaml") as f:
                    config = yaml.safe_load(f)
                    model_path = config.get("model", {}).get("base", "all-MiniLM-L6-v2")
            except Exception:
                model_path = "all-MiniLM-L6-v2" 

        model = self._get_model(model_path)
        embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.tolist()
    
    def _get_model(self, model_path: str) -> SentenceTransformer:
        if model_path not in self._loaded_models:
            logger.info(f"Loading model: {model_path}")
            self._loaded_models[model_path] = SentenceTransformer(model_path)
        return self._loaded_models[model_path]
    
    def switch_model(self, subject: str, model_id: str):
        self.registry.rollback(model_id)
        logger.info(f"Switched {subject} to model {model_id}")
    
    def get_active_models(self) -> dict:
        return self.registry.list_models()
