import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from .contracts.learning import TrainingDataset

logger = logging.getLogger(__name__)

class EmbeddingTrainer:
    """
    Fine-tunes a SentenceTransformer embedding model on domain-specific data.
    """
    
    def __init__(self, model_name: str = None, output_dir: str = "data/models"):
        if model_name is None:
            try:
                import yaml
                with open("config/aion_config.yaml") as f:
                    config = yaml.safe_load(f)
                    model_name = config.get("model", {}).get("base", "all-MiniLM-L6-v2")
            except Exception:
                model_name = "all-MiniLM-L6-v2"
                
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing trainer with base model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def train(self, dataset: TrainingDataset, subject: str, epochs: int = 3, batch_size: int = 32, learning_rate: float = 5e-5, warmup_steps: int = 100) -> str:
        logger.info(f"Starting training: {len(dataset.pairs)} pairs, {epochs} epochs")
        
        examples = []
        for pair in dataset.pairs:
            if pair.negative:
                examples.append(InputExample(texts=[pair.anchor, pair.positive, pair.negative]))
            else:
                examples.append(InputExample(texts=[pair.anchor, pair.positive]))
        
        import random
        random.shuffle(examples)
        dataloader = DataLoader(examples, batch_size=batch_size, shuffle=True)
        
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        evaluator = None
        if len(examples) > 100:
            eval_examples = examples[-len(examples)//10:]
            evaluator = evaluation.InformationRetrievalEvaluator(
                queries={str(i): ex.texts[0] for i, ex in enumerate(eval_examples)},
                corpus={str(i): ex.texts[1] for i, ex in enumerate(eval_examples)},
                relevant_docs={str(i): {str(i)} for i in range(len(eval_examples))}
            )
        
        start_time = time.time()
        
        self.model.fit(
            train_objectives=[(dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            evaluator=evaluator,
            evaluation_steps=len(dataloader) // 2 if evaluator else 0,
            show_progress_bar=True
        )
        
        duration = time.time() - start_time
        checkpoint_path = self._save_checkpoint(subject, duration, dataset)
        
        logger.info(f"Training complete. Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)
    
    def _save_checkpoint(self, subject: str, duration: float, dataset: TrainingDataset) -> Path:
        checkpoint_dir = self.output_dir / subject
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"{subject}_checkpoint_{timestamp}"
        
        checkpoint_path = checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(checkpoint_path))
        
        metadata = {
            "checkpoint_id": checkpoint_id,
            "subject": subject,
            "timestamp": timestamp,
            "duration_seconds": duration,
            "training_pairs": len(dataset.pairs),
            "dataset_id": dataset.dataset_id,
            "base_model": self.model_name
        }
        
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_id}")
        return checkpoint_path
