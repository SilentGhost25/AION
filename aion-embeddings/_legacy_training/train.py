import os
import json
import yaml
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    class DummyWandb:
        def init(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
    wandb = DummyWandb()
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    models
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator

def load_config():
    with open("config/aion_config.yaml") as f:
        return yaml.safe_load(f)

def save_config(config: dict):
    with open("config/aion_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def load_training_pairs(path: str, subject_filter: Optional[str] = None, max_samples: int = 10000) -> List[InputExample]:
    examples = []
    if not Path(path).exists():
        return examples
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(examples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            
            if subject_filter and pair.get("subject") != subject_filter:
                continue
            
            anchor = pair["anchor"]
            positive = pair["positive"]
            negative = pair.get("negative", "")
            
            if negative:
                examples.append(InputExample(texts=[anchor, positive, negative]))
            else:
                examples.append(InputExample(texts=[anchor, positive]))
    
    return examples

def build_model(base_model_name: str, max_seq_length: int) -> SentenceTransformer:
    word_embedding_model = models.Transformer(
        base_model_name,
        max_seq_length=max_seq_length
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    # Add normalization for cosine similarity
    normalize = models.Normalize()
    
    return SentenceTransformer(modules=[
        word_embedding_model,
        pooling_model,
        normalize
    ])

def get_loss(model: SentenceTransformer, loss_name: str):
    loss_map = {
        "mnrl": losses.MultipleNegativesRankingLoss(model),
        "triplet": losses.TripletLoss(model),
        "contrastive": losses.ContrastiveLoss(model),
        "cosine": losses.CosineSimilarityLoss(model),
        "cached_mnrl": losses.CachedMultipleNegativesRankingLoss(model)
    }
    
    if loss_name not in loss_map:
        print(f"[WARN] Unknown loss '{loss_name}', defaulting to mnrl")
        return loss_map["mnrl"]
    
    return loss_map[loss_name]

def build_evaluator(eval_pairs_path: str) -> Optional[InformationRetrievalEvaluator]:
    if not Path(eval_pairs_path).exists():
        print("[WARN] No evaluation set found. Skipping evaluation during training.")
        return None
    
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    with open(eval_pairs_path) as f:
        for i, line in enumerate(f):
            pair = json.loads(line.strip())
            q_id = f"q_{i}"
            d_id = f"d_{i}"
            queries[q_id] = pair["anchor"]
            corpus[d_id] = pair["positive"]
            relevant_docs[q_id] = {d_id}
    
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="aion_eval",
        show_progress_bar=True
    )

def train(
    subject: Optional[str] = None,
    config_override: Optional[Dict] = None,
    run_name: Optional[str] = None
):
    config = load_config()
    
    # Allow runtime config overrides (from Admin UI)
    if config_override:
        for key, value in config_override.items():
            keys = key.split(".")
            cfg = config
            for k in keys[:-1]:
                cfg = cfg[k]
            cfg[keys[-1]] = value
        print(f"[CONFIG] Applied {len(config_override)} overrides")
    
    # Determine output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subject_tag = subject if subject else "base"
    output_path = f"adapters/{subject_tag}/model_{timestamp}"
    
    # Initialize W&B
    run_name = run_name or f"aion_{subject_tag}_{timestamp}"
    wandb.init(
        project="aion-embeddings",
        name=run_name,
        config={
            "base_model": config["model"]["base_model"],
            "subject": subject_tag,
            "loss": config["training"]["loss"],
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"]
        }
    )
    
    print(f"\n{'='*60}")
    print(f"AION Training Run: {run_name}")
    print(f"Subject: {subject_tag}")
    print(f"Base Model: {config['model']['base_model']}")
    print(f"Loss: {config['training']['loss']}")
    print(f"{'='*60}\n")
    
    # Build model
    # Check if we have a fine-tuned base to continue from
    base_model_path = "adapters/base/model_latest"
    if Path(base_model_path).exists() and subject:
        print(f"[MODEL] Loading from fine-tuned base: {base_model_path}")
        model = SentenceTransformer(base_model_path)
    else:
        print(f"[MODEL] Loading from HuggingFace: {config['model']['base_model']}")
        model = build_model(
            config["model"]["base_model"],
            config["model"]["max_seq_length"]
        )
    
    # Load training data
    pairs_path = "data/training_pairs/generated_pairs.jsonl"
    print(f"[DATA] Loading training pairs...")
    examples = load_training_pairs(pairs_path, subject_filter=subject)
    
    if not examples:
        print(f"[ERROR] No training examples found for subject: {subject}")
        wandb.finish()
        return None
    
    print(f"[DATA] {len(examples)} training examples loaded")
    
    # Get loss
    loss_fn = get_loss(model, config["training"]["loss"])
    
    # Build dataloader
    train_dataloader = DataLoader(
        examples,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    # Build evaluator
    evaluator = build_evaluator("data/training_pairs/eval_pairs.jsonl")
    
    # Train
    warmup_steps = int(len(train_dataloader) * config["training"]["epochs"] * config["training"]["warmup_ratio"])
    
    model.fit(
        train_objectives=[(train_dataloader, loss_fn)],
        evaluator=evaluator,
        epochs=config["training"]["epochs"],
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config["training"]["learning_rate"]},
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True,
        callback=lambda score, epoch, steps: wandb.log({
            "eval_score": score,
            "epoch": epoch,
            "steps": steps
        }) if score is not None else None
    )
    
    # Save as "latest" for this subject
    latest_path = f"adapters/{subject_tag}/model_latest"
    model.save(latest_path)
    print(f"\n[SAVED] Model saved to: {output_path}")
    print(f"[SAVED] Latest pointer updated: {latest_path}")
    
    # Save evaluation results
    if evaluator is not None:
        try:
            print("[EVAL] Running final evaluation on best model...")
            eval_metrics = evaluator(model, output_path=output_path, epoch=-1, steps=-1)
            # Evaluator returns a dict or float depending on the exact ST version, usually a dict for IREvaluator
            if isinstance(eval_metrics, dict):
                metrics_out = {}
                for key, val in eval_metrics.items():
                    metrics_out[key] = val
                
                eval_path = os.path.join(output_path, "eval_results.json")
                with open(eval_path, "w") as f:
                    json.dump(metrics_out, f, indent=2)
                print(f"[EVAL] Saved evaluation results to {eval_path}")
        except Exception as e:
            print(f"[WARN] Failed to save evaluation results: {e}")
            
    wandb.finish()
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None, help="Subject to train adapter for (None = base)")
    parser.add_argument("--loss", type=str, default=None, help="Override loss function")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--base_model", type=str, default=None, help="Override base model")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Override max sequence length")
    args = parser.parse_args()
    
    overrides = {}
    if args.loss:
        overrides["training.loss"] = args.loss
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.lr:
        overrides["training.learning_rate"] = args.lr
    if args.batch_size:
        overrides["training.batch_size"] = args.batch_size
    if args.base_model:
        overrides["model.base_model"] = args.base_model
    if args.max_seq_length:
        overrides["model.max_seq_length"] = args.max_seq_length
    
    train(subject=args.subject, config_override=overrides if overrides else None)
