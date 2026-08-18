# core/training_engine.py

import os
import json
import yaml
import torch
import random
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from sentence_transformers import (
    SentenceTransformer, InputExample,
    losses, evaluation, models
)
from torch.utils.data import DataLoader
from storage.database import (
    get_connection, get_state, set_state,
    count_unused_pairs
)

def load_config():
    with open("config/aion_config.yaml") as f:
        return yaml.safe_load(f)


class TrainingEngine:
    """
    Handles all model training.
    Supports:
    - Initial training from scratch (on base model)
    - Incremental training on new data
    - Experience replay to prevent forgetting
    - Automatic evaluation and version management
    - Rollback if new model is worse
    """

    def __init__(self):
        self.config = load_config()
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _get_current_model_path(self) -> str:
        """Get path to currently deployed model."""
        version = get_state("current_model_version")
        if version and version != "v0":
            candidate = self.models_dir / version / "model"
            if candidate.exists():
                return str(candidate)
        # Fall back to base model from HuggingFace
        return self.config["model"]["base"]

    def _load_model(self, model_path: str = None) -> SentenceTransformer:
        """Load a SentenceTransformer model."""
        path = model_path or self._get_current_model_path()
        print(f"[Training] Loading model from: {path}")
        model = SentenceTransformer(path)
        model.max_seq_length = self.config["model"]["max_seq_length"]
        return model

    def _load_training_pairs(
        self,
        limit: int = None,
        include_used: bool = False
    ) -> List[InputExample]:
        """Load training pairs from database."""
        with get_connection() as conn:
            if include_used:
                query = "SELECT anchor, positive, negative, pair_type FROM training_pairs ORDER BY RANDOM()"
            else:
                query = "SELECT anchor, positive, negative, pair_type FROM training_pairs WHERE used_in_training = 0 ORDER BY RANDOM()"

            if limit:
                query += f" LIMIT {limit}"

            rows = conn.execute(query).fetchall()

        examples = []
        for row in rows:
            anchor = row["anchor"]
            positive = row["positive"]
            negative = row["negative"]

            if negative:
                examples.append(InputExample(texts=[anchor, positive, negative]))
            else:
                examples.append(InputExample(texts=[anchor, positive]))

        return examples

    def _load_replay_buffer(self, count: int) -> List[InputExample]:
        """Load old training data for experience replay."""
        return self._load_training_pairs(limit=count, include_used=True)

    def _get_loss(self, model: SentenceTransformer, loss_name: str):
        """Get loss function by name."""
        loss_map = {
            "mnrl": losses.MultipleNegativesRankingLoss(model),
            "triplet": losses.TripletLoss(model),
            "contrastive": losses.ContrastiveLoss(model),
        }
        return loss_map.get(loss_name, loss_map["mnrl"])

    def _build_evaluator(self) -> Optional[evaluation.InformationRetrievalEvaluator]:
        """Build evaluator from stored questions."""
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT question_text, answer_text
                FROM questions
                WHERE answer_text IS NOT NULL AND length(answer_text) > 10
                ORDER BY RANDOM()
                LIMIT 200
            """).fetchall()

        if len(rows) < 10:
            # Not enough evaluation data; use question pairs instead
            rows = conn.execute("""
                SELECT anchor, positive
                FROM training_pairs
                WHERE used_in_training = 1
                ORDER BY RANDOM()
                LIMIT 200
            """).fetchall()

            if len(rows) < 10:
                return None

            queries = {f"q_{i}": r["anchor"] for i, r in enumerate(rows)}
            corpus = {f"d_{i}": r["positive"] for i, r in enumerate(rows)}
        else:
            queries = {f"q_{i}": r["question_text"] for i, r in enumerate(rows)}
            corpus = {f"d_{i}": r["answer_text"] for i, r in enumerate(rows)}

        relevant_docs = {f"q_{i}": {f"d_{i}"} for i in range(len(rows))}

        return evaluation.InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="aion_eval",
            show_progress_bar=False
        )

    def train(
        self,
        epochs: int = None,
        learning_rate: float = None,
        batch_size: int = None,
        loss_name: str = None,
        force: bool = False
    ) -> Dict:
        """
        Main training method.
        Returns training results including version, score, and status.
        """
        config = load_config()  # Fresh read
        t_cfg = config["training"]

        # Check if model is frozen
        if get_state("model_frozen") == "true" and not force:
            return {
                "status": "skipped",
                "reason": "Model is frozen by admin. Use force=True to override."
            }

        # Use overrides or config defaults
        epochs = epochs or t_cfg["epochs"]
        learning_rate = learning_rate or t_cfg["learning_rate"]
        batch_size = batch_size or t_cfg["batch_size"]
        loss_name = loss_name or t_cfg["loss"]

        # Load new training pairs
        new_examples = self._load_training_pairs()
        if not new_examples and not force:
            return {"status": "skipped", "reason": "No new training pairs available."}

        print(f"\n{'='*60}")
        print(f"  AION Training Run")
        print(f"  New examples: {len(new_examples)}")
        print(f"  Epochs: {epochs}")
        print(f"  LR: {learning_rate}")
        print(f"  Loss: {loss_name}")
        print(f"{'='*60}\n")

        # Experience replay
        replay_count = 0
        if t_cfg.get("replay_ratio", 0) > 0 and new_examples:
            replay_count = int(len(new_examples) * t_cfg["replay_ratio"])
            replay_examples = self._load_replay_buffer(replay_count)
            all_examples = new_examples + replay_examples
            print(f"[Training] Replay buffer: {len(replay_examples)} old examples added")
        else:
            all_examples = new_examples

        random.shuffle(all_examples)
        print(f"[Training] Total training examples: {len(all_examples)}")

        # Load model
        model = self._load_model()

        # Get loss
        loss_fn = self._get_loss(model, loss_name)

        # Build dataloader
        dataloader = DataLoader(all_examples, batch_size=batch_size, shuffle=True)

        # Build evaluator
        evaluator = self._build_evaluator()

        # Version info
        current_version = get_state("current_model_version") or "v0"
        try:
            version_num = int(current_version.replace("v", "")) + 1
        except ValueError:
            version_num = 1
        new_version = f"v{version_num}"
        output_path = str(self.models_dir / new_version / "model")

        # Evaluate BEFORE training (baseline)
        baseline_score = 0.0
        if evaluator:
            try:
                baseline_score = evaluator(model, output_path=None)
                if isinstance(baseline_score, dict):
                    baseline_score = list(baseline_score.values())[0] if baseline_score else 0.0
            except Exception:
                baseline_score = 0.0
        print(f"[Training] Baseline score: {baseline_score:.4f}")

        # Train
        start_time = time.time()
        warmup_steps = int(len(dataloader) * epochs * t_cfg.get("warmup_ratio", 0.1))

        model.fit(
            train_objectives=[(dataloader, loss_fn)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            output_path=output_path,
            save_best_model=True if evaluator else False,
            show_progress_bar=True
        )

        duration = time.time() - start_time

        # Evaluate AFTER training
        new_score = 0.0
        if evaluator:
            try:
                trained_model = SentenceTransformer(output_path)
                new_score = evaluator(trained_model, output_path=None)
                if isinstance(new_score, dict):
                    new_score = list(new_score.values())[0] if new_score else 0.0
            except Exception:
                new_score = 0.0
        print(f"[Training] New model score: {new_score:.4f}")

        # Decision: deploy or rollback
        e_cfg = config.get("evaluation", {})
        min_score = e_cfg.get("min_score", 0.0)
        tolerance = 0.02

        deploy = True
        status = "completed"

        if evaluator and new_score < baseline_score - tolerance:
            if e_cfg.get("auto_rollback", True):
                deploy = False
                status = "rejected"
                print(f"[Training] Model REJECTED: score dropped {baseline_score:.4f} -> {new_score:.4f}")
                # Clean up rejected model
                shutil.rmtree(str(self.models_dir / new_version), ignore_errors=True)

        if deploy:
            # Save the model properly
            if not Path(output_path).exists():
                model.save(output_path)

            # Update database
            set_state("current_model_version", new_version)

            with get_connection() as conn:
                conn.execute("""
                    INSERT INTO model_versions
                    (version, model_path, parent_version, training_pairs_count, eval_score, is_deployed, metadata_json)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                """, (
                    new_version, output_path, current_version,
                    len(all_examples), new_score,
                    json.dumps({
                        "epochs": epochs, "lr": learning_rate,
                        "loss": loss_name, "replay_count": replay_count
                    })
                ))

                # Mark old version as not deployed
                conn.execute(
                    "UPDATE model_versions SET is_deployed = 0 WHERE version != ?",
                    (new_version,)
                )

                # Mark training pairs as used
                conn.execute("UPDATE training_pairs SET used_in_training = 1 WHERE used_in_training = 0")

            print(f"[Training] Model {new_version} DEPLOYED successfully")

        # Log the training run
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO training_log
                (version, status, pairs_used, epochs, eval_score, duration_seconds, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                new_version if deploy else f"{new_version}_rejected",
                status, len(all_examples), epochs, new_score, duration,
                datetime.fromtimestamp(start_time).isoformat(),
                datetime.now().isoformat()
            ))

        total_runs = int(get_state("total_training_runs") or "0") + 1
        set_state("total_training_runs", str(total_runs))
        set_state("last_training_time", datetime.now().isoformat())

        return {
            "status": status,
            "version": new_version if deploy else current_version,
            "baseline_score": baseline_score,
            "new_score": new_score,
            "pairs_used": len(all_examples),
            "duration_seconds": round(duration, 2),
            "deployed": deploy
        }

    def rollback(self, to_version: str = None) -> Dict:
        """Rollback to a previous model version."""
        with get_connection() as conn:
            if to_version:
                row = conn.execute(
                    "SELECT version, model_path FROM model_versions WHERE version = ?", (to_version,)
                ).fetchone()
            else:
                current = get_state("current_model_version")
                row = conn.execute(
                    "SELECT parent_version FROM model_versions WHERE version = ?", (current,)
                ).fetchone()
                if row and row["parent_version"]:
                    row = conn.execute(
                        "SELECT version, model_path FROM model_versions WHERE version = ?", (row["parent_version"],)
                    ).fetchone()
                else:
                    row = None

            if not row:
                return {"status": "failed", "reason": "Version not found or no parent version."}

            new_version = row["version"]
            
            # Update database
            set_state("current_model_version", new_version)
            
            conn.execute(
                "UPDATE model_versions SET is_deployed = 1 WHERE version = ?",
                (new_version,)
            )
            conn.execute(
                "UPDATE model_versions SET is_deployed = 0 WHERE version != ?",
                (new_version,)
            )
            
            return {"status": "success", "rolled_back_to": new_version}
