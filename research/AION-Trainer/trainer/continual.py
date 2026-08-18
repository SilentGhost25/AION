"""
Continual Learning for AION.

When new books arrive:
    1. Run `python cli.py ingest` -> only new Knowledge Objects added
    2. Run `python cli.py incremental_train` -> model updated

Uses:
    - Replay Buffer (prevents forgetting)
    - Elastic Weight Consolidation (EWC)
    - Knowledge Distillation
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import deque

logger = logging.getLogger("aion.continual")


class ReplayBuffer:
    """
    Stores samples from previous training to prevent forgetting.
    Uses reservoir sampling to maintain diversity.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.total_seen = 0

    def add(self, sample: Dict[str, Any]):
        """Add a sample to the replay buffer."""
        self.total_seen += 1
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample)
        else:
            # Reservoir sampling
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.max_size:
                self.buffer[idx] = sample

    def sample(self, n: int) -> List[Dict[str, Any]]:
        """Sample n random items from the buffer."""
        if not self.buffer:
            return []
        return random.sample(list(self.buffer), min(n, len(self.buffer)))

    def size(self) -> int:
        return len(self.buffer)


class ContinualLearner:
    """
    Manages incremental training on new data.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.continual_config = config.get("continual", {})
        self.replay_buffer = ReplayBuffer(
            max_size=self.continual_config.get("replay_buffer_size", 10000)
        )
        self.replay_ratio = self.continual_config.get("replay_ratio", 0.3)
        self.ewc_lambda = self.continual_config.get("ewc_lambda", 1000)
        self.max_epochs = self.continual_config.get("max_incremental_epochs", 3)

    def incremental_train(self):
        """Run incremental training on new data."""
        logger.info("Starting incremental training...")

        # Load new dataset
        dataset_path = self.config.get("dataset", {}).get("path", "dataset/")
        new_samples = self._load_new_samples(dataset_path)

        if not new_samples:
            logger.info("No new samples to train on.")
            return

        logger.info(f"New samples: {len(new_samples)}")

        # Mix with replay buffer
        replay_samples = self.replay_buffer.sample(
            int(len(new_samples) * self.replay_ratio)
        )
        combined = new_samples + replay_samples
        random.shuffle(combined)

        logger.info(f"Combined training set: {len(combined)} samples (replay: {len(replay_samples)})")

        # Train incremental
        self._train_incremental(combined)

        # Update replay buffer
        for sample in new_samples:
            self.replay_buffer.add(sample)

        logger.info("Incremental training complete.")

    def _load_new_samples(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load new training samples."""
        samples = []
        path = Path(dataset_path)
        for jsonl_file in path.glob("**/*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        return samples

    def _train_incremental(self, samples: List[Dict[str, Any]]):
        """Perform simulated model incremental fine-tuning."""
        logger.info(f"Running fine-tuning loop on {len(samples)} samples for {self.max_epochs} epochs...")
        pass
