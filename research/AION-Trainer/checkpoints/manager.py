import os
import logging
from pathlib import Path

logger = logging.getLogger("aion.checkpoints")

class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_latest(self):
        mock_ckpt = self.checkpoint_dir / "aion_model_latest.pt"
        if not mock_ckpt.exists():
            with open(mock_ckpt, "w", encoding="utf-8") as f:
                f.write("mock_checkpoint")
        return str(mock_ckpt)

    def get_info(self, checkpoint_path):
        return {
            "version": "1.0.0-candidate",
            "epoch": 10,
            "score": 0.942
        }
