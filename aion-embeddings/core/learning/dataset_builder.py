import hashlib
from datetime import datetime
from typing import List
from .contracts.learning import TrainingPair, TrainingDataset
from .replay_buffer import ReplayBuffer
import logging

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """
    Assembles training datasets by combining new pairs and replay buffer pairs.
    """
    
    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer
    
    def build_training_dataset(
        self,
        new_pairs: List[TrainingPair],
        subject: str = None,
        replay_ratio: float = 0.3
    ) -> TrainingDataset:
        
        logger.info(f"Building dataset: {len(new_pairs)} new pairs, {replay_ratio*100}% replay")
        
        if len(new_pairs) == 0:
            replay_count = 0
        else:
            replay_count = int(len(new_pairs) * replay_ratio / (1 - replay_ratio))
        
        replay_pairs = self.replay_buffer.sample(replay_count, subject=subject)
        logger.info(f"Sampled {len(replay_pairs)} replay pairs")
        
        all_pairs = new_pairs + replay_pairs
        
        import random
        random.shuffle(all_pairs)
        
        dataset = TrainingDataset(
            dataset_id=self._generate_dataset_id(new_pairs),
            pairs=all_pairs,
            total_pairs=len(all_pairs),
            created_at=datetime.now().isoformat(),
            sources={p.source: 1 for p in all_pairs}
        )
        
        logger.info(f"Final dataset: {len(all_pairs)} pairs")
        return dataset
    
    def _generate_dataset_id(self, pairs: List[TrainingPair]) -> str:
        import json
        content = json.dumps(sorted([p.pair_id for p in pairs]))
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_balance_report(self) -> dict:
        return self.replay_buffer.get_statistics()
