import hashlib
from datetime import datetime
from typing import List
import re
from .contracts.learning import TrainingPair, TrainingDataset
import logging

logger = logging.getLogger(__name__)

class TrainingPairGenerator:
    """
    Converts extracted questions into training pairs.
    """
    
    def __init__(self):
        self.pairs: List[TrainingPair] = []
    
    def from_question_extraction(self, questions: List[dict], subject: str) -> List[TrainingPair]:
        pairs = []
        for i, q in enumerate(questions):
            if 'answer' not in q or not q['answer']:
                continue
            pair_id = hashlib.md5(f"{subject}_{i}_{q['text']}".encode()).hexdigest()
            bloom = self._detect_bloom(q['text'])
            pair = TrainingPair(
                pair_id=pair_id,
                anchor=q['text'],
                positive=q['answer'],
                negative=None,
                subject=subject,
                bloom_level=bloom,
                source="question_paper",
                confidence=0.95
            )
            pairs.append(pair)
            logger.info(f"Generated pair: {pair_id}")
        return pairs
    
    def from_topic_cooccurrence(self, text: str, subject: str) -> List[TrainingPair]:
        pairs = []
        sentences = re.split(r'[.!?]+', text)
        from nltk.tokenize import word_tokenize
        from nltk.tag import pos_tag
        
        for i, sent in enumerate(sentences):
            if len(sent.strip()) < 20:
                continue
            tokens = word_tokenize(sent)
            pos_tags = pos_tag(tokens)
            concepts = [word for word, pos in pos_tags if pos.startswith('NN')]
            
            for j in range(len(concepts) - 1):
                pair_id = hashlib.md5(f"{subject}_{i}_{j}".encode()).hexdigest()
                pair = TrainingPair(
                    pair_id=pair_id,
                    anchor=f"What is {concepts[j]}?",
                    positive=sent,
                    negative=None,
                    subject=subject,
                    bloom_level="L2",
                    source="textbook",
                    confidence=0.6
                )
                pairs.append(pair)
        return pairs
    
    def _detect_bloom(self, question_text: str) -> str:
        keywords = {
            "L1": ["define", "list", "name", "state"],
            "L2": ["explain", "describe", "discuss", "summarize"],
            "L3": ["apply", "solve", "calculate", "demonstrate"],
            "L4": ["analyze", "examine", "break down", "differentiate"],
            "L5": ["evaluate", "judge", "critique", "argue"],
            "L6": ["design", "create", "propose", "derive"]
        }
        for level, words in keywords.items():
            if any(word in question_text.lower() for word in words):
                return level
        return "L2"
    
    def build_dataset(self, all_pairs: List[TrainingPair]) -> TrainingDataset:
        dataset_id = hashlib.md5(datetime.now().isoformat().encode()).hexdigest()
        sources = {}
        for pair in all_pairs:
            sources[pair.source] = sources.get(pair.source, 0) + 1
        return TrainingDataset(
            dataset_id=dataset_id,
            pairs=all_pairs,
            total_pairs=len(all_pairs),
            created_at=datetime.now().isoformat(),
            sources=sources
        )
