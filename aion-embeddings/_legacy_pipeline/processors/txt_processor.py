import os
from .base_processor import BaseProcessor

class TxtProcessor(BaseProcessor):
    @classmethod
    def can_process(cls, filepath: str) -> bool:
        return filepath.lower().endswith(('.txt', '.md', '.csv', '.json'))

    @classmethod
    def extract(cls, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
