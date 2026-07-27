from typing import List

class BaseProcessor:
    """Base interface for document processors."""
    
    @classmethod
    def can_process(cls, filepath: str) -> bool:
        """Returns True if this processor can handle the given file."""
        raise NotImplementedError

    @classmethod
    def extract(cls, filepath: str) -> str:
        """Extracts text from the file."""
        raise NotImplementedError
