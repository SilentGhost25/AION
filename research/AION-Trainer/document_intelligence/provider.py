from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
from .document_model import AcademicDocument

class DocumentProvider(ABC):
    """
    Base interface for all document intelligence providers.
    Allows swapping the underlying extraction engine (e.g., OpenDataLoader vs legacy parsers)
    without affecting the rest of AION.
    """
    
    @abstractmethod
    def load(self, file_path: str) -> AcademicDocument:
        """
        Extracts intelligence from the given file and returns a structured AcademicDocument.
        """
        pass
