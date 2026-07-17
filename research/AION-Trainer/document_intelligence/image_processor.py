from typing import Dict, Any, List
from .document_model import AcademicDocument

class ImageProcessor:
    """
    Processes and extracts figures/images from documents.
    """
    def extract_figures(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Mock implementation
        return []

    def enrich_document(self, doc: AcademicDocument, raw_data: Dict[str, Any]):
        doc.figures = self.extract_figures(raw_data)
