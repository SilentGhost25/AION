import json
from pathlib import Path
from .provider import DocumentProvider
from .document_model import AcademicDocument

class OpenDataLoaderProvider(DocumentProvider):
    """
    Implementation of the DocumentProvider that uses OpenDataLoader
    to extract rich intelligence from documents.
    """
    
    def __init__(self, api_key: str = None, endpoint: str = None):
        self.api_key = api_key
        self.endpoint = endpoint

    def load(self, file_path: str) -> AcademicDocument:
        """
        Loads a document via OpenDataLoader and converts it to an AcademicDocument.
        Note: Currently mocks the API call for demonstration.
        """
        # In a real scenario, we'd send the file to OpenDataLoader API and parse the response.
        # For now, we return a mock AcademicDocument.
        
        doc = AcademicDocument(source_path=file_path)
        doc.metadata = {
            "title": Path(file_path).stem,
            "provider": "OpenDataLoader"
        }
        
        # If it's a txt file, read it to support tests
        content = "This is a mock document extracted via OpenDataLoader."
        title = doc.metadata['title']
        if file_path.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.read().strip().split('\n', 1)
                    if len(lines) == 2:
                        title = lines[0].strip()
                        content = lines[1].strip()
                    else:
                        content = lines[0]
            except Exception:
                pass

        # Mocking markdown extraction
        doc.markdown = f"# {title}\n\n{content}"
        
        # Mocking TOC
        doc.toc = [
            {"level": 1, "title": title, "page": 1}
        ]
        
        from document_intelligence.document_model import Section
        # Mocking sections
        doc.sections = [
            Section(title=title, content=content)
        ]
        
        return doc
