import docx
from .base_processor import BaseProcessor

class DocxProcessor(BaseProcessor):
    @classmethod
    def can_process(cls, filepath: str) -> bool:
        return filepath.lower().endswith('.docx')

    @classmethod
    def extract(cls, filepath: str) -> str:
        doc = docx.Document(filepath)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        return text
