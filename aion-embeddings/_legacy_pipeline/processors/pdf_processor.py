import pdfplumber
from .base_processor import BaseProcessor

class PDFProcessor(BaseProcessor):
    @classmethod
    def can_process(cls, filepath: str) -> bool:
        return filepath.lower().endswith('.pdf')

    @classmethod
    def extract(cls, filepath: str) -> str:
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    if any(word in page_text.lower() for word in ["theorem", "proof", "formula"]):
                        page_text = f"[TECHNICAL] {page_text}"
                    if "algorithm" in page_text.lower():
                        page_text = f"[ALGORITHM] {page_text}"
                    text += page_text + "\n"
        return text
