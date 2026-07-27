from .pdf_processor import PDFProcessor
from .docx_processor import DocxProcessor
from .txt_processor import TxtProcessor

PROCESSORS = [
    PDFProcessor,
    DocxProcessor,
    TxtProcessor,
]

def get_processor(filepath: str):
    for processor_class in PROCESSORS:
        if processor_class.can_process(filepath):
            return processor_class
    return None
