"""
AION PDF Artifact Filter
========================
Strips PDF object streams, xref tables, trailers, and binary noise.
"""

import re

def filter_pdf_artifacts(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\b\d+\s+\d+\s+obj\b.*?endobj', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'stream[\s\S]*?endstream', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'xref\s+\d+\s+\d+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'trailer\s*<<.*?>>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'startxref\s+\d+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
