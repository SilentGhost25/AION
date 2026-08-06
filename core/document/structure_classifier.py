"""
Document Structure Classifier — Block-level classification
Per audit: Replace simple header detection with block-level classification:
Heading, Formula, Caption, Algorithm, Procedure, Example, Exercise, Concept, Definition, Table, Diagram text, Footnote

Then extraction becomes dramatically easier.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class DocumentBlock:
    text: str
    block_type: str  # Heading, Formula, Caption, Algorithm, Procedure, Example, Exercise, Concept, Definition, Table, Diagram, Footnote
    confidence: float
    metadata: Dict = None

class StructureClassifier:
    """Classifies every block of document text."""
    
    def classify(self, text: str) -> List[DocumentBlock]:
        blocks = []
        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            block_type, conf = self._classify_block(para)
            blocks.append(DocumentBlock(text=para, block_type=block_type, confidence=conf))
        
        return blocks
    
    def _classify_block(self, text: str) -> Tuple[str, float]:
        # Check for formula (contains math symbols)
        if re.search(r"[=∑∫√∞∂∆\^\*\/\\]|\\frac|\\sum|T\(n\)|2\^|O\(n\)", text):
            if re.search(r"where|is defined|represents", text, re.I):
                return "Definition", 0.8
            return "Formula", 0.9
        
        # Check for heading (short, no period, title case or all caps)
        lines = text.splitlines()
        if len(lines) == 1 and 5 <= len(text) <= 80 and not text.strip().endswith("."):
            if text.isupper() or re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}$", text.strip()):
                if re.match(r"^(Module|Chapter|Unit|Section|Heading)", text, re.I):
                    return "Heading", 0.95
                return "Heading", 0.85
        
        # Check for caption
        if re.match(r"^(Figure|Table|Diagram|Graph)\s+\d+", text, re.I):
            return "Caption", 0.95
        
        # Check for algorithm (contains steps, pseudocode keywords)
        if re.search(r"\b(for|while|if|then|else|procedure|function|algorithm)\b.*\b(do|begin|end|return)\b", text, re.I | re.S):
            return "Algorithm", 0.9
        
        # Check for procedure (numbered steps)
        if re.search(r"^\s*\d+[\.\)]\s+[A-Z]", text, re.M) and len(re.findall(r"^\s*\d+[\.\)]", text, re.M)) >= 2:
            return "Procedure", 0.85
        
        # Check for example
        if re.match(r"^(Example|For example|Consider|Suppose)\b", text, re.I):
            return "Example", 0.9
        
        # Check for exercise/question
        if re.match(r"^(Exercise|Question|Problem|Assignment)\b", text, re.I) or re.search(r"\?\s*$", text):
            if len(text.split()) < 50:
                return "Exercise", 0.85
        
        # Check for definition
        if re.search(r"\bis defined as\b|\brefers to\b|\bdenotes\b|\bmeans\b|\bknown as\b", text, re.I):
            return "Definition", 0.9
        
        # Check for table (contains | or tab-separated)
        if text.count("|") >= 2 or text.count("\t") >= 2:
            return "Table", 0.85
        
        # Check for diagram text (contains figure references)
        if re.search(r"\b(figure|diagram|graph|shows|depicts|illustrates)\b.*\b(below|above|following)\b", text, re.I):
            return "Diagram", 0.75
        
        # Check for footnote (short, starts with * or number)
        if re.match(r"^(\*|\d+)\s+[a-z]", text) and len(text) < 100:
            return "Footnote", 0.7
        
        # Default: concept
        return "Concept", 0.6
    
    def get_concept_blocks(self, blocks: List[DocumentBlock]) -> List[DocumentBlock]:
        """Filter to only concept/definition blocks for extraction."""
        return [b for b in blocks if b.block_type in ["Concept", "Definition", "Procedure", "Example", "Algorithm"]]
    
    def get_stats(self, blocks: List[DocumentBlock]) -> Dict:
        from collections import Counter
        counts = Counter(b.block_type for b in blocks)
        return dict(counts)
