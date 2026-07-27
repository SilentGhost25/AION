"""
AION Module: Concept Learner
Maturity:    v0.1 — RULE-BASED (paragraph length & key phrase heuristic)
Upgrades to: Academic Genome Builder (Neural Concept Extractor + Graph Relation Mapper)
             - populate definition_dna, relationship_dna, bloom_dna
Contract:    CleanedDocument -> list[Concept] (see schemas.py)
"""

from typing import List
from .schemas import CleanedDocument, Concept
from .memory import ConceptMemoryStore
from .content_validator import validate_chunk, clean_chunk


class Learner:
    def __init__(self, memory_store: ConceptMemoryStore = None):
        self.memory_store = memory_store or ConceptMemoryStore()

    def learn(self, document: CleanedDocument) -> List[Concept]:
        """
        Parses cleaned document text into CONCEPT-SIZED chunks (80-400 words),
        not individual lines. This prevents noise fragments from becoming concepts.
        """
        raw_paragraphs = document.clean_text.split("\n\n")

        MIN_CONCEPT_WORDS = 80
        MAX_CONCEPT_WORDS = 400

        chunks = []
        current_chunk_lines = []
        current_word_count = 0

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            para_words = len(para.split())

            if para_words >= MIN_CONCEPT_WORDS:
                if current_chunk_lines and current_word_count >= MIN_CONCEPT_WORDS:
                    chunks.append("\n".join(current_chunk_lines))
                    current_chunk_lines = []
                    current_word_count = 0

                chunks.append(para)
                continue

            current_chunk_lines.append(para)
            current_word_count += para_words

            if current_word_count >= MAX_CONCEPT_WORDS:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = []
                current_word_count = 0

        if current_chunk_lines and current_word_count >= MIN_CONCEPT_WORDS:
            chunks.append("\n".join(current_chunk_lines))

        extracted_concepts = []
        for chunk_text in chunks:
            p_str = clean_chunk(chunk_text)
            quality = validate_chunk(p_str)

            if quality.is_valid and len(p_str.split()) >= MIN_CONCEPT_WORDS:
                raw_concept = Concept(
                    content=p_str,
                    confidence=quality.score,
                    source_dna=document.doc_id,
                    definition_dna=(
                        p_str if "defined as" in p_str.lower()
                        or "is a" in p_str.lower()
                        or "refers to" in p_str.lower()
                        else None
                    ),
                )
                persisted_concept = self.memory_store.upsert_concept(
                    raw_concept, save_now=False
                )
                extracted_concepts.append(persisted_concept)

        self.memory_store.save()
        return extracted_concepts
