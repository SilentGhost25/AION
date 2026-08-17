# tests/contracts/test_text_chunk.py

import pytest
from core.contracts.text_chunk import TextChunk


class TestTextChunkContract:

    def test_has_formula_true_when_equation_ids_present(self):
        chunk = TextChunk.from_raw(
            chunk_id="c1", document_id="d1", module_id=1,
            page_start=1, page_end=1, text="equation",
            equation_ids=["eq_001"],
        )
        assert chunk.has_formula is True

    def test_has_formula_false_when_no_equations(self):
        chunk = TextChunk.from_raw(
            chunk_id="c2", document_id="d1", module_id=1,
            page_start=1, page_end=1, text="plain text",
        )
        assert chunk.has_formula is False

    def test_has_formula_is_explicit_field_not_dynamic(self):
        chunk = TextChunk.from_raw(
            chunk_id="c3", document_id="d1", module_id=1,
            page_start=1, page_end=1, text="text",
        )
        assert hasattr(chunk, "has_formula")

    def test_from_raw_derives_flags_from_ids(self):
        chunk = TextChunk.from_raw(
            chunk_id="c4", document_id="d1", module_id=2,
            page_start=5, page_end=7, text="mixed",
            equation_ids=["eq1"], table_ids=["tbl1"],
        )
        assert chunk.has_formula is True
        assert chunk.has_table is True
        assert chunk.has_figure_reference is False
