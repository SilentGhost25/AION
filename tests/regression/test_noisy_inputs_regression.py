"""
Noisy Inputs OCR Regression Test
"""

from v0_1.vre import FigureClassification, FigureExtractionResult, QuantityParser, VKOC, VKOValidator


def test_noisy_ocr_quantity_repair_invariant():
    raw_ocr = "V1 10O R1 20O R2 30O"
    clean_val = QuantityParser.normalize_symbol(raw_ocr)

    assert "Ω" in clean_val

    ext = FigureExtractionResult(status="SUCCESS", image_path="fake.png", page_number=1, bbox=(0, 0, 100, 100), confidence=0.85, extraction_method="OCR")
    cls = FigureClassification(domain="ECE", figure_class="CIRCUIT_RESISTIVE", operations=["EQUIVALENT_RESISTANCE"], confidence=0.85)
    vko = VKOC.build(ext, cls)

    valid, errors = VKOValidator.validate(vko)
    assert valid is True
    assert len(errors) == 0
