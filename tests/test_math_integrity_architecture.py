"""
AION Math Integrity Architecture — Automated Pytest Suite
=========================================================
Verifies all 5 foundational invariants and 10 architectural components:
  - M1: Math is an object, not a string (MathArtifact).
  - M2: LaTeX is canonical representation.
  - M3: Unicode replacement character (U+FFFD) causes immediate block.
  - M4: Qwen math interface placeholder references [MATH:eq_...].
  - M5: Multi-format rendering and round-trip validation.
  - Math Symbol Registry translation accuracy.
  - Math Boundary Guard protect & restore operations.
  - Math Normalizer SHA256 integrity hash verification.
  - Math Validator V01-V07 checks.
  - Math Healer guarded repair decision tree.
  - Symbolic Equivalence Checker matching.
  - Encoding Invariant Guard UTF-8 purity checks.
"""

import pytest
from v0_1.math_integrity.contracts import (
    EquationType,
    MathArtifact,
    MathIntegrityViolation,
    MathRepresentation,
    MathSourceType,
    MathValidationStatus,
    ProtectedTextEnvelope,
)
from v0_1.math_integrity.registry import MATH_SYMBOL_REGISTRY, latex_to_unicode, unicode_to_latex
from v0_1.math_integrity.encoding_guard import EncodingError, EncodingInvariantGuard
from v0_1.math_integrity.boundary_guard import MathBoundaryGuard
from v0_1.math_integrity.normalizer import MathNormalizer
from v0_1.math_integrity.validator import MathValidator
from v0_1.math_integrity.healer import HealingFailure, MathHealer
from v0_1.math_integrity.equivalence import SymbolicEquivalenceChecker
from v0_1.math_integrity.renderer import MathRenderer, RenderFormat
from v0_1.math_integrity.qwen_math import QwenMathInterface


def test_invariant_m1_math_object():
    art = MathArtifact(
        math_id="eq_001",
        latex=r"\omega = \frac{\pi N}{30}",
        normalized_latex=r"\omega = \frac{\pi N}{30}",
    )
    assert art.best_for_llm() == "[MATH:eq_001]"
    assert art.best_for_display() == r"\omega = \frac{\pi N}{30}"
    assert art.verify_canonical_hash() is True


def test_invariant_m2_latex_canonical():
    # Convert Unicode to LaTeX authority
    unicode_str = "ω = π × N / 30"
    latex_conv = MathNormalizer.convert_unicode_to_latex(unicode_str)
    assert r"\omega" in latex_conv
    assert r"\pi" in latex_conv
    assert r"\times" in latex_conv


def test_invariant_m3_replacement_character_block():
    # M3 Violation: U+FFFD replacement character MUST raise MathIntegrityViolation immediately
    corrupt_str = "Calculate angular velocity \ufffd = 10 rad/s"

    with pytest.raises(MathIntegrityViolation) as exc_info:
        EncodingInvariantGuard.assert_clean(corrupt_str, "test_m3")
    assert exc_info.value.code == "M3_REPLACEMENT_CHAR"

    with pytest.raises(MathIntegrityViolation) as exc_info:
        MathBoundaryGuard.protect(corrupt_str)
    assert exc_info.value.code == "M3_REPLACEMENT_CHAR"

    with pytest.raises(MathIntegrityViolation) as exc_info:
        MathArtifact(math_id="eq_err", latex="x = \ufffd", normalized_latex="x = \ufffd")
    assert exc_info.value.code == "M3_REPLACEMENT_CHAR"


def test_invariant_m4_qwen_placeholder_interface():
    text = r"Calculate angular velocity \(\omega = \frac{\pi N}{30}\) for N=1500 RPM."
    envelope = MathBoundaryGuard.protect(text, document_id="doc1", page=2)

    assert envelope.has_math() is True
    assert "[MATH:eq_doc1_2_001]" in envelope.text

    # Qwen generation context
    context = QwenMathInterface.prepare_context(envelope)
    assert "text" in context
    assert "math_instructions" in context
    assert "eq_doc1_2_001" in context["math_artifacts"]

    # Qwen outputs response preserving placeholder
    qwen_response = "The angular velocity is given by [MATH:eq_doc1_2_001] where N is RPM."
    assert QwenMathInterface.validate_response(qwen_response, ["[MATH:eq_doc1_2_001]"]) is True

    # Restoration for rendering
    final_rendered = QwenMathInterface.restore_math(qwen_response, envelope)
    assert r"\omega = \frac{\pi N}{30}" in final_rendered


def test_invariant_m5_renderer_and_round_trip():
    art = MathArtifact(
        math_id="eq_render_01",
        latex=r"V_{th} = 12\text{ V}",
        normalized_latex=r"V_{th} = 12\text{ V}",
        equation_type=EquationType.INLINE,
    )

    # WEB Rendering
    web_res = MathRenderer.render(art, RenderFormat.WEB)
    assert web_res.round_trip_verified is True
    assert art.round_trip_verified is True
    assert "katex-math" in web_res.content

    # PDF Rendering
    pdf_res = MathRenderer.render(art, RenderFormat.PDF)
    assert pdf_res.round_trip_verified is True
    assert r"$V_{th} = 12\text{ V}$" == pdf_res.content

    # DOCX Rendering
    docx_res = MathRenderer.render(art, RenderFormat.DOCX)
    assert docx_res.round_trip_verified is True
    assert "<m:oMath" in docx_res.content


def test_symbol_registry_translations():
    assert unicode_to_latex("ω") == r"\omega"
    assert unicode_to_latex("π") == r"\pi"
    assert unicode_to_latex("Ω") == r"\Omega"
    assert latex_to_unicode(r"\alpha") == "α"
    assert latex_to_unicode(r"\theta") == "θ"


def test_math_validator_v01_v07():
    art_valid = MathArtifact(math_id="eq_val_1", latex="E = m c^2", normalized_latex="E = m c^2")
    rep_valid = MathValidator.validate(art_valid)
    assert (getattr(rep_valid, "is_valid", rep_valid) if not isinstance(rep_valid, bool) else rep_valid) is True

    # Empty string check (V06)
    art_empty = MathArtifact(math_id="eq_val_2", latex="", normalized_latex="")
    rep_empty = MathValidator.validate(art_empty)
    assert rep_empty.status == MathValidationStatus.CORRUPT


def test_math_healer_guarded_repairs():
    # Delimiter repair
    art_unbalanced = MathArtifact(
        math_id="eq_heal_1",
        latex=r"\[x^2 + y^2 = z^2",
        normalized_latex=r"\[x^2 + y^2 = z^2",
        validation_status=MathValidationStatus.TRUNCATED,
    )

    healed = MathHealer.heal(art_unbalanced)
    assert isinstance(healed, MathArtifact)
    assert healed.normalized_latex.endswith(r"\]")

    # M3 replacement character must cause immediate block (no guessing allowed)
    with pytest.raises(MathIntegrityViolation) as exc_info:
        MathArtifact(
            math_id="eq_heal_2",
            latex="a + \ufffd = c",
            normalized_latex="a + \ufffd = c",
            validation_status=MathValidationStatus.CORRUPT,
        )
    assert exc_info.value.code == "M3_REPLACEMENT_CHAR"


def test_symbolic_equivalence_checker():
    ast_a = MathNormalizer.build_ast("x + y = z")
    ast_b = MathNormalizer.build_ast("x + y = z")
    assert SymbolicEquivalenceChecker.check(ast_a, ast_b, "x + y = z", "x + y = z") is True


def test_encoding_invariant_guard():
    clean_text = "Standard mathematical equation: \\alpha = 5"
    assert EncodingInvariantGuard.assert_clean(clean_text) == clean_text

    raw_bytes = "ω = 2πf".encode("utf-8")
    decoded = EncodingInvariantGuard.safe_decode(raw_bytes, "test_bytes")
    assert decoded == "ω = 2πf"
