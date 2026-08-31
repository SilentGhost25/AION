# core/validation/math_validator.py

import re
import subprocess
import logging
from core.generation.output_schema import QuestionOutput, MathBlock
from core.validation.common import CheckResult, RetryAction

LOG = logging.getLogger(__name__)


class MathRenderFailure(Exception):
    """Raised when KaTeX math rendering fails."""
    pass


class KaTeXAvailabilityGate:
    """KaTeX availability and rendering gate."""

    _verified = False
    _cmd = None

    @classmethod
    def verify(cls) -> bool:
        """Verify KaTeX by actually rendering a tiny expression."""
        if cls._verified and cls._cmd:
            return True

        import subprocess

        candidates = [
            ["npx", "--no-install", "katex"],
            ["npx", "--yes", "katex"],
        ]

        errors = []

        for cmd in candidates:
            try:
                res = subprocess.run(
                    cmd,
                    input="x",
                    text=True,
                    capture_output=True,
                    timeout=20,
                    shell=False,
                )

                if res.returncode == 0 and "katex" in res.stdout.lower():
                    cls._verified = True
                    cls._cmd = cmd
                    LOG.info(
                        "KaTeXAvailabilityGate verified with: %s",
                        " ".join(cmd),
                    )
                    return True

                errors.append(
                    f"{' '.join(cmd)} -> exit={res.returncode}, "
                    f"stderr={res.stderr.strip()}"
                )

            except Exception as exc:
                errors.append(f"{' '.join(cmd)} -> {exc}")

        cls._verified = False
        cls._cmd = None
        LOG.error("KaTeX probe failed: %s", " | ".join(errors))
        return False

    @classmethod
    def render(cls, latex: str, display_mode: bool = False) -> str:
        """Render one LaTeX expression using KaTeX."""
        import subprocess

        if not isinstance(latex, str) or not latex.strip():
            raise MathRenderFailure("LaTeX expression is empty")

        if not cls.verify():
            raise MathRenderFailure("KaTeX executable unavailable")

        try:
            cmd = list(cls._cmd)
            cmd.extend(["--strict", "warn"])
            if display_mode:
                cmd.append("--display-mode")

            res = subprocess.run(
                cmd,
                input=latex.replace("\\t{","\\text{").replace("\\owtie","\\bowtie").replace("\\ext","\\text").replace("\\newline",""),
                text=True,
                capture_output=True,
                timeout=20,
                shell=False,
            )
        except Exception as exc:
            raise MathRenderFailure(
                f"KaTeX subprocess failed: {exc}"
            ) from exc

        if res.returncode != 0:
            message = (res.stderr or res.stdout or "unknown KaTeX error").strip().split("\n")[0][:200]
            raise MathRenderFailure(
                f"KaTeX render failed: {message}"
            )

        rendered = res.stdout.strip()

        if not rendered:
            raise MathRenderFailure("KaTeX returned empty output")

        return rendered

    @staticmethod
    def _latex_to_unicode_fallback(latex: str) -> str:
        """Convert common LaTeX symbols to Unicode when KaTeX fails."""
        import re
        s = latex
        # Common relational algebra symbols
        s = s.replace(r'\sigma', 'σ')
        s = s.replace(r'\pi', 'π')
        s = s.replace(r'\bowtie', '⋈')
        s = s.replace(r'\cup', '∪')
        s = s.replace(r'\cap', '∩')
        s = s.replace(r'\in', '∈')
        s = s.replace(r'\notin', '∉')
        s = s.replace(r'\forall', '∀')
        s = s.replace(r'\exists', '∃')
        s = s.replace(r'\leq', '≤')
        s = s.replace(r'\geq', '≥')
        s = s.replace(r'\neq', '≠')
        s = s.replace(r'\rightarrow', '→')
        s = s.replace(r'\leftarrow', '←')
        s = s.replace(r'\sum', 'Σ')
        s = s.replace(r'\int', '∫')
        s = s.replace(r'\infty', '∞')
        s = s.replace(r'\partial', '∂')
        s = s.replace(r'\nabla', '∇')
        s = s.replace(r'\subseteq', '⊆')
        s = s.replace(r'\supseteq', '⊇')
        s = s.replace(r'\emptyset', '∅')
        s = s.replace(r'\times', '×')
        s = s.replace(r'\div', '÷')
        s = s.replace(r'\pm', '±')
        s = s.replace(r'\approx', '≈')
        s = s.replace(r'\equiv', '≡')
        s = s.replace(r'\cdots', '…')
        s = s.replace(r'\ldots', '…')
        # Remove remaining backslash commands (show as-is)
        s = re.sub(r'\\[a-zA-Z]+', '', s)
        # Clean up braces
        s = s.replace('{', '').replace('}', '')
        return s.strip()




def validate_math_consistency(output: QuestionOutput) -> CheckResult:
    """
    H3 — Missing or orphan [MATH:id] references are hard failures.
    Called after schema validation (which also checks this).
    """
    referenced = set(re.findall(r"\[MATH:([^\]]+)\]", output.question_text))
    declared   = {b.block_id for b in output.math_blocks}

    orphan = referenced - declared
    if orphan:
        return CheckResult.fail(
            "MATH_PLACEHOLDER_UNRESOLVED",
            f"[MATH:...] references not in declared blocks: {orphan}. "
            f"Final question would contain broken placeholders.",
            action=RetryAction.REGENERATE
        )

    unused = declared - referenced
    if unused:
        return CheckResult.fail(
            "MATH_BLOCK_UNREFERENCED",
            f"Declared MathBlocks never appear in question text: {unused}. "
            f"Qwen declared math it didn't use.",
            action=RetryAction.REGENERATE
        )

    return CheckResult.pass_()


def validate_math_block_with_render(block: MathBlock) -> CheckResult:
    """Renderer-aware validation. KaTeX is mandatory — no silent skip."""

    if "\ufffd" in block.latex or "\x00" in block.latex:
        return CheckResult.critical(
            "MATH_CORRUPTED",
            f"Block {block.block_id} contains corruption chars"
        )

    if not block.latex.strip():
        return CheckResult.fail("MATH_EMPTY", f"Block {block.block_id} is empty")

    try:
        block.latex.encode("utf-8").decode("utf-8")
    except Exception as e:
        return CheckResult.critical("MATH_ENCODING_FAILURE", str(e))

    # Brace balance
    depth = 0
    for ch in block.latex:
        if ch == "{": depth += 1
        elif ch == "}": depth -= 1
        if depth < 0:
            return CheckResult.fail("MATH_UNBALANCED_BRACES",
                                    f"Block {block.block_id}: unmatched }}")
    if depth != 0:
        return CheckResult.fail("MATH_UNCLOSED_BRACES",
                                f"Block {block.block_id}: {depth} unclosed braces")

    # \\frac structure
    frac_count = block.latex.count(r"\frac")
    frac_ok    = len(re.findall(r"\\frac\s*\{[^}]*\}\s*\{[^}]*\}", block.latex))
    if frac_count > 0 and frac_ok < frac_count:
        return CheckResult.fail("MATH_INCOMPLETE_FRAC",
                                f"Block {block.block_id}: \\frac missing argument")

    # KaTeX render (mandatory — not optional)
    try:
        rendered = KaTeXAvailabilityGate.render(
            block.latex, display_mode=block.display_mode
        )
        if "katex-error" in rendered.lower():
            return CheckResult.fail("MATH_RENDER_ERROR",
                                    f"Block {block.block_id}: KaTeX error class")
        if len(rendered.strip()) < 20:
            return CheckResult.fail("MATH_RENDER_EMPTY",
                                    f"Block {block.block_id}: render too short")
    except MathRenderFailure as e:
        return CheckResult.fail("MATH_RENDER_FAILURE",
                                f"Block {block.block_id}: {e}")

    return CheckResult.pass_()
